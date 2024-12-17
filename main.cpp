/*
 * ff-rknn-v4l2-thread - Grab NV12 frames, AI Inference, Render it on screen
 * 2023 Alexander Finger <alex.mobigo@gmail.com>
 *
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include <SDL2/SDL.h>
#include <SDL2/SDL_thread.h>

#include <drm_fourcc.h>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <iostream>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavdevice/avdevice.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext_drm.h>
#include <libavutil/pixfmt.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#include <rga/RgaApi.h>
#include <rga/rga.h>

#include <SDL_FontCache.h>
#include <postprocess.h>
#include <rknn/rknn_api.h>

#define ALIGN(x, a) ((x) + (a - 1)) & (~(a - 1))
#define DRM_ALIGN(val, align) ((val + (align - 1)) & ~(align - 1))

#ifndef DRM_FORMAT_NV12_10
#define DRM_FORMAT_NV12_10 fourcc_code('N', 'A', '1', '2')
#endif

#ifndef DRM_FORMAT_NV15
#define DRM_FORMAT_NV15 fourcc_code('N', 'A', '1', '5')
#endif

#define MODEL_WIDTH 640
#define MODEL_HEIGHT 640

#define argt_a 36430 // -a
#define argt_b 36431 // -b
#define argt_c 36432 // -c
#define argt_i 36438 // -i
#define argt_x 36453 // -x
#define argt_y 36454 // -y
#define argt_l 36441 // -l
#define argt_m 36442 // -m
#define argt_o 36444 // -o
#define argt_t 36449 // -t
#define argt_e 36434 // -e
#define argt_f 36435 // -f
#define argt_r 36447 // -r
#define argt_d 36433 // -d
#define argt_p 36445 // -p
#define argt_s 36448 // -s

static unsigned int hash_me(char *str);

/* --- RKNN --- */
int channel = 3;
int width = MODEL_WIDTH;
int height = MODEL_HEIGHT;
unsigned char *model_data;
int model_data_size = 0;
char *model_name = NULL;
float scale_w = 1.0f; // (float)width / img_width;
float scale_h = 1.0f; // (float)height / img_height;
detect_result_group_t detect_result_group;
std::vector<float> out_scales;
std::vector<int32_t> out_zps;
rknn_context ctx;
rknn_input_output_num io_num;
rknn_input inputs[2];
rknn_tensor_attr output_attrs[256];
rknn_tensor_attr input_attrs[16];
size_t actual_size = 0;
const float nms_threshold = NMS_THRESH;
const float box_conf_threshold = BOX_THRESH;
char* labelsListFile = (char*)"/usr/share/model/coco_80_labels_list.txt";
/* --- SDL --- */
int alphablend;
int accur;
unsigned int obj2det;
int frameSize_rknn;
void *resize_buf;
Uint32 format;
SDL_Texture *texture;
SDL_Texture* captureTexture;
SDL_Window *window = NULL;
SDL_Renderer *renderer = NULL;

AVFormatContext *input_ctx = NULL;
AVStream *video = NULL;
int video_stream = 0;
AVCodecParameters *codecpar;
AVCodecContext *codec_ctx = NULL;
AVFormatContext *pOutputFormatCtx = nullptr;
AVStream* pOutStream = nullptr;
AVPacket* outpkt = av_packet_alloc();
AVCodecContext* pOutCodecCtx = nullptr;
const AVCodec* pOutputCodec = nullptr;
AVCodec *codec;
AVFrame *frame;
AVPacket pkt;
AVFrame* pFrameSDL;
AVFrame* pFrameYUV;
SwsContext* swsCtx;

int screen_width = 1024;
int screen_height = 600;
int screen_left = 0;
int screen_top = 0;
int frame_width = 1920;
int frame_height = 1080;
int v4l2;  // v4l2 h264
int rtsp;  // rtsp h264
int rtmp;  // flv h264
int http;  // flv h264
int delay; // ms
char *pixel_format;
char *sensor_frame_size;
char *sensor_frame_rate;

int y_size = frame_width * frame_height;
int uv_size = (frame_width * frame_height) / 2;

double avg_inference_time = 0.0, inference_time = 0.0;
float frmrate = 0.0;      // Measured frame rate
float avg_frmrate = 0.0;  // avg frame rate
float prev_frmrate = 0.0; // avg frame rate
Uint32 currtime;
Uint32 lasttime;
int loop_counter = 0;
const int frmrate_update = 30;

FC_Font *font_small;
FC_Font *font_large;
FC_Font *font_big;

SDL_mutex *mutex;
SDL_cond *cond_read_frame;
SDL_cond *cond_decode_frame;
SDL_cond *cond_inference_frame;
SDL_cond *cond_display_frame;

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

enum AVPixelFormat get_format(AVCodecContext *Context, const enum AVPixelFormat *PixFmt)
{
    while (*PixFmt != AV_PIX_FMT_NONE) {
        if (*PixFmt == AV_PIX_FMT_DRM_PRIME)
            return AV_PIX_FMT_DRM_PRIME;
        if (*PixFmt == AV_PIX_FMT_NV12)
            return AV_PIX_FMT_NV12;
        PixFmt++;
    }
    return AV_PIX_FMT_NONE;
}

static int drm_rga_buf(int src_Width, int src_Height, int wStride, int hStride, int src_fd, int src_format, int dst_Width,
                       int dst_Height, int dst_format, int frameSize, char *buf)
{
    rga_info_t src;
    rga_info_t dst;
    int ret;
    // int hStride = (src_Height + 15) & (~15);
    // int wStride = (src_Width + 15) & (~15);
    // int dhStride = (dst_Height + 15) & (~15);
    // int dwStride = (dst_Width + 15) & (~15);

    memset(&src, 0, sizeof(rga_info_t));
    src.fd = src_fd;
    src.mmuFlag = 1;

    memset(&dst, 0, sizeof(rga_info_t));
    dst.fd = -1;
    dst.virAddr = buf;
    dst.mmuFlag = 1;

    rga_set_rect(&src.rect, 0, 0, src_Width, src_Height, wStride, hStride, src_format);
    rga_set_rect(&dst.rect, 0, 0, dst_Width, dst_Height, dst_Width, dst_Height, dst_format);

    ret = c_RkRgaBlit(&src, &dst, NULL);
    return ret;
}

static int fast_rga_buf(int src_Width, int src_Height, int src_wStride, int src_hStride, int src_format, char *sbuf,
                        int dst_Width, int dst_Height, int dst_wStride, int dst_hStride, int dst_format, char *dbuf)
{
    rga_info_t src;
    rga_info_t dst;
    int ret;

    memset(&src, 0, sizeof(rga_info_t));
    src.fd = -1;
    src.mmuFlag = 1;
    src.virAddr = sbuf;

    memset(&dst, 0, sizeof(rga_info_t));
    dst.fd = -1;
    dst.virAddr = dbuf;
    dst.mmuFlag = 1;

    rga_set_rect(&src.rect, 0, 0, src_Width, src_Height, src_wStride, src_hStride, src_format);
    rga_set_rect(&dst.rect, 0, 0, dst_Width, dst_Height, dst_wStride, dst_hStride, dst_format);

    ret = c_RkRgaBlit(&src, &dst, NULL);
    return ret;
}

#if 0
static char *drm_get_rgaformat_str(uint32_t drm_fmt)
{
  switch (drm_fmt) {
  case DRM_FORMAT_NV12:
    return "RK_FORMAT_YCbCr_420_SP";
  case DRM_FORMAT_NV12_10:
    return "RK_FORMAT_YCbCr_420_SP_10B";
  case DRM_FORMAT_NV15:
    return "RK_FORMAT_YCbCr_420_SP_10B";
  case DRM_FORMAT_NV16:
    return "RK_FORMAT_YCbCr_422_SP";
  case DRM_FORMAT_YUYV:
    return "RK_FORMAT_YUYV_422";
  case DRM_FORMAT_UYVY:
    return "RK_FORMAT_UYVY_422";
  default:
    return "0";
  }
}
#endif

static uint32_t drm_get_rgaformat(uint32_t drm_fmt)
{
    switch (drm_fmt) {
    case DRM_FORMAT_NV12:
        return RK_FORMAT_YCbCr_420_SP;
    case DRM_FORMAT_NV12_10:
        return RK_FORMAT_YCbCr_420_SP_10B;
    case DRM_FORMAT_NV15:
        return RK_FORMAT_YCbCr_420_SP_10B;
    case DRM_FORMAT_NV16:
        return RK_FORMAT_YCbCr_422_SP;
    case DRM_FORMAT_YUYV:
        return RK_FORMAT_YUYV_422;
    case DRM_FORMAT_UYVY:
        return RK_FORMAT_UYVY_422;
    default:
        return 0;
    }
}

static void displayTextureNV12(unsigned char *imageData)
{
    unsigned char *texture_data = NULL;
    int texture_pitch = 0;

    if (loop_counter++ % frmrate_update == 0) {
        currtime = SDL_GetTicks(); // [ms]
        if (currtime - lasttime > 0) {
            frmrate = frmrate_update * (1000.0 / (currtime - lasttime));
        }
        lasttime = currtime;
        avg_frmrate = (prev_frmrate + frmrate) / 2.0;
        prev_frmrate = frmrate;
    }

    int y_size = frame_width * frame_height;
    int uv_size = (frame_width * frame_height) / 2;

    int total_size = y_size + uv_size;

    SDL_LockTexture(texture, 0, (void **)&texture_data, &texture_pitch);

    memcpy(texture_data, (unsigned char *)imageData, total_size);

    SDL_UnlockTexture(texture);
    SDL_RenderClear(renderer);
    SDL_RenderCopy(renderer, texture, NULL, NULL);

    // Draw Objects
    char text[256];
    SDL_Rect rect;
    SDL_Rect rect_bar;
    unsigned int obj;
    int accur_obj;
    int clr;
    for (int i = 0; i < detect_result_group.count; i++) {
        detect_result_t *det_result = &(detect_result_group.results[i]);

        sprintf(text, "%s %.1f%%", det_result->name, det_result->prop * 100);

    printf("%s @ (%d %d %d %d) %f\n",
           det_result->name,
           det_result->box.left,
           det_result->box.top,
           det_result->box.right,
           det_result->box.bottom,
           det_result->prop);

        if (obj2det) {
            obj = hash_me(det_result->name);
            if (obj != obj2det) {
                continue;
            }
        }
        if (accur) {
            accur_obj = (int)(det_result->prop * 100.0);
            if (accur_obj < accur) {
                continue;
            }
        }

        rect.x = det_result->box.left;
        rect.y = det_result->box.top;
        rect.w = det_result->box.right - det_result->box.left + 1;
        rect.h = det_result->box.bottom - det_result->box.top + 1;

        if (det_result->name[0] == 'p' && det_result->name[1] == 'e')
            clr = 1;
        else if (det_result->name[0] == 'c' && det_result->name[1] == 'a')
            clr = 2;
        else if (det_result->name[0] == 'b' && det_result->name[1] == 'u')
            clr = 3;
        else if (det_result->name[0] == 'b' && det_result->name[1] == 'i')
            clr = 4;
        else if (det_result->name[0] == 'm' && det_result->name[1] == 'o')
            clr = 5;
        else if (det_result->name[0] == 'b' && det_result->name[3] == 'k')
            clr = 6;
        else if (det_result->name[0] == 'u' && det_result->name[1] == 'm')
            clr = 7;
        else
            clr = 0;
        if (alphablend) {
            if (clr == 1)
                SDL_SetRenderDrawColor(renderer, 255, 0, 0, alphablend);
            else if (clr == 2)
                SDL_SetRenderDrawColor(renderer, 0, 255, 0, alphablend);
            else if (clr == 3)
                SDL_SetRenderDrawColor(renderer, 255, 0, 255, alphablend);
            else if (clr == 4)
                SDL_SetRenderDrawColor(renderer, 255, 255, 0, alphablend);
            else if (clr == 5)
                SDL_SetRenderDrawColor(renderer, 128, 155, 255, alphablend);
            else if (clr == 6)
                SDL_SetRenderDrawColor(renderer, 128, 128, 128, alphablend);
            else if (clr == 7)
                SDL_SetRenderDrawColor(renderer, 255, 255, 255, alphablend);
            else
                SDL_SetRenderDrawColor(renderer, 0, 0, 255, alphablend);
            SDL_RenderFillRect(renderer, &rect);
        }
        if (clr == 1)
            SDL_SetRenderDrawColor(renderer, 255, 0, 0, SDL_ALPHA_OPAQUE);
        else if (clr == 2)
            SDL_SetRenderDrawColor(renderer, 0, 255, 0, SDL_ALPHA_OPAQUE);
        else if (clr == 3)
            SDL_SetRenderDrawColor(renderer, 255, 0, 255, SDL_ALPHA_OPAQUE);
        else if (clr == 4)
            SDL_SetRenderDrawColor(renderer, 255, 255, 0, SDL_ALPHA_OPAQUE);
        else if (clr == 5)
            SDL_SetRenderDrawColor(renderer, 128, 155, 255, SDL_ALPHA_OPAQUE);
        else if (clr == 6)
            SDL_SetRenderDrawColor(renderer, 128, 128, 128, SDL_ALPHA_OPAQUE);
        else if (clr == 7)
            SDL_SetRenderDrawColor(renderer, 255, 255, 255, SDL_ALPHA_OPAQUE);
        else
            SDL_SetRenderDrawColor(renderer, 0, 0, 255, SDL_ALPHA_OPAQUE);
        SDL_RenderDrawRect(renderer, &rect);

        rect_bar.x = rect.x;
        rect_bar.h = 16;
        rect_bar.w = rect.w;
        if (rect.w < 80)
            rect_bar.h += 16;
        rect_bar.y = rect.y - rect_bar.h;
        SDL_RenderFillRect(renderer, &rect_bar);
        rect_bar.y -= 1;
        FC_DrawBox(font_small, renderer, rect_bar, text);
    }

    SDL_SetRenderDrawColor(renderer, 120, 120, 120, 115);
    rect.x = 0;
    rect.y = 0;
    rect.w = 310;
    rect.h = 90;
    SDL_RenderFillRect(renderer, &rect);

    rect = FC_Draw(font_large, renderer, 0, 0, "%.1f FPS", frmrate);
    rect.y += rect.h;
    FC_Draw(font_large, renderer, rect.x, rect.y, "Inference Time: %.1f ms", avg_inference_time);

    SDL_RenderPresent(renderer);
}

static unsigned int hash_me(char *str)
{
    unsigned int hash = 32;
    while (*str) {
        hash = ((hash << 5) + hash) + (*str++);
    }
    return hash;
}

void print_help(void)
{
    fprintf(stderr, "ff-rknn parameters:\n"
                    "-x displayed width\n"
                    "-y displayed height\n"
                    "-m rknn model\n"
                    "-f protocol (v4l2 only)\n"
                    "-p pixel format (h264) - camera\n"
                    "-s video frame size (WxH) - camera\n"
                    "-r video frame rate - camera\n"
                    "-o unique object to detect\n"
                    "-b use alpha blend on detected objects (1 ~ 255)\n"
                    "-a accuracy perc (1 ~ 100)\n");
}

/*-------------------------------------------
  Functions
  -------------------------------------------*/
static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
    unsigned char *data;
    int ret;

    data = NULL;

    if (NULL == fp) {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0) {
        fprintf(stderr, "blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char *)malloc(sz);
    if (data == NULL) {
        fprintf(stderr, "buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

static unsigned char *load_model(char *filename, int *model_size)
{

    FILE *fp;
    unsigned char *data;

    if (!filename)
        return NULL;

    fp = fopen(filename, "rb");
    if (NULL == fp) {
        fprintf(stderr, "Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}

static int saveFloat(const char *file_name, float *output, int element_size)
{
    FILE *fp;
    fp = fopen(file_name, "w");
    for (int i = 0; i < element_size; i++) {
        fprintf(fp, "%.6f\n", output[i]);
    }
    fclose(fp);
    return 0;
}

static int eventThread(void *data)
{
    int *finished = (int *)data;
    SDL_Event event;

    SDL_Log("Event thread running...");
    while (!*finished) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                *finished = 1;
                break;
            }
            if (event.type == SDL_KEYDOWN) {
                if (event.key.keysym.sym == SDLK_ESCAPE) {
                    *finished = 1;
                    break;
                } else {
                    if (event.key.keysym.sym == SDLK_q) {
                        *finished = 1;
                        break;
                    }
                }
            }
        }
        SDL_Delay(50);
    }
    SDL_Log("Program quit after %d ticks", event.quit.timestamp);
    return 0;
}

int create_mutex(void)
{
    if (!(mutex = SDL_CreateMutex())) {
        av_log(NULL, AV_LOG_FATAL, "SDL_CreateMutex(): %s\n", SDL_GetError());
        return 0;
    }
    if (!(cond_read_frame = SDL_CreateCond())) {
        av_log(NULL, AV_LOG_FATAL, "SDL_CreateCond(cond_read_frame): %s\n", SDL_GetError());
        return 0;
    }
    if (!(cond_decode_frame = SDL_CreateCond())) {
        av_log(NULL, AV_LOG_FATAL, "SDL_CreateCond(cond_decode_frame): %s\n", SDL_GetError());
        return 0;
    }
    if (!(cond_inference_frame = SDL_CreateCond())) {
        av_log(NULL, AV_LOG_FATAL, "SDL_CreateCond(cond_inference_frame): %s\n", SDL_GetError());
        return 0;
    }
    if (!(cond_display_frame = SDL_CreateCond())) {
        av_log(NULL, AV_LOG_FATAL, "SDL_CreateCond(cond_display_frame): %s\n", SDL_GetError());
        return 0;
    }
    return 1;
}

void destroy_mutex(void)
{
    SDL_DestroyMutex(mutex);
    SDL_DestroyCond(cond_read_frame);
    SDL_DestroyCond(cond_decode_frame);
    SDL_DestroyCond(cond_inference_frame);
    SDL_DestroyCond(cond_display_frame);
}

static int readpktThread(void *data)
{
    int *finished = (int *)data;
    int ret;
    int err = 3;

    ret = 0;
    while (ret >= 0 && !*finished) {
        SDL_LockMutex(mutex);
        if ((ret = av_read_frame(input_ctx, &pkt)) < 0) {
            if (ret == AVERROR(EAGAIN) && err > 0) {
                ret = 0;
                err--;
                SDL_Log("Read Frame WAIT!");
                SDL_Delay(5);
                SDL_UnlockMutex(mutex);
                continue;
            }
            finished = 1;
            SDL_Log("Read Frame error!");
            SDL_UnlockMutex(mutex);
            break; /* error */
        }
        err = 3;
        SDL_CondSignal(cond_decode_frame);
        SDL_CondWait(cond_read_frame, mutex);
        SDL_UnlockMutex(mutex);
    }
    SDL_Log("Read Frame quit!");
    SDL_CondSignal(cond_decode_frame);
    SDL_CondSignal(cond_inference_frame);
    SDL_CondSignal(cond_display_frame);
    return 0;
}

static int inferenceThread(void *data)
{
    int *finished = (int *)data;
    RgaSURF_FORMAT src_format;
    RgaSURF_FORMAT dst_format;
    int hStride, wStride;
    SDL_Rect rect;
    int ret;
    struct timeval start_time, stop_time;

    ret = 0;
    while (ret >= 0 && !*finished) {
        SDL_LockMutex(mutex);
        gettimeofday(&start_time, NULL);

        inputs[0].buf = resize_buf;

        rknn_inputs_set(ctx, io_num.n_input, inputs);
        rknn_output outputs[io_num.n_output];

        memset(outputs, 0, sizeof(outputs));

        for (int i = 0; i < io_num.n_output; i++) {
            outputs[i].want_float = 0;
        }

        ret = rknn_run(ctx, NULL);
        ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);

        // post process
        scale_w = (float)width / screen_width;
        scale_h = (float)height / screen_height;

        for (int i = 0; i < io_num.n_output; ++i) {
            out_scales.push_back(output_attrs[i].scale);
            out_zps.push_back(output_attrs[i].zp);
        }

        post_process((int8_t *)outputs[0].buf, (int8_t *)outputs[1].buf, (int8_t *)outputs[2].buf,
                     height, width, box_conf_threshold, nms_threshold,
                     scale_w, scale_h, out_zps, out_scales, &detect_result_group);

        ret = rknn_outputs_release(ctx, io_num.n_output, outputs);

        gettimeofday(&stop_time, NULL);
        inference_time = ((__get_us(stop_time) - __get_us(start_time)) / 1000);
        avg_inference_time = (avg_inference_time + inference_time) / 2.0;

        SDL_CondSignal(cond_display_frame);
        SDL_CondWait(cond_inference_frame, mutex);
        SDL_UnlockMutex(mutex);
    }

    SDL_Log("Inference Frame quit!");
    SDL_CondSignal(cond_read_frame);
    SDL_CondSignal(cond_display_frame);
    return 0;
}

static int decode(AVCodecContext *dec_ctx, AVFrame *frame, AVPacket *pkt)
{
    int ret;

    ret = avcodec_send_packet(dec_ctx, pkt);
    if (ret < 0) {
        fprintf(stderr, "Error sending a packet for decoding\n");
        return ret;
    }
    ret = 0;
    while (ret >= 0) {

        ret = avcodec_receive_frame(dec_ctx, frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        } else if (ret < 0) {
            fprintf(stderr, "Error during decoding!\n");
            return ret;
        }

        sws_scale(swsCtx, (const uint8_t* const*)frame->data, frame->linesize, 0, codec_ctx->height,
                  pFrameYUV->data, pFrameYUV->linesize);

        // Configurar pFrameYUV antes de la codificación
        pFrameYUV->pts = frame->pts;
        // pFrameSDL->pts = pFrameYUV->pts;

        // int ret = avcodec_send_frame(pOutCodecCtx, pFrameYUV);
        // if (ret >= 0) {
        //     ret = avcodec_receive_packet(pOutCodecCtx, outpkt);
        //     if (ret >= 0) {
        //         outpkt->stream_index = pOutStream->index;
        //         av_interleaved_write_frame(pOutputFormatCtx, outpkt);
        //         av_packet_unref(outpkt);
        //     }
        // }

        break;
    }
    return 0;
}

static int decodeThread(void *data)
{
    int *finished = (int *)data;
    RgaSURF_FORMAT src_format;
    RgaSURF_FORMAT dst_format;
    int hStride, wStride;

    int ret;

    ret = 0;
    while (ret >= 0 && !*finished) {
        SDL_LockMutex(mutex);
        ret = decode(codec_ctx, frame, &pkt);
        if (ret < 0) {
            *finished = 1;
            SDL_UnlockMutex(mutex);
            break;
        }
        av_packet_unref(&pkt);
        /* ------------ RKNN ----------- */
        src_format = RK_FORMAT_YCbCr_420_SP;
        dst_format = RK_FORMAT_BGR_888;

        fast_rga_buf(frame->width, frame->height, frame->width, frame->height, src_format, (char *)pFrameYUV->data[0], width, height,
                     width, height, dst_format, (char *)resize_buf);

        SDL_CondSignal(cond_inference_frame);
        SDL_CondWait(cond_decode_frame, mutex);
        SDL_UnlockMutex(mutex);
    }

    SDL_Log("Decode Frame quit!");
    SDL_CondSignal(cond_read_frame);
    SDL_CondSignal(cond_inference_frame);
    SDL_CondSignal(cond_display_frame);
    return 0;
}

int main(int argc, char *argv[])
{
    SDL_Event event;
    SDL_Thread *keybthread;
    SDL_Thread *readthread;
    SDL_Thread *inferencethread;
    SDL_Thread *decodethread;
    int status;
    // SDL_SysWMinfo info;
    SDL_version sdl_compiled;
    SDL_version sdl_linked;
    Uint32 wflags = 0 | SDL_WINDOW_OPENGL | SDL_WINDOW_ALWAYS_ON_TOP | SDL_WINDOW_FULLSCREEN;

    /* -- encoding -- */
    int ret, kmsgrab = 0;
    int lindex, opt;
    char *codec_name = NULL;
    char *video_name = NULL;
    char *pixel_format = NULL, *size_window = NULL;
    AVDictionary *opts = NULL;
    AVDictionaryEntry *dict = NULL;
    const AVInputFormat *ifmt = NULL;
    int nframe = 1;
    int finished = 0;
    int i = 1;
    unsigned int a;
    int fpts;
    int raw_video, skip_some_frames;
    SDL_Rect rect;

    a = 0;

    while (i < argc) {
        a = hash_me(argv[i++]);
        switch (a) {
        case argt_c:
            codec_name = argv[i];
            break;
        case argt_e:
            // enc_file_name = argv[i];
            break;
        case argt_i:
            video_name = argv[i];
            break;
        case argt_x:
            screen_width = atoi(argv[i]);
            break;
        case argt_y:
            screen_height = atoi(argv[i]);
            break;
        case argt_l:
            screen_left = atoi(argv[i]);
            break;
        case argt_t:
            screen_top = atoi(argv[i]);
            break;
        case argt_f:
            // v4l2 = atoi(argv[i]);
            v4l2 = !strncasecmp(argv[i], "v4l2", 4);
            rtsp = !strncasecmp(argv[i], "rtsp", 4);
            rtmp = !strncasecmp(argv[i], "rtmp", 4);
            http = !strncasecmp(argv[i], "http", 4);
            break;
        case argt_r:
            sensor_frame_rate = argv[i];
            break;
        case argt_d:
            delay = atoi(argv[i]);
            break;
        case argt_p:
            pixel_format = argv[i];
            break;
        case argt_s:
            sensor_frame_size = argv[i];
            sscanf(sensor_frame_size, "%dx%d", &frame_width, &frame_height);
            break;
        case argt_m:
            model_name = argv[i];
            break;
        case argt_o:
            obj2det = hash_me(argv[i]);
            break;
        case argt_b:
            alphablend = atoi(argv[i]);
            break;
        case argt_a:
            accur = atoi(argv[i]);
            break;
        default:
            break;
        }
        i++;
    }

    if (!video_name) {
        fprintf(stderr, "No stream to play! Please pass an input.\n");
        print_help();
        return -1;
    }
    if (!model_name) {
        fprintf(stderr, "No model to load! Please pass a model.\n");
        print_help();
        return -1;
    }
    if (screen_width <= 0)
        screen_width = 960;
    if (screen_height <= 0)
        screen_height = 540;
    if (screen_left <= 0)
        screen_left = 0;
    if (screen_top <= 0)
        screen_top = 0;

    font_small = FC_CreateFont();
    if (!font_small) {
        fprintf(stderr, "No small ttf can be created.\n");
        return -1;
    }
    font_large = FC_CreateFont();
    if (!font_large) {
        fprintf(stderr, "No large ttf can be created.\n");
        return -1;
    }
    font_big = FC_CreateFont();
    if (!font_big) {
        fprintf(stderr, "No big ttf can be created\n");
        return -1;
    }

    create_mutex();

    /* Create the neural network */
    model_data_size = 0;
    model_data = load_model(model_name, &model_data_size);
    if (!model_data) {
        fprintf(stderr, "Error locading model: `%s`\n", model_name);
        return -1;
    }
    fprintf(stderr, "Model: %s - size: %d.\n", model_name, model_data_size);
    ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
    if (ret < 0) {
        fprintf(stderr, "rknn_init error ret=%d\n", ret);
        return -1;
    }

    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0) {
        fprintf(stderr, "rknn_init error ret=%d\n", ret);
        return -1;
    }
    fprintf(stderr, "sdk version: %s driver version: %s\n", version.api_version, version.drv_version);

    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) {
        fprintf(stderr, "rknn_init error ret=%d\n", ret);
        return -1;
    }
    fprintf(stderr, "model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    rknn_tensor_attr input_attrs[io_num.n_input + 1];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        // fprintf(stderr, "RKNN_QUERY_OUTPUT_ATTR output_attrs[%d].index=%d\n", i, i);
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
            fprintf(stderr, "rknn_init error ret=%d\n", ret);
            return -1;
        }
    }

    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
    }

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        channel = input_attrs[0].dims[1];
        width = input_attrs[0].dims[2];
        height = input_attrs[0].dims[3];
    } else {
        width = input_attrs[0].dims[1];
        height = input_attrs[0].dims[2];
        channel = input_attrs[0].dims[3];
    }

    fprintf(stderr, "model: %dx%dx%d\n", width, height, channel);
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = width * height * channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;

    input_ctx = avformat_alloc_context();
    if (!input_ctx) {
        av_log(0, AV_LOG_ERROR, "Cannot allocate input format (Out of memory?)\n");
        return -1;
    }

    av_dict_set(&opts, "num_capture_buffers", "128", 0);
    if (rtsp) {
        // av_dict_set(&opts, "rtsp_transport", "tcp", 0);
        av_dict_set(&opts, "rtsp_flags", "prefer_tcp", 0);
    }
    if (v4l2) {
        avdevice_register_all();
        ifmt = (AVInputFormat*) av_find_input_format("v4l2");
        if (!ifmt) {
            av_log(0, AV_LOG_ERROR, "Cannot find input format: v4l2\n");
            return -1;
        }
           input_ctx->flags |= AVFMT_FLAG_NONBLOCK;
        // input_ctx->flags |= AVFMT_FLAG_NOBUFFER;
        // input_ctx->flags |= AVFMT_FLAG_FLUSH_PACKETS;
        // input_ctx->flags |= AVFMT_FLAG_NOPARSE;
        // input_ctx->flags |= AVFMT_FLAG_GENPTS;
        if (pixel_format) {
            av_dict_set(&opts, "input_format", pixel_format, 0);
        }
        if (sensor_frame_size)
            av_dict_set(&opts, "video_size", sensor_frame_size, 0);
        if (sensor_frame_rate)
            av_dict_set(&opts, "framerate", sensor_frame_rate, 0);

#if 1
        av_dict_set(&opts, "fflags", "nobuffer", 0);
        av_dict_set(&opts, "num_capture_buffers", "16", 0);
        av_dict_set(&opts, "flags", "low_delay", 0);
        av_dict_set(&opts, "max_delay", "0", 0);
        av_dict_set(&opts, "probesize", "32", 0);

        av_dict_set(&opts, "avioflags", "direct", 0);
        av_dict_set(&opts, "analyzeduration", "0", 0);
        av_dict_set(&opts, "setpts", "0", 0);
        av_dict_set(&opts, "sync", "ext", 0);
        av_dict_set(&opts, "tune", "zerolatency", 0);
#endif
    }
    if (rtmp) {
        ifmt = av_find_input_format("flv");
        if (!ifmt) {
            av_log(0, AV_LOG_ERROR, "Cannot find input format: flv\n");
            return -1;
        }
        av_dict_set(&opts, "fflags", "nobuffer", 0);
    }

    if (http) {
        av_dict_set(&opts, "fflags", "nobuffer", 0);
    }

    if (avformat_open_input(&input_ctx, video_name, ifmt, &opts) != 0) {
        av_log(0, AV_LOG_ERROR, "Cannot open input file '%s'\n", video_name);
        avformat_close_input(&input_ctx);
        return -1;
    }

    if (avformat_find_stream_info(input_ctx, NULL) < 0) {
        av_log(0, AV_LOG_ERROR, "Cannot find input stream information.\n");
        avformat_close_input(&input_ctx);
        return -1;
    }

    /* find the video stream information */
    ret = av_find_best_stream(input_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
    if (ret < 0) {
        av_log(0, AV_LOG_ERROR, "Cannot find a video stream in the input file\n");
        avformat_close_input(&input_ctx);
        return -1;
    }
    video_stream = ret;

    /* find the video decoder: ie: h264_rkmpp / h264_rkmpp_decoder */
    codecpar = input_ctx->streams[video_stream]->codecpar;
    if (!codecpar) {
        av_log(0, AV_LOG_ERROR, "Unable to find stream!\n");
        avformat_close_input(&input_ctx);
        return -1;
    }

#if 0
    if (codecpar->codec_id != AV_CODEC_ID_H264) {
        av_log(0, AV_LOG_ERROR, "H264 support only!\n");
        avformat_close_input(&input_ctx);
        return -1;
    }
#endif

    codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) {
        av_log(0, AV_LOG_ERROR, "Could not allocate video codec context!\n");
        avformat_close_input(&input_ctx);
        return -1;
    }

    video = input_ctx->streams[video_stream];
    if (avcodec_parameters_to_context(codec_ctx, video->codecpar) < 0) {
        av_log(0, AV_LOG_ERROR, "Error with the codec!\n");
        avformat_close_input(&input_ctx);
        avcodec_free_context(&codec_ctx);
        return -1;
    }

    av_dict_set(&opts, "threads", "auto", 0);

#if 0
    while (dict = av_dict_get(opts, "", dict, AV_DICT_IGNORE_SUFFIX)) {
        fprintf(stderr, "dict: %s -> %s\n", dict->key, dict->value);
    }
#endif

    /* open it */
    if (avcodec_open2(codec_ctx, codec, &opts) < 0) {
        av_log(0, AV_LOG_ERROR, "Could not open codec!\n");
        avformat_close_input(&input_ctx);
        avcodec_free_context(&codec_ctx);
        return -1;
    }

    pFrameYUV = av_frame_alloc();
    int numBytes = av_image_get_buffer_size(AV_PIX_FMT_YUV420P, codec_ctx->width, codec_ctx->height, 1);
    uint8_t* buffer = (uint8_t*)av_malloc(numBytes * sizeof(uint8_t));

    pFrameYUV->format = AV_PIX_FMT_YUV420P;
    pFrameYUV->width = codec_ctx->width;
    pFrameYUV->height = codec_ctx->height;

    // Inicializa los campos de datos de imagen y las líneas de paso (stride) en pFrameYUV
    av_image_fill_arrays(pFrameYUV->data, pFrameYUV->linesize, buffer, AV_PIX_FMT_YUV420P, codec_ctx->width, codec_ctx->height, 1);

    // Configuración de swsContext para la conversión de formatos de imagen
    swsCtx = sws_getContext(codec_ctx->width, codec_ctx->height, codec_ctx->pix_fmt,
                                        codec_ctx->width, codec_ctx->height, AV_PIX_FMT_YUV420P, SWS_BICUBIC,
                                        nullptr, nullptr, nullptr);

    av_dict_free(&opts);

    frame = av_frame_alloc();
    if (!frame) {
        fprintf(stderr, "Could not allocate video frame\n");
        avformat_close_input(&input_ctx);
        avcodec_free_context(&codec_ctx);
        return -1;
    }

    frame->format = AV_PIX_FMT_YUV420P;
    frame->width = codec_ctx->width;
    frame->height = codec_ctx->height;

    SDL_VERSION(&sdl_compiled);
    SDL_GetVersion(&sdl_linked);
    SDL_Log("SDL: compiled with=%d.%d.%d linked against=%d.%d.%d", sdl_compiled.major, sdl_compiled.minor, sdl_compiled.patch,
            sdl_linked.major, sdl_linked.minor, sdl_linked.patch);

    // SDL_SetHint(SDL_HINT_RENDER_DRIVER, "opengles2");
    // SDL_SetHint(SDL_HINT_VIDEO_WAYLAND_ALLOW_LIBDECOR, "0");
    if (SDL_Init(SDL_INIT_EVERYTHING) < 0) {
        SDL_Log("SDL_Init failed (%s)", SDL_GetError());
        avformat_close_input(&input_ctx);
        avcodec_free_context(&codec_ctx);
        return -1;
    }

    SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_ACCELERATED_VISUAL, 1);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES);

    window = SDL_CreateWindow("ff-rknn-v4l2-thread", screen_left, screen_top, screen_width, screen_height, wflags);
    SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "linear");
    if (window) {
        renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
        if (!renderer) {
            av_log(NULL, AV_LOG_WARNING, "Failed to initialize a hardware accelerated renderer: %s\n", SDL_GetError());
            renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
        }
    }
    if (!window || !renderer) {
        SDL_Log("Unable to Create Window or the Renderer failed (%s)", SDL_GetError());
        goto error_exit;
    }

    if (alphablend) {
        SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
    }
    SDL_ShowWindow(window);
    SDL_SetWindowPosition(window, screen_left, screen_top);

    format = SDL_PIXELFORMAT_YV12;

    if (rtsp) {
        format = SDL_PIXELFORMAT_IYUV;
    }
    texture = SDL_CreateTexture(renderer, format, SDL_TEXTUREACCESS_STREAMING, screen_width, screen_height);
    if (!texture) {
        av_log(NULL, AV_LOG_FATAL, "Failed to create texturer %dx%d: %s", screen_width, screen_height, SDL_GetError());
        goto error_exit;
    }

    frameSize_rknn = width * height * channel;
    resize_buf = calloc(1, frameSize_rknn);

    if (!resize_buf) {
        av_log(NULL, AV_LOG_FATAL, "Failed to create rknn buf: %dx%d", width, height);
        goto error_exit;
    }

    FC_LoadFont(font_small, renderer, "/usr/share/fonts/liberation/LiberationMono-Bold.ttf", 16, FC_MakeColor(255, 255, 255, 255), TTF_STYLE_NORMAL);
    FC_LoadFont(font_large, renderer, "/usr/share/fonts/liberation/LiberationMono-Bold.ttf", 26, FC_MakeColor(255, 255, 255, 155), TTF_STYLE_NORMAL);
    FC_LoadFont(font_big, renderer, "/usr/share/fonts/liberation/LiberationMono-Bold.ttf", 72, FC_MakeColor(255, 55, 5, 255), TTF_STYLE_NORMAL);

    rect.x = 0;
    rect.y = 0;
    rect.w = screen_width;
    rect.h = screen_height;
    i = 3;
    if (sensor_frame_rate)
        skip_some_frames = i * atoi(sensor_frame_rate);
    else
        skip_some_frames = i * 30;
    ret = 0;
    while (ret >= 0 && skip_some_frames) {
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 55);
        SDL_RenderFillRect(renderer, &rect);
        SDL_SetRenderDrawColor(renderer, 255, 50, 50, SDL_ALPHA_OPAQUE);
        FC_Draw(font_big, renderer, screen_width / 2 - 220, screen_height / 2 - 100, "Buffering... %d", skip_some_frames);
        SDL_RenderPresent(renderer);
        if ((ret = av_read_frame(input_ctx, &pkt)) < 0) {
            if (ret == AVERROR(EAGAIN)) {
                ret = 0;
                continue;
            }
            break;
        }
        skip_some_frames--;
    }

    finished = 0;
    keybthread = SDL_CreateThread(eventThread, "SDL_EventThread", (void *)&finished);
    readthread = SDL_CreateThread(readpktThread, "SDL_ReadThread", (void *)&finished);
    decodethread = SDL_CreateThread(decodeThread, "SDL_DecodeThread", (void *)&finished);
    inferencethread = SDL_CreateThread(inferenceThread, "SDL_InferenceThread", (void *)&finished);

    while (!finished) {
        SDL_LockMutex(mutex);
        displayTextureNV12((unsigned char *)pFrameYUV->data[0]);
        SDL_CondSignal(cond_read_frame);
        SDL_CondWait(cond_display_frame, mutex);
        SDL_UnlockMutex(mutex);
        SDL_Delay(1);
        if (finished) {
            SDL_Log("Quit!");
            SDL_CondSignal(cond_read_frame);
            break;
        }
    }
    SDL_Delay(40);
    /* flush the codec */
    decode(codec_ctx, frame, NULL);

    SDL_Log("Program wait for the threads...");
    SDL_WaitThread(keybthread, &status);
    SDL_WaitThread(readthread, &status);
    SDL_WaitThread(decodethread, &status);
    SDL_WaitThread(inferencethread, &status);
    SDL_Log("Program exit!");

    destroy_mutex();

error_exit:

    if (input_ctx)
        avformat_close_input(&input_ctx);
    if (codec_ctx)
        avcodec_free_context(&codec_ctx);
    if (frame) {
        av_frame_free(&frame);
    }

    if (pFrameYUV)
        av_frame_free(&pFrameYUV);
    if (pFrameSDL)
        av_frame_free(&pFrameSDL);
    if (outpkt)
        av_packet_free(&outpkt);
    if (pOutputFormatCtx)
        avformat_free_context(pOutputFormatCtx);
    if (pOutCodecCtx)
        avcodec_free_context(&pOutCodecCtx);

    av_free(buffer);
    sws_freeContext(swsCtx);

    if (resize_buf) {
        free(resize_buf);
    }
    if (renderer) {
        SDL_DestroyRenderer(renderer);
    }
    if (window) {
        SDL_DestroyWindow(window);
    }
    if (font_small) {
        FC_FreeFont(font_small);
    }
    if (font_large) {
        FC_FreeFont(font_large);
    }
    if (font_big) {
        FC_FreeFont(font_big);
    }
    SDL_Quit();

    // release
    if (ctx) {
        ret = rknn_destroy(ctx);
    }

    if (model_data) {
        free(model_data);
    }

    deinitPostProcess();

    fprintf(stderr, "Avg FPS: %.1f\n", avg_frmrate);
    fprintf(stderr, "Avg Infer: %f\n", avg_inference_time);
}
