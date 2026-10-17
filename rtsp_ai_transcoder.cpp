extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
#include <libavutil/time.h>
#include <libswscale/swscale.h>
}

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>

// ==================== 传输协议开关 ====================
// 默认使用 UDP（低延迟）。若要改为 TCP（不丢包），取消下面这行的注释：
#define USE_TCP_FOR_RTSP
// =====================================================

// ==================== YOLOv8 参数 ====================
const int   INPUT_SIZE      = 640;
const float CONF_THRESHOLD  = 0.25f;
const float NMS_THRESHOLD   = 0.45f;

struct Detection {
    cv::Rect box;
    float    conf;
    int      class_id;
};

// ==================== 工具函数 ====================
static cv::Mat letterbox(const cv::Mat& img, float& scale, int& dw, int& dh) {
    int w = img.cols, h = img.rows;
    scale = std::min((float)INPUT_SIZE / w, (float)INPUT_SIZE / h);
    int nw = (int)(w * scale);
    int nh = (int)(h * scale);
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(nw, nh));
    dw = (INPUT_SIZE - nw) / 2;
    dh = (INPUT_SIZE - nh) / 2;
    cv::Mat output(INPUT_SIZE, INPUT_SIZE, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(output(cv::Rect(dw, dh, nw, nh)));
    return output;
}

static std::vector<Detection> nms(std::vector<Detection>& dets, float iou_thres) {
    std::vector<Detection> result;
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b) { return a.conf > b.conf; });
    std::vector<bool> suppressed(dets.size(), false);
    for (size_t i = 0; i < dets.size(); i++) {
        if (suppressed[i]) continue;
        result.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); j++) {
            if (suppressed[j]) continue;
            cv::Rect inter = dets[i].box & dets[j].box;
            float inter_area = inter.area();
            float union_area = dets[i].box.area() + dets[j].box.area() - inter_area;
            if (union_area > 0 && inter_area / union_area > iou_thres) {
                suppressed[j] = true;
            }
        }
    }
    return result;
}

// ==================== RTSP 连接函数 ====================
static bool open_rtsp_input(AVFormatContext** ctx, AVCodecContext** dec,
                            const char* url, int& video_idx) {
    AVDictionary* opts = NULL;
#ifdef USE_TCP_FOR_RTSP
    av_dict_set(&opts, "rtsp_transport", "tcp", 0);
    av_dict_set(&opts, "buffer_size", "1024000", 0);
    av_dict_set(&opts, "max_delay", "500000", 0);
    av_dict_set(&opts, "stimeout", "5000000", 0);
#endif

    if (avformat_open_input(ctx, url, NULL, &opts) < 0) {
        av_dict_free(&opts);
        return false;
    }
    av_dict_free(&opts);

    if (avformat_find_stream_info(*ctx, NULL) < 0) {
        avformat_close_input(ctx);
        return false;
    }

    video_idx = -1;
    for (unsigned i = 0; i < (*ctx)->nb_streams; i++) {
        if ((*ctx)->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_idx = i;
            break;
        }
    }
    if (video_idx < 0) {
        avformat_close_input(ctx);
        return false;
    }

    AVCodecParameters* par = (*ctx)->streams[video_idx]->codecpar;
    const AVCodec* decoder = avcodec_find_decoder(par->codec_id);
    if (!decoder) {
        avformat_close_input(ctx);
        return false;
    }
    *dec = avcodec_alloc_context3(decoder);
    avcodec_parameters_to_context(*dec, par);
    (*dec)->pkt_timebase = (*ctx)->streams[video_idx]->time_base;
    if (avcodec_open2(*dec, decoder, NULL) < 0) {
        avcodec_free_context(dec);
        avformat_close_input(ctx);
        return false;
    }
    return true;
}

// ==================== RTMP 连接函数 ====================
static bool open_rtmp_output(AVFormatContext** out_ctx, AVStream** out_stream,
                             AVCodecContext* enc_ctx, const char* url) {
    if (*out_ctx) {
        if ((*out_ctx)->pb) avio_closep(&(*out_ctx)->pb);
        avformat_free_context(*out_ctx);
        *out_ctx = NULL;
    }

    if (avformat_alloc_output_context2(out_ctx, NULL, "flv", url) < 0 || !*out_ctx) {
        return false;
    }

    *out_stream = avformat_new_stream(*out_ctx, NULL);
    (*out_stream)->time_base = (AVRational){1, 1000};
    avcodec_parameters_from_context((*out_stream)->codecpar, enc_ctx);

    if (avio_open(&(*out_ctx)->pb, url, AVIO_FLAG_WRITE) < 0) {
        avformat_free_context(*out_ctx);
        *out_ctx = NULL;
        return false;
    }

    if (avformat_write_header(*out_ctx, NULL) < 0) {
        avio_closep(&(*out_ctx)->pb);
        avformat_free_context(*out_ctx);
        *out_ctx = NULL;
        return false;
    }
    return true;
}

// ==================== 主函数 ====================
int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Usage: %s <rtsp_input_url> <rtmp_output_url>\n", argv[0]);
        return -1;
    }
    const char* input_url  = argv[1];
    const char* output_url = argv[2];

    // ---------- 所有变量顶部声明，避免 goto 跨初始化 ----------
    AVFormatContext* in_fmt_ctx  = NULL;
    AVFormatContext* out_fmt_ctx = NULL;
    AVCodecContext*  dec_ctx     = NULL;
    AVCodecContext*  enc_ctx     = NULL;
    AVCodecParameters* in_codecpar = NULL;
    const AVCodec*   decoder     = NULL;
    const AVCodec*   encoder     = NULL;
    AVStream*        out_stream  = NULL;
    AVPacket*        in_pkt      = NULL;
    AVPacket*        out_pkt     = NULL;
    AVFrame*         dec_frame   = NULL;
    AVFrame*         enc_frame   = NULL;
    SwsContext*      sws_yuv2bgr = NULL;
    SwsContext*      sws_bgr2yuv = NULL;
    uint8_t*         bgr_buffer  = NULL;
    AVRational       in_time_base = {0, 1};
    AVRational       in_framerate = {0, 1};
    int rtmp_fail_count = 0;
    const int RTMP_MAX_FAIL = 10;
    int video_stream_index = -1;
    int frame_count = 0;
    int ret = 0;
    int send_ret = 0;
    int write_ret = 0;
    int img_w = 0, img_h = 0;
    int bgr_linesize[4] = {0, 0, 0, 0};
    uint8_t* bgr_data[4] = {NULL, NULL, NULL, NULL};
    // reconnect flag
    bool just_reconnected = false;
    AVFrame* last_frame = NULL;       // ★ 缓存最后一帧
    bool has_last_frame = false;      // ★ 是否已有缓存
    bool waiting_for_keyframe = false;   // ★ 新增：等待 I 帧标志
    int64_t pts_counter = 0;
    int64_t stream_start_us = av_gettime_relative();    // ★ 新增
    int64_t next_frame_time_us = av_gettime_relative();   // ★ 新增：下一帧该处理的真实时间

    // ---------- ONNX Runtime 初始化 ----------
    Ort::Env ort_env(ORT_LOGGING_LEVEL_WARNING, "yolo");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // 启用 CUDA Provider
    OrtCUDAProviderOptions cuda_options;
    memset(&cuda_options, 0, sizeof(cuda_options));
    cuda_options.device_id = 0;
    cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
    cuda_options.gpu_mem_limit = SIZE_MAX;
    cuda_options.arena_extend_strategy = 0;
    session_options.AppendExecutionProvider_CUDA(cuda_options);

    Ort::Session session(ort_env, "yolov8n.onnx", session_options);

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name_alloc  = session.GetInputNameAllocated(0, allocator);
    auto output_name_alloc = session.GetOutputNameAllocated(0, allocator);
    const char* input_name  = input_name_alloc.get();
    const char* output_name = output_name_alloc.get();
    printf("[ONNX] input name=%s, output name=%s\n", input_name, output_name);

    auto in_shape  = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    auto out_shape = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    printf("[ONNX] input shape=[%ld,%ld,%ld,%ld], output shape=[%ld,%ld,%ld]\n",
           in_shape[0], in_shape[1], in_shape[2], in_shape[3],
           out_shape[0], out_shape[1], out_shape[2]);

    std::vector<int64_t> onnx_input_shape = {1, 3, INPUT_SIZE, INPUT_SIZE};
    std::vector<float>   onnx_input_data(3 * INPUT_SIZE * INPUT_SIZE);

    // ---------- FFmpeg 初始化 ----------
    avformat_network_init();

    // ==================== RTSP 首次连接（带重试） ====================
    const int RECONNECT_DELAY_SEC = 3;
    int reconnect_attempts = 0;

    while (!open_rtsp_input(&in_fmt_ctx, &dec_ctx, input_url, video_stream_index)) {
        reconnect_attempts++;
        printf("[RTSP] Connect failed (attempt %d), retrying in %d sec...\n",
               reconnect_attempts, RECONNECT_DELAY_SEC);
        av_usleep(RECONNECT_DELAY_SEC * 1000000);
    }
    printf("[RTSP] Connected (attempt %d)\n", reconnect_attempts + 1);

    in_codecpar  = in_fmt_ctx->streams[video_stream_index]->codecpar;
    in_time_base = in_fmt_ctx->streams[video_stream_index]->time_base;
    in_framerate = in_fmt_ctx->streams[video_stream_index]->avg_frame_rate;

    // 编码器：NVENC 硬编
    encoder = avcodec_find_encoder_by_name("h264_nvenc");
    if (!encoder) { printf("No h264_nvenc\n"); ret = -1; goto cleanup; }
    enc_ctx = avcodec_alloc_context3(encoder);
    enc_ctx->width  = dec_ctx->width;
    enc_ctx->height = dec_ctx->height;
    if (in_framerate.num > 0 && in_framerate.den > 0) {
        enc_ctx->framerate = in_framerate;
        enc_ctx->time_base = av_inv_q(in_framerate);
    } else {
        enc_ctx->framerate = (AVRational){30, 1};
        enc_ctx->time_base = (AVRational){1, 30};
    }
    enc_ctx->pix_fmt      = AV_PIX_FMT_YUV420P;
    enc_ctx->bit_rate     = 4000000;
    enc_ctx->gop_size     = 30;
    enc_ctx->max_b_frames = 0;
    enc_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    if (avcodec_open2(enc_ctx, encoder, NULL) < 0) { printf("Open NVENC fail\n"); ret = -1; goto cleanup; }

    img_w = enc_ctx->width;
    img_h = enc_ctx->height;

    // 输出 RTMP
    if (!open_rtmp_output(&out_fmt_ctx, &out_stream, enc_ctx, output_url)) {
        printf("[RTMP] Open failed\n"); ret = -1; goto cleanup;
    }

    // SwsContext：YUV <-> BGR
    sws_yuv2bgr = sws_getContext(img_w, img_h, dec_ctx->pix_fmt,
                                 img_w, img_h, AV_PIX_FMT_BGR24,
                                 SWS_BILINEAR, NULL, NULL, NULL);
    sws_bgr2yuv = sws_getContext(img_w, img_h, AV_PIX_FMT_BGR24,
                                 img_w, img_h, AV_PIX_FMT_YUV420P,
                                 SWS_BILINEAR, NULL, NULL, NULL);
    if (!sws_yuv2bgr || !sws_bgr2yuv) { printf("SwsContext fail\n"); ret = -1; goto cleanup; }

    bgr_buffer = (uint8_t*)av_malloc(img_w * img_h * 3);
    bgr_data[0] = bgr_buffer;
    bgr_linesize[0] = img_w * 3;

    enc_frame = av_frame_alloc();
    enc_frame->format = AV_PIX_FMT_YUV420P;
    enc_frame->width  = img_w;
    enc_frame->height = img_h;
    av_frame_get_buffer(enc_frame, 0);
    last_frame = av_frame_alloc();
    last_frame->format = AV_PIX_FMT_YUV420P;
    last_frame->width  = img_w;
    last_frame->height = img_h;
    av_frame_get_buffer(last_frame, 0);

    in_pkt    = av_packet_alloc();
    out_pkt   = av_packet_alloc();
    dec_frame = av_frame_alloc();

    // ==================== 转码主循环 ====================
    printf("[Pipeline] Begin AI transcode: %s -> %s\n", input_url, output_url);

    while (true) {
        int read_ret = av_read_frame(in_fmt_ctx, in_pkt);

        if (read_ret == AVERROR(EAGAIN)) {
            av_usleep(1000);   // 1ms
            continue;
        }

        if (read_ret < 0) {
            // ==================== 断线重连 ====================
            printf("[RTSP] Stream disconnected (%s), reconnecting...\n",
                   av_err2str(read_ret));

            // 清理输入相关资源（保留编码器和 RTMP 输出）
            avcodec_free_context(&dec_ctx);
            avformat_close_input(&in_fmt_ctx);

            // ==================== 重连循环（带心跳） ====================
            int reconn = 0;
            while (!open_rtsp_input(&in_fmt_ctx, &dec_ctx, input_url, video_stream_index)) {
                reconn++;
                printf("[RTSP] Reconnect failed (attempt %d), retrying...\n", reconn);

                // ★ 心跳：每 1/3 秒推一帧缓存画面，避免 SRS 踢下线
                if (has_last_frame) {
                    int heartbeat_frames = RECONNECT_DELAY_SEC * 3;   // 3 fps
                    for (int i = 0; i < heartbeat_frames; i++) {
                        av_frame_make_writable(last_frame);
                        // 用真实时间计算 PTS，保证单调递增，不会发生时间戳倒退
                        int64_t heartbeat_us = av_gettime_relative() - stream_start_us;
                        last_frame->pts = av_rescale_q(heartbeat_us, {1, 1000000}, enc_ctx->time_base);
                        send_ret = avcodec_send_frame(enc_ctx, last_frame);
                        while (send_ret == AVERROR(EAGAIN)) {
                            if (avcodec_receive_packet(enc_ctx, out_pkt) == 0) {
                                out_pkt->stream_index = out_stream->index;
                                av_packet_rescale_ts(out_pkt, enc_ctx->time_base, out_stream->time_base);
                                int wret = av_interleaved_write_frame(out_fmt_ctx, out_pkt);
                                if (wret == AVERROR(EPIPE) || wret == AVERROR(ECONNRESET)) {
                                    printf("[RTMP] heartbeat failed, reconnecting RTMP...\n");
                                    int retry = 0;
                                    while (!open_rtmp_output(&out_fmt_ctx, &out_stream, enc_ctx, output_url)) {
                                        retry++;
                                        printf("[RTMP] Reconnect failed (attempt %d), retrying in 1s...\n", retry);
                                        av_usleep(1000000);
                                    }
                                    printf("[RTMP] Reconnected successfully!\n");
                                } 
                                av_packet_unref(out_pkt);
                            }
                            send_ret = avcodec_send_frame(enc_ctx, last_frame);
                        }
                        while (avcodec_receive_packet(enc_ctx, out_pkt) == 0) {
                            out_pkt->stream_index = out_stream->index;
                            av_packet_rescale_ts(out_pkt, enc_ctx->time_base, out_stream->time_base);
                            int wret = av_interleaved_write_frame(out_fmt_ctx, out_pkt);
                            if (wret == AVERROR(EPIPE) || wret == AVERROR(ECONNRESET)) {
                                printf("[RTMP] heartbeat failed, reconnecting RTMP...\n");
                                int retry = 0;
                                while (!open_rtmp_output(&out_fmt_ctx, &out_stream, enc_ctx, output_url)) {
                                    retry++;
                                    printf("[RTMP] Reconnect failed (attempt %d), retrying in 1s...\n", retry);
                                    av_usleep(1000000);
                                }
                                printf("[RTMP] Reconnected successfully!\n");
                            }
                            av_packet_unref(out_pkt);
                        }
                        av_usleep(333333);   // 1/3 秒
                    }
                } else {
                    // 还没有任何帧缓存，纯等待
                    av_usleep(RECONNECT_DELAY_SEC * 1000000);
                }
            }
            printf("[RTSP] Reconnected after %d failed attempts\n", reconn);

            just_reconnected = true;
            waiting_for_keyframe = true;
            dec_ctx->skip_frame = AVDISCARD_NONKEY;
                        // ==================== 重连后检查输入流参数变化 ====================
            if (dec_ctx->width != img_w || dec_ctx->height != img_h) {
                printf("[RTSP] Resolution changed: %dx%d -> %dx%d, rebuilding FULL pipeline\n",
                       img_w, img_h, dec_ctx->width, dec_ctx->height);

                // 1. 清理旧的资源
                if (enc_frame) av_frame_free(&enc_frame);
                if (enc_ctx) avcodec_free_context(&enc_ctx);
                if (sws_yuv2bgr) sws_freeContext(sws_yuv2bgr);
                if (sws_bgr2yuv) sws_freeContext(sws_bgr2yuv);
                if (bgr_buffer) av_free(bgr_buffer);
                if (last_frame) av_frame_free(&last_frame);
                if (out_fmt_ctx) {
                    if (out_fmt_ctx->pb) avio_closep(&out_fmt_ctx->pb);
                    avformat_free_context(out_fmt_ctx);
                    out_fmt_ctx = NULL;
                }

                // 2. 更新尺寸
                img_w = dec_ctx->width;
                img_h = dec_ctx->height;
                has_last_frame = false;

                // 3. 重建 NVENC 编码器
                enc_ctx = avcodec_alloc_context3(encoder);
                enc_ctx->width  = img_w;
                enc_ctx->height = img_h;
                if (in_framerate.num > 0 && in_framerate.den > 0) {
                    enc_ctx->framerate = in_framerate;
                    enc_ctx->time_base = av_inv_q(in_framerate);
                } else {
                    enc_ctx->framerate = (AVRational){30, 1};
                    enc_ctx->time_base = (AVRational){1, 30};
                }
                enc_ctx->pix_fmt      = AV_PIX_FMT_YUV420P;
                enc_ctx->bit_rate     = 4000000;
                enc_ctx->gop_size     = 30;
                enc_ctx->max_b_frames = 0;
                enc_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
                if (avcodec_open2(enc_ctx, encoder, NULL) < 0) { 
                    printf("Rebuild NVENC fail\n"); ret = -1; goto cleanup; 
                }

                // 4. 重建 RTMP 输出 (这里调用我们上一轮写的 open_rtmp_output 函数)
                if (!open_rtmp_output(&out_fmt_ctx, &out_stream, enc_ctx, output_url)) {
                    printf("Rebuild RTMP fail\n"); ret = -1; goto cleanup;
                }

                // 5. 重建 SwsContext 和 BGR 缓冲区
                sws_yuv2bgr = sws_getContext(img_w, img_h, dec_ctx->pix_fmt,
                                             img_w, img_h, AV_PIX_FMT_BGR24,
                                             SWS_BILINEAR, NULL, NULL, NULL);
                sws_bgr2yuv = sws_getContext(img_w, img_h, AV_PIX_FMT_BGR24,
                                             img_w, img_h, AV_PIX_FMT_YUV420P,
                                             SWS_BILINEAR, NULL, NULL, NULL);
                bgr_buffer = (uint8_t*)av_malloc(img_w * img_h * 3);
                bgr_data[0] = bgr_buffer;
                bgr_linesize[0] = img_w * 3;

                // 6. 重建编码帧
                enc_frame = av_frame_alloc();
                enc_frame->format = AV_PIX_FMT_YUV420P;
                enc_frame->width  = img_w;
                enc_frame->height = img_h;
                av_frame_get_buffer(enc_frame, 0);

                // 7. 重建心跳缓存帧
                last_frame = av_frame_alloc();
                last_frame->format = AV_PIX_FMT_YUV420P;
                last_frame->width  = img_w;
                last_frame->height = img_h;
                av_frame_get_buffer(last_frame, 0);

                printf("[RTSP] Pipeline rebuilt for %dx%d\n", img_w, img_h);
            }
            // 更新输入流参数
            in_codecpar  = in_fmt_ctx->streams[video_stream_index]->codecpar;
            in_time_base = in_fmt_ctx->streams[video_stream_index]->time_base;
            in_framerate = in_fmt_ctx->streams[video_stream_index]->avg_frame_rate;

            avformat_flush(in_fmt_ctx);  // ★ 新增：清空由于网络爆发缓存的旧包
            continue;
        }
        // ==================== 正常处理 ====================
        if (in_pkt->stream_index != video_stream_index) {
            av_packet_unref(in_pkt);
            continue;
        }
        if (avcodec_send_packet(dec_ctx, in_pkt) < 0) {
            av_packet_unref(in_pkt);
            continue;
        }
        av_packet_unref(in_pkt);

        while (avcodec_receive_frame(dec_ctx, dec_frame) == 0) {
            // 重连后等待第一个关键帧
            if (waiting_for_keyframe) {
                if (dec_frame->flags & AV_FRAME_FLAG_KEY) {
                    printf("[RTSP] Got first keyframe after reconnect\n");
                    waiting_for_keyframe = false;
                    dec_ctx->skip_frame = AVDISCARD_DEFAULT;
                } else {
                    av_frame_unref(dec_frame);
                    continue;
                }
            }
            // 1. YUV -> BGR
            sws_scale(sws_yuv2bgr, dec_frame->data, dec_frame->linesize, 0, img_h,
                      bgr_data, bgr_linesize);
            cv::Mat bgr_img(img_h, img_w, CV_8UC3, bgr_buffer, bgr_linesize[0]);

            // 2. Letterbox 到 640x640
            float scale; int dw, dh;
            cv::Mat letterboxed = letterbox(bgr_img, scale, dw, dh);

            // 3. 转 float / NCHW / 归一化
            cv::Mat rgb;
            cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);
            cv::Mat float_img;
            rgb.convertTo(float_img, CV_32FC3, 1.0 / 255.0);
            std::vector<cv::Mat> channels(3);
            cv::split(float_img, channels);
            for (int c = 0; c < 3; c++) {
                memcpy(onnx_input_data.data() + c * INPUT_SIZE * INPUT_SIZE,
                       channels[c].data, INPUT_SIZE * INPUT_SIZE * sizeof(float));
            }

            // 4. ONNX 推理
            Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                mem_info, onnx_input_data.data(), onnx_input_data.size(),
                onnx_input_shape.data(), onnx_input_shape.size());
            const char* in_names[]  = {input_name};
            const char* out_names[] = {output_name};
            auto outputs = session.Run(Ort::RunOptions{nullptr}, in_names, &input_tensor, 1, out_names, 1);

            // 5. 解析输出 [1, 84, 8400]
            float* out_data = outputs[0].GetTensorMutableData<float>();
            auto out_info = outputs[0].GetTensorTypeAndShapeInfo();
            auto out_dims = out_info.GetShape();
            int num_classes = (int)out_dims[1] - 4;
            int num_boxes   = (int)out_dims[2];

            std::vector<Detection> detections;
            for (int i = 0; i < num_boxes; i++) {
                float cx = out_data[0 * num_boxes + i];
                float cy = out_data[1 * num_boxes + i];
                float w  = out_data[2 * num_boxes + i];
                float h  = out_data[3 * num_boxes + i];

                float max_conf = 0;
                int max_class = -1;
                for (int c = 0; c < num_classes; c++) {
                    float cf = out_data[(4 + c) * num_boxes + i];
                    if (cf > max_conf) { max_conf = cf; max_class = c; }
                }
                if (max_conf < CONF_THRESHOLD) continue;

                float x1 = (cx - w / 2 - dw) / scale;
                float y1 = (cy - h / 2 - dh) / scale;
                float x2 = (cx + w / 2 - dw) / scale;
                float y2 = (cy + h / 2 - dh) / scale;

                x1 = std::max(0.0f, std::min((float)img_w, x1));
                y1 = std::max(0.0f, std::min((float)img_h, y1));
                x2 = std::max(0.0f, std::min((float)img_w, x2));
                y2 = std::max(0.0f, std::min((float)img_h, y2));
                if (x2 <= x1 || y2 <= y1) continue;

                Detection det;
                det.box = cv::Rect((int)x1, (int)y1, (int)(x2 - x1), (int)(y2 - y1));
                det.conf = max_conf;
                det.class_id = max_class;
                detections.push_back(det);
            }

            // 6. NMS
            std::vector<Detection> final_dets = nms(detections, NMS_THRESHOLD);

            // 7. 画框
            for (auto& det : final_dets) {
                cv::rectangle(bgr_img, det.box, cv::Scalar(0, 255, 0), 2);
                char label[64];
                snprintf(label, sizeof(label), "id:%d %.2f", det.class_id, det.conf);
                cv::putText(bgr_img, label,
                            cv::Point(det.box.x, std::max(15, det.box.y - 5)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
            }

            // 8. BGR -> YUV420P
            av_frame_make_writable(enc_frame);
            sws_scale(sws_bgr2yuv, bgr_data, bgr_linesize, 0, img_h,
                      enc_frame->data, enc_frame->linesize);

            // ★ 缓存当前帧，用于断流期间的"心跳"
            av_frame_make_writable(last_frame);
            av_frame_copy(last_frame, enc_frame);
            has_last_frame = true;

            // 9. PTS：用真实时间计算，断流重连后 PTS 会自动跳到当前时刻
            int64_t elapsed_us = av_gettime_relative() - stream_start_us;
            enc_frame->pts = av_rescale_q(elapsed_us, {1, 1000000}, enc_ctx->time_base);
            if (just_reconnected) {
                enc_frame->pict_type = AV_PICTURE_TYPE_I;
                just_reconnected = false;
            }
            // 10. 发送到编码器
            send_ret = avcodec_send_frame(enc_ctx, enc_frame);
            while (send_ret == AVERROR(EAGAIN)) {
                if (avcodec_receive_packet(enc_ctx, out_pkt) == 0) {
                    out_pkt->stream_index = out_stream->index;
                    av_packet_rescale_ts(out_pkt, enc_ctx->time_base, out_stream->time_base);
                    // --- 统一处理 RTMP 推流失败 ---
                    write_ret = av_interleaved_write_frame(out_fmt_ctx, out_pkt);
                    if (write_ret < 0) {
                        rtmp_fail_count++;
                        printf("[RTMP] push fail: %d (total %d)\n", write_ret, rtmp_fail_count);
                        
                        if (rtmp_fail_count >= 5) {
                            printf("[RTMP] Connection lost, reconnecting RTMP...\n");
                            rtmp_fail_count = 0;

                            // 重建 RTMP 连接
                            int retry = 0;
                            while (!open_rtmp_output(&out_fmt_ctx, &out_stream, enc_ctx, output_url)) {
                                retry++;
                                printf("[RTMP] Reconnect failed (attempt %d), retrying in 1s...\n", retry);
                                av_usleep(1000000);
                            }
                            printf("[RTMP] Reconnected successfully!\n");
                        }
                    } else {
                        rtmp_fail_count = 0;
                    }
                    av_packet_unref(out_pkt);
                    frame_count++;
                }
                send_ret = avcodec_send_frame(enc_ctx, enc_frame);
            }
            // 11. 接收编码后的包
            while (avcodec_receive_packet(enc_ctx, out_pkt) == 0) {
                out_pkt->stream_index = out_stream->index;
                av_packet_rescale_ts(out_pkt, enc_ctx->time_base, out_stream->time_base);
                // --- 统一处理 RTMP 推流失败 ---
                write_ret = av_interleaved_write_frame(out_fmt_ctx, out_pkt);
                if (write_ret < 0) {
                    rtmp_fail_count++;
                    printf("[RTMP] push fail: %d (total %d)\n", write_ret, rtmp_fail_count);
                    if (rtmp_fail_count >= 5) {
                        printf("[RTMP] Connection lost, reconnecting RTMP...\n");
                        rtmp_fail_count = 0;

                        // 重建 RTMP 连接
                        int retry = 0;
                        while (!open_rtmp_output(&out_fmt_ctx, &out_stream, enc_ctx, output_url)) {
                            retry++;
                            printf("[RTMP] Reconnect failed (attempt %d), retrying in 1s...\n", retry);
                            av_usleep(1000000);
                        }
                        printf("[RTMP] Reconnected successfully!\n");
                    }
                } else {
                    rtmp_fail_count = 0;
                }
                av_packet_unref(out_pkt);
                frame_count++;
            }

            av_frame_unref(dec_frame);

            // ★ 帧率控制 (Pacing)：防止重连瞬间数据爆发导致 ffplay 卡顿
            int fps = 30;
            if (in_framerate.num > 0 && in_framerate.den > 0) {
                fps = in_framerate.num / in_framerate.den;
                if (fps <= 0 || fps > 120) fps = 30; // 防御性保护
            }
            int64_t frame_duration_us = 1000000 / fps;
            
            next_frame_time_us += frame_duration_us;
            int64_t now_us = av_gettime_relative();
            if (next_frame_time_us > now_us) {
                av_usleep(next_frame_time_us - now_us);
            } else {
                // 如果处理速度远慢于实时（比如机器太卡），重置基准时间，避免死命追赶
                next_frame_time_us = now_us;
            }
        }
    }

    // 冲刷编码器
    avcodec_send_frame(enc_ctx, NULL);
    while (avcodec_receive_packet(enc_ctx, out_pkt) == 0) {
        out_pkt->stream_index = out_stream->index;
        av_packet_rescale_ts(out_pkt, enc_ctx->time_base, out_stream->time_base);
        av_interleaved_write_frame(out_fmt_ctx, out_pkt);
        av_packet_unref(out_pkt);
    }
    av_write_trailer(out_fmt_ctx);
    printf("[Pipeline] AI Transcode finished, total frames: %d\n", frame_count);

cleanup:
    if (bgr_buffer) av_free(bgr_buffer);
    if (sws_yuv2bgr) sws_freeContext(sws_yuv2bgr);
    if (sws_bgr2yuv) sws_freeContext(sws_bgr2yuv);
    if (in_pkt)  av_packet_free(&in_pkt);
    if (out_pkt) av_packet_free(&out_pkt);
    if (dec_frame) av_frame_free(&dec_frame);
    if (enc_frame) av_frame_free(&enc_frame);
    if (last_frame) av_frame_free(&last_frame);
    if (dec_ctx) avcodec_free_context(&dec_ctx);
    if (enc_ctx) avcodec_free_context(&enc_ctx);
    if (out_fmt_ctx) {
        if (out_fmt_ctx->pb) avio_closep(&out_fmt_ctx->pb);
        avformat_free_context(out_fmt_ctx);
    }
    if (in_fmt_ctx) avformat_close_input(&in_fmt_ctx);
    avformat_network_deinit();
    return ret;
}