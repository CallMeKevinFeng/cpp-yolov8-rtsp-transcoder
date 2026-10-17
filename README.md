# C++ Real-Time AI Video Pipeline: RTSP + YOLOv8 + NVENC + RTMP

A production-grade C++ pipeline that pulls an RTSP stream, runs **YOLOv8 object detection** via **ONNX Runtime with CUDA acceleration**, draws bounding boxes, and pushes H.264 to an RTMP server using **NVIDIA NVENC hardware encoding**.

Built for real-world scenarios: IP camera analytics, edge AI inference, live streaming enhancement, and video surveillance systems.

---

## Architecture

```
┌─────────────┐    ┌──────────┐    ┌─────────┐    ┌──────────────┐
│  RTSP Input │───▶│  Decode  │───▶│ YUV→BGR │───▶│ YOLOv8 (CUDA)│
│  (IP Camera)│    │  (CPU)   │    │(swscale)│    │ ONNX Runtime │
└─────────────┘    └──────────┘    └─────────┘    └──────┬───────┘
                                                          │
                                                          ▼
┌─────────────┐    ┌──────────┐    ┌─────────┐    ┌──────────────┐
│ RTMP Output │◀───│  NVENC   │◀───│ BGR→YUV │◀───│  Draw Boxes  │
│    (SRS)    │    │  Encode  │    │(swscale)│    │  (OpenCV)    │
└─────────────┘    └──────────┘    └─────────┘    └──────────────┘
```

---

## Key Features

- **Real-time YOLOv8 inference** — ONNX Runtime CUDA Execution Provider, ~10 ms/frame on GTX 1080Ti
- **NVENC H.264 hardware encoding** — offloads encoding to GPU, CPU usage < 30%
- **Low-latency end-to-end** — under 200 ms on local network
- **Frame-accurate PTS handling** — linear PTS counter prevents tearing from out-of-order RTSP timestamps
- **UDP / TCP RTSP transport** — selectable at compile time (`#define USE_TCP_FOR_RTSP`)
- **Clean C++ resource management** — unified cleanup with `goto`, no memory leaks
- **Structured detection parsing** — NMS + confidence thresholding implemented from scratch

---

## Performance

Measured on Ubuntu 22.04, GTX 1080Ti, Intel i7, 1080p @ 30 fps.

| Stage | Detail | Time / Cost |
|-------|--------|-------------|
| Decoding | CPU (H.264 software) | ~5 ms/frame |
| YUV → BGR | `sws_scale` | ~2 ms/frame |
| Inference | YOLOv8n, CUDA EP | ~10 ms/frame |
| Draw boxes | OpenCV `rectangle` + `putText` | ~1 ms/frame |
| BGR → YUV | `sws_scale` | ~2 ms/frame |
| Encoding | NVENC H.264, 4 Mbps | GPU-accelerated |
| **End-to-end latency** | Local network | **< 200 ms** |
| **CPU usage** | Total process | **< 30%** |

---

## Tech Stack

| Component | Version / Detail |
|-----------|------------------|
| Language | C++17 |
| Video I/O | FFmpeg 4.0+ (`libavcodec`, `libavformat`, `libavutil`, `libswscale`) |
| Image processing | OpenCV 4.x |
| AI inference | ONNX Runtime 1.18.1 (GPU build) |
| GPU compute | CUDA 11.8 + cuDNN 8 (via conda-forge) |
| Encoding | NVIDIA NVENC (`h264_nvenc`) |
| Detection model | YOLOv8n (COCO, 80 classes) |

---

## Prerequisites

### System packages
```bash
sudo apt update
sudo apt install -y build-essential pkg-config \
    libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
    libopencv-dev
```

### GPU stack (via conda — avoids system CUDA conflicts)
```bash
conda install -c conda-forge cudatoolkit=11.8 cudnn=8.9 libcublas=11.11 -y
```

### ONNX Runtime GPU (1.18.1)
```bash
cd $HOME
wget https://github.com/microsoft/onnxruntime/releases/download/v1.18.1/onnxruntime-linux-x64-gpu-1.18.1.tgz
tar -xzf onnxruntime-linux-x64-gpu-1.18.1.tgz
mv onnxruntime-linux-x64-gpu-1.18.1 onnxruntime
ls onnxruntime/lib/ | grep cuda
```

---

## Build

```bash
g++ rtsp_ai_transcoder.cpp -o rtsp_ai_transcoder \
    $(pkg-config --cflags --libs libavcodec libavformat libavutil libswscale opencv4) \
    -I$HOME/onnxruntime/include \
    -L$HOME/onnxruntime/lib \
    -lonnxruntime -lonnxruntime_providers_cuda -lonnxruntime_providers_shared \
    -Wl,-rpath,$HOME/onnxruntime/lib:$CONDA_PREFIX/lib
```

---

## Model Setup

Export YOLOv8n to ONNX (one-time):

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ultralytics onnx onnxruntime
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', opset=12, simplify=True)"
ls -lh yolov8n.onnx   # ~12 MB
```

Place `yolov8n.onnx` in the same directory as the executable.

---

## Run

### 1. Start RTSP source (MediaMTX)
```bash
./mediamtx
```

### 2. Push a test stream
```bash
ffmpeg -re -stream_loop -1 -i test.mp4 \
    -c:v libx264 -preset veryfast \
    -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/live/test
```

### 3. Start RTMP sink (SRS)
```bash
sudo docker run -d --restart=always \
    -p 1935:1935 -p 1985:1985 -p 8080:8080 \
    --name srs ossrs/srs:5
```

### 4. Run the AI transcoder
```bash
./rtsp_ai_transcoder \
    rtsp://127.0.0.1:8554/live/test \
    rtmp://127.0.0.1:1935/live/ai
```

### 5. Play the AI-annotated stream
```bash
ffplay rtmp://127.0.0.1:1935/live/ai
```

---

## Transport Mode: UDP vs TCP

| Mode | Latency | Stability | When to use |
|------|---------|-----------|-------------|
| **UDP** (default) | Lowest | Packet loss on unstable networks | LAN, same-datacenter |
| **TCP** | +50–150 ms | No packet loss, in-order delivery | Public internet, IP cameras over WAN |

Enable TCP by uncommenting the following line in `rtsp_ai_transcoder.cpp`:

```cpp
// #define USE_TCP_FOR_RTSP
```

Then recompile.

---

## Implementation Notes

### Why CPU software decoding instead of `h264_cuvid`?
`h264_cuvid` (NVDEC) on Pascal architecture occasionally produces PTS timebase warnings (`Invalid pkt_timebase`) and, in combination with certain RTSP sources, causes decoder state desync. CPU software decoding + NVENC hardware encoding is the **industry-standard stable combination** for 1080p streams and keeps CPU usage well under 30%.

### Why a linear PTS counter?
RTSP streams with B-frames often reorder timestamps. Passing raw `best_effort_timestamp` into NVENC can result in:
- Frame tearing / macroblock corruption
- Reference frame mismatch errors (`co located POCs unavailable`)

A monotonically increasing PTS counter (`pts_counter++`) produces clean, sequential frames while maintaining real-time playback.

### YOLOv8 output parsing
The model outputs a `[1, 84, 8400]` tensor:
- `84 = 4 (cx, cy, w, h) + 80 (COCO class scores)`
- `8400 = predictions across 3 feature map scales`

Per-box decoding:
1. Find `argmax` of the 80 class scores
2. Filter by `CONF_THRESHOLD = 0.25`
3. Reverse letterbox scaling back to original frame coordinates
4. Apply NMS with `IoU threshold = 0.45`

---

## Roadmap

- [x] RTSP input, NVENC output
- [x] YOLOv8 inference via ONNX Runtime CUDA
- [x] Bounding box overlay with OpenCV
- [x] RTSP auto-reconnect on network drop
- [x] Frame rate pacing for live streams
- [ ] Multi-stream concurrent processing
- [x] Zero-copy GPU pipeline (CUDA ↔ NVENC)
- [ ] WebRTC (WHIP/WHEP) output support

---

## Author

**Kevin** — Senior C/C++ Engineer specializing in high-performance video streaming and AI pipelines.

- **Expertise**: FFmpeg, GStreamer, WebRTC, NVENC/NVDEC, TensorRT, ONNX Runtime, CUDA
- **GitHub**: [@CallMeKevinFeng](https://github.com/CallMeKevinFeng)
- **Available for freelance projects** on Upwork — FFmpeg optimization, low-latency streaming, edge AI inference, custom video pipelines.

---

## License

MIT License — applies only to the source code in this repository.

**Note**: This project uses FFmpeg (LGPL/GPL) and links against NVIDIA NVENC. Commercial distribution may require compliance with:
- FFmpeg LGPL/GPL obligations (dynamic linking recommended)
- H.264/HEVC patent licensing (MPEG LA)
- NVIDIA NVENC SDK terms of use



