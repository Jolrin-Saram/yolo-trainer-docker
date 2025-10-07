# YOLO Auto Trainer - Docker Image

🚀 **어디서든 한 번에 실행 가능한 YOLO 자동 학습 및 라벨링 시스템**

[![Docker Pulls](https://img.shields.io/docker/pulls/username/yolo-auto-trainer.svg)](https://hub.docker.com/r/username/yolo-auto-trainer)
[![Docker Image Size](https://img.shields.io/docker/image-size/username/yolo-auto-trainer/latest.svg)](https://hub.docker.com/r/username/yolo-auto-trainer)

## 🎯 주요 기능

- ✨ **멀티 GPU 병렬 학습** - 최대 4개 GPU 동시 사용 (DDP 자동 활성화)
- 🚀 **자동 라벨링** - 학습된 모델로 새 이미지 자동 라벨링
- 🎨 **GUI 인터페이스** - 직관적인 PyQt5 기반 UI
- 🐳 **완전 포터블** - Docker로 어디서든 실행
- ⚡ **빠른 시작** - 단 한 줄 명령으로 실행 가능

## 🚀 빠른 시작 (단 한 줄!)

### GPU 모드 (권장)
```bash
docker run --gpus all -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/trained_models:/app/trained_models \
  username/yolo-auto-trainer:latest
```

### CPU 모드
```bash
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/trained_models:/app/trained_models \
  username/yolo-auto-trainer:latest
```

### GUI 모드 (Linux/Mac)
```bash
xhost +local:docker
docker run --gpus all -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/trained_models:/app/trained_models \
  username/yolo-auto-trainer:latest
```

## 📋 사전 요구사항

### 필수
- Docker (20.10+)
- 8GB RAM 이상

### GPU 사용 시 (선택사항)
- NVIDIA GPU
- NVIDIA Container Toolkit

#### Linux - NVIDIA Container Toolkit 설치
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### Windows (WSL2)
- Docker Desktop 설치
- "Use the WSL 2 based engine" 활성화
- NVIDIA GPU 드라이버 설치 (Windows용)

## 💡 사용 예제

### 1. 데이터셋 준비 및 학습
```bash
# 1. 데이터셋을 ./data 디렉토리에 배치
mkdir -p data/images data/labels

# 2. Docker 컨테이너 실행
docker run --gpus all -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/dataset_prepared:/app/dataset_prepared \
  -v $(pwd)/trained_models:/app/trained_models \
  username/yolo-auto-trainer:latest

# 3. GUI에서 학습 시작 또는 CLI 사용
```

### 2. 명령줄로 직접 학습
```bash
docker run --gpus all -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/trained_models:/app/trained_models \
  username/yolo-auto-trainer:latest \
  bash -c "
    python3 prepare_dataset.py /app/data /app/data/classes.txt 20 && \
    python3 process_and_train.py \
      /app/dataset_prepared/data.yaml \
      training_options/yolov8x.pt \
      training_options/yolov8x.yaml \
      300 Silu 0,1,2,3 /app/trained_models 0.01 0.1 64
  "
```

### 3. 자동 라벨링
```bash
docker run --gpus all -it --rm \
  -v $(pwd)/unlabeled:/app/unlabeled \
  -v $(pwd)/trained_models:/app/trained_models \
  username/yolo-auto-trainer:latest \
  python3 auto_labeler.py \
    /app/unlabeled \
    /app/trained_models/best.pt \
    0.25 0.7 /app/unlabeled
```

### 4. 특정 GPU만 사용
```bash
# GPU 0과 1만 사용
docker run --gpus '"device=0,1"' -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/trained_models:/app/trained_models \
  username/yolo-auto-trainer:latest
```

## 📁 볼륨 마운트 구조

| 호스트 경로 | 컨테이너 경로 | 설명 |
|------------|-------------|------|
| `./data` | `/app/data` | 원본 데이터셋 (이미지 + 라벨) |
| `./dataset_prepared` | `/app/dataset_prepared` | 전처리된 데이터셋 |
| `./trained_models` | `/app/trained_models` | 학습된 모델 저장 |
| `./logs` | `/app/logs` | 학습 로그 파일 |

## ⚙️ 환경 변수

| 변수명 | 기본값 | 설명 |
|-------|--------|------|
| `DISPLAY` | `:0` | X11 디스플레이 (GUI 모드) |
| `NVIDIA_VISIBLE_DEVICES` | `all` | 사용할 GPU 선택 |
| `PYTHONUNBUFFERED` | `1` | 실시간 로그 출력 |

## 🔧 고급 설정

### Docker Compose 사용
```yaml
version: '3.8'
services:
  yolo-trainer:
    image: username/yolo-auto-trainer:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    volumes:
      - ./data:/app/data
      - ./trained_models:/app/trained_models
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
```

```bash
docker-compose up -d
```

### 커스텀 빌드 (GitHub에서 직접)
```bash
docker build \
  --build-arg REPO_URL=https://github.com/jolrinsaram/yolo-trainer-docker.git \
  --build-arg REPO_BRANCH=main \
  -t my-yolo-trainer:custom \
  - < <(curl -s https://raw.githubusercontent.com/jolrinsaram/yolo-trainer-docker/main/Dockerfile)
```

## 📊 성능

| GPU 개수 | 배치 크기 | Epoch당 시간 | 속도 향상 |
|---------|---------|------------|----------|
| 1 GPU   | 16      | ~10분      | 1.0x     |
| 2 GPUs  | 32      | ~5분       | 2.0x     |
| 4 GPUs  | 64      | ~3분       | 3.3x     |

*테스트 환경: NVIDIA RTX 4090, YOLOv8x, 8000장 이미지*

## 🐛 문제 해결

### GPU가 인식되지 않음
```bash
# GPU 상태 확인
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# nvidia-container-toolkit 재설치
sudo apt-get install --reinstall nvidia-container-toolkit
sudo systemctl restart docker
```

### 권한 오류
```bash
# 현재 사용자로 실행
docker run --user $(id -u):$(id -g) ...

# 또는 디렉토리 권한 변경
sudo chown -R $(id -u):$(id -g) data/ trained_models/
```

### 메모리 부족
```bash
# SHM 크기 증가
docker run --shm-size=8g ...
```

## 📦 이미지 정보

- **Base Image**: `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`
- **Python**: 3.10
- **PyTorch**: 2.5.1+cu121
- **CUDA**: 12.1
- **Ultralytics**: 8.3.204

## 🌐 클라우드 배포

### AWS EC2
```bash
# p3.2xlarge 인스턴스 (Tesla V100)
# Deep Learning AMI 사용

docker run --gpus all -d \
  -v ~/data:/app/data \
  -v ~/trained_models:/app/trained_models \
  username/yolo-auto-trainer:latest
```

### Google Cloud Platform
```bash
# Compute Engine (GPU 인스턴스)
gcloud compute instances create yolo-trainer \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-v100,count=1 \
  --image-family=pytorch-latest-gpu

# 컨테이너 실행
docker run --gpus all -d username/yolo-auto-trainer:latest
```

## 📚 추가 문서

- [전체 사용 가이드](https://github.com/jolrinsaram/yolo-trainer-docker/blob/main/DOCKER_GUIDE.md)
- [멀티 GPU 설정](https://github.com/jolrinsaram/yolo-trainer-docker/blob/main/README_MULTIGPU.md)
- [데이터셋 준비](https://github.com/jolrinsaram/yolo-trainer-docker/blob/main/DATASET_PREP_IMPROVEMENTS.md)

## 🤝 기여

이슈와 PR을 환영합니다!

## 📄 라이선스

MIT License

## 🔗 링크

- [GitHub Repository](https://github.com/jolrinsaram/yolo-trainer-docker)
- [Docker Hub](https://hub.docker.com/r/username/yolo-auto-trainer)
- [이슈 트래커](https://github.com/jolrinsaram/yolo-trainer-docker/issues)

---

**한 번의 명령으로 강력한 YOLO 학습 환경을 시작하세요!** 🚀
