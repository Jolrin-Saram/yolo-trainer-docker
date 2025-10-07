# Docker 실행 가이드 - YOLO Auto Trainer

Docker를 사용하면 어디서든 일관된 환경에서 YOLO 자동 학습 툴을 실행할 수 있습니다.

## 📋 사전 요구사항

### 1. Docker 설치
- **Linux**: Docker Engine + Docker Compose
- **Windows**: Docker Desktop (WSL2 백엔드)
- **Mac**: Docker Desktop

### 2. NVIDIA GPU 지원 (선택사항이지만 강력 권장)

#### Linux
```bash
# NVIDIA Container Toolkit 설치
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### Windows (WSL2)
```powershell
# Docker Desktop 설정에서 "Use the WSL 2 based engine" 활성화
# NVIDIA GPU 드라이버 설치 (호스트 Windows용)
# WSL2에 CUDA 자동 지원됨
```

### 3. 설치 확인
```bash
# Docker 버전 확인
docker --version
docker-compose --version

# GPU 지원 확인 (선택사항)
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# .env 파일 생성
cp .env.example .env

# 필요시 .env 파일 편집
nano .env
```

### 2. Docker 이미지 빌드
```bash
# 이미지 빌드
docker-compose build

# 또는 특정 서비스만 빌드
docker-compose build yolo-trainer
```

### 3. 컨테이너 실행

#### GUI 모드 (Linux/Mac)
```bash
# X11 권한 허용 (Linux)
xhost +local:docker

# 컨테이너 실행
docker-compose up -d yolo-trainer

# 로그 확인
docker-compose logs -f yolo-trainer
```

#### Headless 모드 (모든 플랫폼)
```bash
# Headless 서비스 실행
docker-compose --profile headless up -d yolo-trainer-headless

# 로그 모니터링
docker-compose logs -f yolo-trainer-headless
```

#### Windows (GUI 없이 명령줄만)
```bash
# Windows에서는 GUI가 제한적이므로 headless 모드 권장
docker-compose --profile headless up -d yolo-trainer-headless

# 또는 대화형 셸로 접속
docker-compose run --rm yolo-trainer bash
```

## 📁 데이터 볼륨 구조

컨테이너는 다음 디렉토리를 호스트와 공유합니다:

```
.
├── data/                    # 원본 데이터셋
├── dataset_prepared/        # 전처리된 데이터셋
├── trained_models/          # 학습된 모델 저장
├── logs/                    # 학습 로그
├── config.json             # 설정 파일
└── datasets/               # 추가 데이터셋 (선택사항)
```

## 💡 사용 예제

### 예제 1: GUI 모드로 학습 시작 (Linux)
```bash
# 1. X11 권한 설정
xhost +local:docker

# 2. 컨테이너 시작
docker-compose up -d yolo-trainer

# 3. GUI 애플리케이션이 자동으로 열립니다
# 브라우저나 VNC로 접속 가능
```

### 예제 2: 명령줄로 직접 학습 실행
```bash
# 컨테이너 내부 셸 접속
docker-compose run --rm yolo-trainer bash

# 데이터셋 준비
python3 prepare_dataset.py /datasets/my_dataset /datasets/my_dataset/classes.txt 20

# 학습 시작
python3 process_and_train.py \
  /app/dataset_prepared/data.yaml \
  training_options/yolov8x.pt \
  training_options/yolov8x.yaml \
  300 \
  Silu \
  0,1,2,3 \
  /app/trained_models \
  0.01 \
  0.1 \
  64
```

### 예제 3: 자동 라벨링
```bash
# 컨테이너에서 자동 라벨링 실행
docker-compose run --rm yolo-trainer \
  python3 auto_labeler.py \
  /datasets/unlabeled_images \
  /app/trained_models/best.pt \
  0.25 \
  0.7 \
  /datasets/unlabeled_images
```

### 예제 4: 특정 GPU만 사용
```bash
# .env 파일 수정
echo "NVIDIA_VISIBLE_DEVICES=0,1" > .env

# 또는 직접 지정
docker-compose run --rm \
  -e NVIDIA_VISIBLE_DEVICES=0,1 \
  yolo-trainer python3 run_training_ui.py
```

## 🔧 고급 설정

### 사용자 정의 이미지 빌드
```bash
# 커스텀 빌드 인수 사용
docker build \
  --build-arg CUDA_VERSION=12.1.0 \
  --build-arg PYTHON_VERSION=3.10 \
  -t yolo-trainer:custom .
```

### 멀티 스테이지 빌드로 이미지 크기 최적화
Dockerfile을 수정하여 멀티 스테이지 빌드 사용 가능

### 영구 볼륨 사용
```yaml
# docker-compose.yml에 추가
volumes:
  yolo-data:
  yolo-models:

services:
  yolo-trainer:
    volumes:
      - yolo-data:/app/data
      - yolo-models:/app/trained_models
```

## 🐛 문제 해결

### GPU가 인식되지 않음
```bash
# GPU 상태 확인
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# nvidia-container-toolkit 재시작
sudo systemctl restart docker

# Docker 데몬 설정 확인
cat /etc/docker/daemon.json
```

### GUI가 표시되지 않음 (Linux)
```bash
# X11 권한 확인
xhost +local:docker

# DISPLAY 환경변수 확인
echo $DISPLAY

# .env 파일에서 DISPLAY 설정
echo "DISPLAY=$DISPLAY" >> .env
```

### 권한 오류
```bash
# 볼륨 디렉토리 권한 설정
sudo chown -R $(id -u):$(id -g) data/ dataset_prepared/ trained_models/

# 또는 컨테이너를 현재 사용자로 실행
docker-compose run --rm --user $(id -u):$(id -g) yolo-trainer
```

### 메모리 부족
```bash
# Docker Desktop 메모리 제한 증가
# Settings > Resources > Memory를 16GB 이상으로 설정

# 또는 docker-compose.yml에서 제한 설정
services:
  yolo-trainer:
    deploy:
      resources:
        limits:
          memory: 16G
```

## 📊 성능 최적화

### 1. SHM 크기 증가 (대용량 배치)
```yaml
services:
  yolo-trainer:
    shm_size: '8gb'
```

### 2. 데이터 로더 워커 수 조정
환경 변수로 설정 가능:
```bash
docker-compose run --rm \
  -e WORKERS=8 \
  yolo-trainer python3 process_and_train.py
```

### 3. 캐시 디렉토리 마운트
```yaml
volumes:
  - ~/.cache:/root/.cache
```

## 🔄 업데이트 및 유지보수

### 이미지 업데이트
```bash
# 최신 코드로 재빌드
docker-compose build --no-cache yolo-trainer

# 기존 컨테이너 중지 및 제거
docker-compose down

# 새 이미지로 시작
docker-compose up -d yolo-trainer
```

### 컨테이너 정리
```bash
# 중지된 컨테이너 제거
docker-compose down

# 볼륨까지 모두 제거
docker-compose down -v

# 이미지도 함께 제거
docker-compose down --rmi all
```

### 로그 확인 및 디버깅
```bash
# 실시간 로그 확인
docker-compose logs -f yolo-trainer

# 특정 시간 이후 로그
docker-compose logs --since 30m yolo-trainer

# 컨테이너 내부 접속
docker-compose exec yolo-trainer bash

# 컨테이너 상태 확인
docker-compose ps
docker stats
```

## 🌐 클라우드 배포

### AWS EC2
```bash
# p3/p4 인스턴스에 Deep Learning AMI 사용
# Docker 및 nvidia-docker 사전 설치됨

# 저장소 클론
git clone <repository-url>
cd portable

# 빌드 및 실행
docker-compose up -d yolo-trainer-headless
```

### Google Cloud Platform
```bash
# Compute Engine GPU 인스턴스 생성
# Container-Optimized OS 또는 Ubuntu 선택

# 동일한 방식으로 실행
docker-compose up -d
```

### Azure
```bash
# NC 시리즈 VM 사용
# GPU 드라이버 및 Docker 설치 후 동일
```

## 📦 Docker Hub에 이미지 푸시

```bash
# 이미지 태그
docker tag yolo-auto-trainer:latest your-username/yolo-auto-trainer:latest

# Docker Hub 로그인
docker login

# 이미지 푸시
docker push your-username/yolo-auto-trainer:latest

# 다른 곳에서 사용
docker pull your-username/yolo-auto-trainer:latest
docker run --gpus all -it your-username/yolo-auto-trainer:latest
```

## 🎯 요약

**기본 워크플로우:**
1. `docker-compose build` - 이미지 빌드
2. `docker-compose up -d` - 컨테이너 시작
3. 데이터셋을 `./data/` 또는 `./datasets/`에 배치
4. GUI 또는 CLI로 학습 실행
5. `./trained_models/`에서 결과 확인

**주요 명령어:**
- 시작: `docker-compose up -d`
- 중지: `docker-compose down`
- 로그: `docker-compose logs -f`
- 셸 접속: `docker-compose exec yolo-trainer bash`

이제 Docker를 통해 어디서든 YOLO 자동 학습 환경을 쉽게 구축할 수 있습니다! 🚀
