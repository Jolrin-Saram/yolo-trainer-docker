# 🚀 YOLO Auto Trainer - 빠른 시작 가이드

## ⚡ 단 한 줄로 시작하기

### 1️⃣ GPU 환경에서 실행 (권장)

```bash
docker run --gpus all -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/trained_models:/app/trained_models \
  yourusername/yolo-auto-trainer:latest
```

### 2️⃣ CPU 환경에서 실행

```bash
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/trained_models:/app/trained_models \
  yourusername/yolo-auto-trainer:latest
```

---

## 📦 사전 준비 (최초 1회)

### Windows (PowerShell)

```powershell
# 1. Docker Desktop 설치
# https://www.docker.com/products/docker-desktop

# 2. NVIDIA GPU 사용 시 - WSL2 활성화
wsl --install

# 3. 작업 디렉토리 생성
mkdir yolo-project
cd yolo-project
mkdir data, trained_models
```

### Linux

```bash
# 1. Docker 설치
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 2. NVIDIA Container Toolkit 설치 (GPU 사용 시)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# 3. 작업 디렉토리 생성
mkdir -p yolo-project/{data,trained_models}
cd yolo-project
```

### Mac

```bash
# 1. Docker Desktop 설치
# https://www.docker.com/products/docker-desktop

# 2. 작업 디렉토리 생성
mkdir -p yolo-project/{data,trained_models}
cd yolo-project
```

---

## 🎯 시나리오별 사용법

### 시나리오 1: GUI로 학습하기 (Linux/Mac)

```bash
# X11 권한 허용 (Linux)
xhost +local:docker

# GUI 실행
docker run --gpus all -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/trained_models:/app/trained_models \
  yourusername/yolo-auto-trainer:latest
```

**GUI가 열리면:**
1. "Dataset Folder" 버튼 클릭 → 데이터셋 선택
2. "Classes File" 선택 → classes.txt 파일 선택
3. 학습 파라미터 설정
4. "Start Training" 클릭

---

### 시나리오 2: 명령줄로 자동 학습 (모든 플랫폼)

```bash
# 데이터셋을 ./data에 준비한 후

docker run --gpus all -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/trained_models:/app/trained_models \
  yourusername/yolo-auto-trainer:latest \
  bash -c "
    python3 prepare_dataset.py /app/data /app/data/classes.txt 20 && \
    python3 process_and_train.py \
      /app/dataset_prepared/data.yaml \
      training_options/yolov8x.pt \
      training_options/yolov8x.yaml \
      300 Silu 0 /app/trained_models 0.01 0.1 16
  "
```

**학습 완료 후:**
- 모델: `./trained_models/best.pt`
- 로그: `./trained_models/`

---

### 시나리오 3: 자동 라벨링

```bash
# 라벨링할 이미지를 ./unlabeled에 준비
mkdir unlabeled

docker run --gpus all -it --rm \
  -v $(pwd)/unlabeled:/app/unlabeled \
  -v $(pwd)/trained_models:/app/trained_models \
  yourusername/yolo-auto-trainer:latest \
  python3 auto_labeler.py \
    /app/unlabeled \
    /app/trained_models/best.pt \
    0.25 0.7 /app/unlabeled
```

**결과:**
- `./unlabeled/*.txt` - YOLO 형식 라벨 파일 생성

---

### 시나리오 4: 쉘 접속해서 작업

```bash
# 대화형 쉘 실행
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  yourusername/yolo-auto-trainer:latest \
  bash

# 컨테이너 내부에서
cd /workspace
python3 prepare_dataset.py ...
python3 process_and_train.py ...
exit
```

---

### 시나리오 5: 백그라운드 실행 (서버 환경)

```bash
# 백그라운드에서 실행
docker run --gpus all -d --name yolo-trainer \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/trained_models:/app/trained_models \
  yourusername/yolo-auto-trainer:latest \
  bash -c "
    python3 prepare_dataset.py /app/data /app/data/classes.txt 20 && \
    python3 process_and_train.py \
      /app/dataset_prepared/data.yaml \
      training_options/yolov8x.pt \
      training_options/yolov8x.yaml \
      300 Silu 0,1,2,3 /app/trained_models 0.01 0.1 64
  "

# 로그 확인
docker logs -f yolo-trainer

# 중지
docker stop yolo-trainer
docker rm yolo-trainer
```

---

## 🔧 주요 파라미터 설명

### 학습 파라미터
```bash
python3 process_and_train.py \
  <data.yaml> \          # 데이터셋 설정 파일
  <weights.pt> \         # 사전 학습 가중치
  <model.yaml> \         # 모델 구조
  <epochs> \             # 학습 에폭 (예: 300)
  <activation> \         # 활성화 함수 (Silu, ReLU 등)
  <device> \             # GPU 설정 (0 또는 0,1,2,3)
  <save_dir> \           # 저장 디렉토리
  <lr0> \                # 학습률 (예: 0.01)
  <dropout> \            # 드롭아웃 (예: 0.1)
  <batch_size>           # 배치 크기 (예: 16)
```

### 라벨링 파라미터
```bash
python3 auto_labeler.py \
  <images_path> \        # 이미지 디렉토리
  <model_path> \         # 모델 파일 (.pt)
  <conf_threshold> \     # 신뢰도 임계값 (0.25)
  <iou_threshold> \      # IoU 임계값 (0.7)
  <save_path>            # 라벨 저장 경로
```

---

## 📁 데이터셋 구조

### 학습용 데이터셋
```
data/
├── 1/
│   ├── image1.jpg
│   ├── image1.txt
│   ├── image2.jpg
│   └── image2.txt
├── 2/
│   ├── image3.jpg
│   ├── image3.txt
│   └── ...
└── classes.txt        # 클래스 이름 목록
```

### classes.txt 예시
```
person
car
dog
cat
```

---

## ⚙️ docker-compose 사용 (선택사항)

### docker-compose.yml 다운로드
```bash
curl -O https://raw.githubusercontent.com/yourusername/yolo-auto-trainer/main/docker-compose.yml
```

### 실행
```bash
# GUI 모드
docker-compose up -d

# Headless 모드
docker-compose --profile headless up -d

# 로그 확인
docker-compose logs -f

# 중지
docker-compose down
```

---

## 🎓 예제: 처음부터 끝까지

### 1. 준비
```bash
mkdir my-yolo-project
cd my-yolo-project
mkdir -p data/train/images data/train/labels
```

### 2. 데이터셋 배치
```bash
# 이미지와 라벨을 data/train/ 에 복사
cp /path/to/your/images/* data/train/images/
cp /path/to/your/labels/* data/train/labels/

# classes.txt 생성
echo -e "class1\nclass2\nclass3" > data/classes.txt
```

### 3. 학습 실행
```bash
docker run --gpus all -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/trained_models:/app/trained_models \
  yourusername/yolo-auto-trainer:latest \
  bash -c "
    python3 prepare_dataset.py /app/data /app/data/classes.txt 20 && \
    python3 process_and_train.py \
      /app/dataset_prepared/data.yaml \
      training_options/yolov8n.pt \
      training_options/yolov8n.yaml \
      100 Silu 0 /app/trained_models 0.01 0.0 16
  "
```

### 4. 결과 확인
```bash
ls -lh trained_models/
# best.pt - 최고 성능 모델
# last.pt - 마지막 에폭 모델
```

### 5. 추론/라벨링
```bash
docker run --gpus all -it --rm \
  -v $(pwd)/new_images:/app/new_images \
  -v $(pwd)/trained_models:/app/trained_models \
  yourusername/yolo-auto-trainer:latest \
  python3 auto_labeler.py \
    /app/new_images \
    /app/trained_models/best.pt \
    0.25 0.7 /app/new_images
```

---

## 🐛 문제 해결

### GPU가 인식되지 않음
```bash
# GPU 확인
nvidia-smi

# Docker에서 GPU 테스트
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### 권한 오류
```bash
# Linux/Mac: 디렉토리 권한 설정
sudo chown -R $(id -u):$(id -g) data/ trained_models/

# 또는 현재 사용자로 실행
docker run --user $(id -u):$(id -g) ...
```

### 메모리 부족
```bash
# SHM 크기 증가
docker run --shm-size=8g ...

# 배치 크기 감소
# batch_size를 16 → 8 또는 4로 줄이기
```

### Windows에서 경로 문제
```powershell
# PowerShell에서 절대 경로 사용
docker run --gpus all -it --rm `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/trained_models:/app/trained_models `
  yourusername/yolo-auto-trainer:latest
```

---

## 🌟 팁 & 트릭

### 1. 더 빠른 학습
- GPU 여러 개 사용: `device` 파라미터를 `0,1,2,3`으로
- 배치 크기 증가: GPU 메모리에 따라 16 → 32 → 64

### 2. 더 나은 성능
- 더 큰 모델 사용: `yolov8n.pt` → `yolov8x.pt`
- 더 많은 에폭: 100 → 300 → 500

### 3. 실험 추적
```bash
# TensorBoard 로그 확인
docker run -it --rm \
  -v $(pwd)/trained_models:/logs \
  -p 6006:6006 \
  tensorflow/tensorflow \
  tensorboard --logdir /logs --host 0.0.0.0
```

---

## 📚 더 알아보기

- [전체 Docker 가이드](DOCKER_GUIDE.md)
- [멀티 GPU 설정](README_MULTIGPU.md)
- [데이터셋 준비](DATASET_PREP_IMPROVEMENTS.md)
- [GitHub 저장소](https://github.com/yourusername/yolo-auto-trainer)

---

**이제 시작할 준비가 되었습니다!** 🎉

단 한 줄의 명령으로 강력한 YOLO 학습 환경을 시작하세요.

```bash
docker run --gpus all -it yourusername/yolo-auto-trainer:latest
```
