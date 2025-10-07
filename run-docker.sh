#!/bin/bash

# YOLO Auto Trainer - Docker 실행 스크립트
# Usage: ./run-docker.sh [mode] [options]
# Modes: gui, headless, shell

set -e

# 설정
IMAGE_NAME=${DOCKER_IMAGE:-"jolrinsaram/yolo-trainer-docker:latest"}
MODE=${1:-"gui"}

# 색상
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 디렉토리 생성
mkdir -p data dataset_prepared trained_models logs

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}YOLO Auto Trainer - Docker Runner${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# GPU 확인
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
    GPU_FLAGS="--gpus all"
else
    echo -e "${YELLOW}⚠ No NVIDIA GPU detected, running in CPU mode${NC}"
    GPU_FLAGS=""
fi

echo -e "${BLUE}Mode: ${MODE}${NC}"
echo ""

case ${MODE} in
    gui)
        echo -e "${GREEN}Starting in GUI mode...${NC}"

        # Linux/Mac GUI 지원
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            xhost +local:docker 2>/dev/null || true
            docker run ${GPU_FLAGS} -it --rm \
                -e DISPLAY=${DISPLAY} \
                -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
                -v $(pwd)/data:/app/data \
                -v $(pwd)/dataset_prepared:/app/dataset_prepared \
                -v $(pwd)/trained_models:/app/trained_models \
                -v $(pwd)/logs:/app/logs \
                --name yolo-trainer \
                ${IMAGE_NAME}
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            # Mac with XQuartz
            echo -e "${YELLOW}Make sure XQuartz is running and 'Allow connections from network clients' is enabled${NC}"
            docker run ${GPU_FLAGS} -it --rm \
                -e DISPLAY=host.docker.internal:0 \
                -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
                -v $(pwd)/data:/app/data \
                -v $(pwd)/dataset_prepared:/app/dataset_prepared \
                -v $(pwd)/trained_models:/app/trained_models \
                -v $(pwd)/logs:/app/logs \
                --name yolo-trainer \
                ${IMAGE_NAME}
        else
            echo -e "${YELLOW}GUI mode not supported on this platform. Use 'headless' or 'shell' mode.${NC}"
            exit 1
        fi
        ;;

    headless)
        echo -e "${GREEN}Starting in headless mode...${NC}"
        echo -e "${YELLOW}Use this mode for automated training via CLI${NC}"

        docker run ${GPU_FLAGS} -d \
            -v $(pwd)/data:/app/data \
            -v $(pwd)/dataset_prepared:/app/dataset_prepared \
            -v $(pwd)/trained_models:/app/trained_models \
            -v $(pwd)/logs:/app/logs \
            --name yolo-trainer-headless \
            ${IMAGE_NAME} \
            tail -f /dev/null

        echo -e "${GREEN}✓ Container started in background${NC}"
        echo -e "${BLUE}View logs: docker logs -f yolo-trainer-headless${NC}"
        echo -e "${BLUE}Execute commands: docker exec yolo-trainer-headless python3 <script.py>${NC}"
        echo -e "${BLUE}Stop container: docker stop yolo-trainer-headless && docker rm yolo-trainer-headless${NC}"
        ;;

    shell)
        echo -e "${GREEN}Starting interactive shell...${NC}"

        docker run ${GPU_FLAGS} -it --rm \
            -v $(pwd)/data:/app/data \
            -v $(pwd)/dataset_prepared:/app/dataset_prepared \
            -v $(pwd)/trained_models:/app/trained_models \
            -v $(pwd)/logs:/app/logs \
            --name yolo-trainer-shell \
            ${IMAGE_NAME} \
            bash
        ;;

    train)
        echo -e "${GREEN}Starting training...${NC}"

        # 파라미터 확인
        DATASET_PATH=${2:-"/app/data"}
        CLASSES_FILE=${3:-"/app/data/classes.txt"}
        VAL_RATIO=${4:-"20"}
        EPOCHS=${5:-"100"}
        GPU_DEVICES=${6:-"0"}

        docker run ${GPU_FLAGS} -it --rm \
            -v $(pwd)/data:/app/data \
            -v $(pwd)/dataset_prepared:/app/dataset_prepared \
            -v $(pwd)/trained_models:/app/trained_models \
            -v $(pwd)/logs:/app/logs \
            --name yolo-trainer-auto \
            ${IMAGE_NAME} \
            bash -c "
                python3 prepare_dataset.py ${DATASET_PATH} ${CLASSES_FILE} ${VAL_RATIO} && \
                python3 process_and_train.py \
                    /app/dataset_prepared/data.yaml \
                    training_options/yolov8x.pt \
                    training_options/yolov8x.yaml \
                    ${EPOCHS} \
                    Silu \
                    ${GPU_DEVICES} \
                    /app/trained_models \
                    0.01 \
                    0.1 \
                    16
            "
        ;;

    label)
        echo -e "${GREEN}Starting auto-labeling...${NC}"

        IMAGES_PATH=${2:-"/app/data/images"}
        MODEL_PATH=${3:-"/app/trained_models/best.pt"}
        CONF=${4:-"0.25"}
        IOU=${5:-"0.7"}

        docker run ${GPU_FLAGS} -it --rm \
            -v $(pwd)/data:/app/data \
            -v $(pwd)/trained_models:/app/trained_models \
            --name yolo-labeler \
            ${IMAGE_NAME} \
            python3 auto_labeler.py ${IMAGES_PATH} ${MODEL_PATH} ${CONF} ${IOU} ${IMAGES_PATH}
        ;;

    *)
        echo -e "${YELLOW}Unknown mode: ${MODE}${NC}"
        echo ""
        echo "Usage: $0 [mode] [options]"
        echo ""
        echo "Modes:"
        echo "  gui       - Start with GUI (Linux/Mac only)"
        echo "  headless  - Start in background without GUI"
        echo "  shell     - Open interactive bash shell"
        echo "  train     - Auto-train: $0 train [dataset_path] [classes_file] [val_ratio] [epochs] [gpu_devices]"
        echo "  label     - Auto-label: $0 label [images_path] [model_path] [conf] [iou]"
        echo ""
        echo "Examples:"
        echo "  $0 gui"
        echo "  $0 shell"
        echo "  $0 train /app/data /app/data/classes.txt 20 300 0,1,2,3"
        echo "  $0 label /app/data/unlabeled /app/trained_models/best.pt 0.3 0.7"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Done!${NC}"
echo -e "${GREEN}========================================${NC}"
