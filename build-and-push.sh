#!/bin/bash

# YOLO Auto Trainer - Docker 이미지 빌드 및 푸시 스크립트
# Usage: ./build-and-push.sh [docker-username] [version]

set -e

# 설정
DOCKER_USERNAME=${1:-"yourusername"}
IMAGE_NAME="yolo-auto-trainer"
VERSION=${2:-"latest"}
REPO_URL=${REPO_URL:-"https://github.com/yourusername/yolo-auto-trainer.git"}
REPO_BRANCH=${REPO_BRANCH:-"main"}

# 색상 출력
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}YOLO Auto Trainer - Docker Build${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}Docker Username:${NC} $DOCKER_USERNAME"
echo -e "${GREEN}Image Name:${NC} $IMAGE_NAME"
echo -e "${GREEN}Version:${NC} $VERSION"
echo -e "${GREEN}Repository URL:${NC} $REPO_URL"
echo -e "${GREEN}Branch:${NC} $REPO_BRANCH"
echo ""

# 1. Dockerfile 존재 확인
if [ ! -f "Dockerfile" ]; then
    echo -e "${RED}Error: Dockerfile not found!${NC}"
    exit 1
fi

# 2. Docker 빌드
echo -e "${BLUE}[1/4] Building Docker image...${NC}"
docker build \
    --build-arg REPO_URL=${REPO_URL} \
    --build-arg REPO_BRANCH=${REPO_BRANCH} \
    -t ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION} \
    -t ${DOCKER_USERNAME}/${IMAGE_NAME}:latest \
    .

echo -e "${GREEN}✓ Build completed successfully!${NC}"
echo ""

# 3. 이미지 크기 확인
echo -e "${BLUE}[2/4] Image information:${NC}"
docker images ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}
echo ""

# 4. 테스트 실행
echo -e "${BLUE}[3/4] Testing image...${NC}"
echo "Checking Python and PyTorch installation..."
docker run --rm ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION} \
    python3 -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo -e "${GREEN}✓ Image test passed!${NC}"
echo ""

# 5. Docker Hub에 푸시
read -p "Do you want to push to Docker Hub? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}[4/4] Pushing to Docker Hub...${NC}"

    # 로그인 확인
    docker login

    # 푸시
    docker push ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}
    docker push ${DOCKER_USERNAME}/${IMAGE_NAME}:latest

    echo -e "${GREEN}✓ Successfully pushed to Docker Hub!${NC}"
    echo ""
    echo -e "${GREEN}Your image is now available at:${NC}"
    echo -e "${BLUE}docker pull ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}${NC}"
else
    echo -e "${BLUE}Skipping push to Docker Hub${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Build process completed!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}To run the image:${NC}"
echo -e "docker run --gpus all -it --rm \\"
echo -e "  -v \$(pwd)/data:/app/data \\"
echo -e "  -v \$(pwd)/trained_models:/app/trained_models \\"
echo -e "  ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}"
