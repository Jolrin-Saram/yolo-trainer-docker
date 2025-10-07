# Use NVIDIA CUDA base image with PyTorch support
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# GitHub repository URL
ARG REPO_URL=https://github.com/Jolrin-Saram/yolo-trainer-docker.git
ARG REPO_BRANCH=main

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Clone the repository from GitHub
RUN git clone --depth 1 --branch ${REPO_BRANCH} ${REPO_URL} /tmp/repo && \
    cp -r /tmp/repo/* /app/ && \
    cp -r /tmp/repo/.* /app/ 2>/dev/null || true && \
    rm -rf /tmp/repo

# Install Python dependencies (use simplified requirements for Docker)
RUN if [ -f requirements-docker.txt ]; then \
        pip3 install --no-cache-dir -r requirements-docker.txt; \
    else \
        pip3 install --no-cache-dir -r requirements.txt; \
    fi

# Create necessary directories
RUN mkdir -p /app/data \
    /app/dataset_prepared \
    /app/trained_models \
    /app/logs

# Set environment variables for GUI support (optional)
ENV DISPLAY=:0
ENV QT_X11_NO_MITSHM=1

# Expose port for potential web interface (future enhancement)
EXPOSE 8000

# Default command - run the training UI
# For headless mode, this can be overridden
CMD ["python3", "run_training_ui.py"]
