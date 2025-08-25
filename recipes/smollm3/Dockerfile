FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
ENV BASE_DIR=/workspace/dacp-train

# Install system dependencies in one layer, clean cache
RUN apt update && apt-get install -y \
    git \
    wget \
    curl \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libssl-dev \
    libffi-dev \
    python3 \
    python3-pip \
    git \
    git-lfs && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# set python3 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# install pytorch
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu122

# Copy hydra-fineval project directory
COPY ./ $BASE_DIR

# Create workspace
RUN chmod -R 777 $BASE_DIR

# Install Python packages from the project
WORKDIR $BASE_DIR
RUN pip install --no-cache-dir -e .

# Install flash attention 2
RUN pip install "flash-attn==2.7.4.post1" --no-build-isolation

# Default CMD
CMD ["tail", "-f", "/dev/null"]
