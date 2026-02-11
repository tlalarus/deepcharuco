FROM python:3.10-slim

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    COCO_ROOT=/data/dataset/coco

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    sudo \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt

RUN python -m pip install --upgrade pip "setuptools<81" wheel \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.1.0 \
    && pip install -r /tmp/requirements.txt \
    && pip uninstall -y opencv-python opencv-python-headless || true \
    && pip install "opencv-contrib-python<4.7.0" \
    && pip install \
        PyYAML \
        pydantic \
    && pip install rectangle_packer==2.0.1 \
    && pip install --no-deps git+https://github.com/JunkyByte/python-gridwindow.git

COPY . /workspace

RUN mkdir -p /data/dataset/coco \
    && cp /workspace/src/demo_config.yaml /workspace/src/config.yaml \
    && sed -i "s|/home/adryw/dataset/coco25|${COCO_ROOT}|g" /workspace/src/config.yaml \
    && chown -R ${USERNAME}:${USERNAME} /workspace /data

USER ${USERNAME}
WORKDIR /workspace/src

CMD ["bash"]
