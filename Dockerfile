FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

# 필수 시스템 패키지 (최소화) - GUI 없는 서버라면 이 정도면 충분
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget unzip build-essential cmake libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리
WORKDIR /workspace

# pip 최신화
RUN pip install --upgrade pip

# Python 패키지 설치 (torch 계열은 이미 베이스에 있음 → 여기선 설치 X)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir jupyterlab==4.0.11 \
 && pip cache purge

# (선택) gridwindow가 꼭 필요하면 pip로 직접 설치 (Git clone 불필요)
# headless 환경에선 시각화 호출(CV GUI)이 실패할 수 있으니 사용 시 주의
RUN pip install --no-cache-dir git+https://github.com/JunkyByte/python-gridwindow.git

# 런타임 안정/성능 환경변수
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    OMP_NUM_THREADS=1 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
    MPLBACKEND=Agg

# 포트 오픈(필요 시)
EXPOSE 8888

# Jupyter를 컨테이너 기본 실행으로 쓰고 싶다면 주석 해제
# CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]