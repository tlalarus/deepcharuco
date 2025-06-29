#!/bin/bash

ENV_NAME="deepcharuco-env"
PYTHON_BIN="python3.8"
REQUIREMENTS_FILE="requirements.txt"

# 1. 기존 가상환경 디렉토리 삭제 (있다면)
if [ -d "$ENV_NAME" ]; then
    echo "🔁 기존 가상환경 '$ENV_NAME' 삭제 중..."
    rm -rf "$ENV_NAME"
fi

# 2. 가상환경 생성
echo "🚀 '$ENV_NAME' 가상환경 생성 중..."
$PYTHON_BIN -m venv "$ENV_NAME"

# 3. 가상환경 활성화
echo "✅ 가상환경 활성화 중..."
source "$ENV_NAME/bin/activate"

# 4. pip 최신화
echo "📦 pip 최신화..."
pip install --upgrade pip

# 5. requirements.txt 설치
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "📥 패키지 설치 중 ($REQUIREMENTS_FILE)..."
    pip install -r "$REQUIREMENTS_FILE"
else
    echo "⚠️ '$REQUIREMENTS_FILE' 파일이 없습니다. 설치 생략."
fi

# 6. PyTorch 설치 (CPU-only, 필요 시 수정)
echo "🔥 PyTorch 설치 중..."
pip install torch==1.13.1 torchvision==0.14.1

# 7. 완료 안내
echo ""
echo "✅ 가상환경 '$ENV_NAME' 설정 완료!"
echo "👉 사용하려면 다음 명령어 실행:"
echo "source $ENV_NAME/bin/activate"
