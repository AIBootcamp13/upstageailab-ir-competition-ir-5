#!/bin/bash

# 환경 설정 스크립트
# Python 3.11, Poetry 설치, /workspace 작업 환경 구성

set -e  # 에러 발생시 스크립트 중단

echo "시스템 패키지 업데이트 중..."
apt update
apt upgrade -y

apt-get update
apt-get upgrade -y

echo "Python 빌드에 필요한 의존성 패키지 설치 중..."
apt install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
libffi-dev liblzma-dev git-all

echo "🚀 환경 설정을 시작합니다..."

cd /root

# 1. /workspace 디렉토리 생성
echo "📁 /workspace 디렉토리 생성 중..."
mkdir -p /workspace
chmod 755 /workspace

# 2. conda 환경 생성 (Python 3.11)
echo "📦 Python 3.11 conda 환경 생성 중..."
conda create -n py311 python=3.11.11 -y

# 3. 생성된 환경 활성화 확인
echo "🔄 conda 환경 확인 중..."
source /opt/conda/etc/profile.d/conda.sh
conda activate py311
python --version

# 4. pip 업그레이드
echo "⬆️ pip 및 setuptools 업그레이드 중..."
pip install -U pip setuptools

# 6. Poetry 설치 (root 계정의 홈 디렉토리에)
echo "📝 Poetry 설치 중..."
# Poetry를 /root에 설치하도록 POETRY_HOME 설정
export POETRY_HOME="/root/.poetry"
curl -sSL https://install.python-poetry.org | python3 -

# 7. /root/.bashrc 수정 - 기존 HOME 설정 제거하고 새 설정 추가
echo "⚙️ .bashrc 설정 수정 중..."

# 기존 잘못된 HOME 설정 제거
sed -i '/export HOME=\/data\/ephemeral\/home/d' /root/.bashrc
sed -i '/mkdir -p \$HOME 2> \/dev\/null/d' /root/.bashrc
sed -i '/cd \$HOME/d' /root/.bashrc

# 기존 우리가 추가한 설정이 있다면 제거
sed -i '/PYTORCH_VERSION=2.1.0/d' /root/.bashrc
sed -i '/# Custom Environment Setup/,/# End Custom Environment Setup/d' /root/.bashrc

# 새로운 설정을 .bashrc 끝에 추가
cat >> /root/.bashrc << 'EOF'

# Custom Environment Setup
# HOME을 올바르게 설정 (root 계정)
export HOME=/root

# Poetry PATH 추가
export PATH="/root/.poetry/bin:$PATH"

# PYTHONPATH PATH 추가
export PYTHONPATH="/workspace/upstageailab-ir-competition-ir-5"

# conda 환경 자동 활성화
source /opt/conda/etc/profile.d/conda.sh
conda activate py311

# SSH 로그인시 /workspace로 이동
if [[ $- == *i* ]] && [[ -n "$SSH_CONNECTION" ]]; then
    cd /workspace
fi

# End Custom Environment Setup
EOF

source /root/.bashrc

poetry config virtualenvs.in-project true
poetry config virtualenvs.in-project true

echo ""
echo "🎉 환경 설정이 완료되었습니다!"
echo ""
echo "📋 설정 내용:"
echo "  ✓ Python 3.11 conda 환경 (py311) 생성"
echo "  ✓ Poetry 설치 (/root/.poetry/)"
echo "  ✓ /workspace 디렉토리 생성"
echo "  ✓ HOME 경로 수정 (/root 로 설정)"
echo "  ✓ PYTHONPATH 를 clone 한 repository 로 설정"
echo "  ✓ SSH 로그인시 자동으로 /workspace로 이동"
echo ""
echo "🔄 변경사항 적용을 위해 다음을 실행하세요."
echo "  SSH 재접속"
