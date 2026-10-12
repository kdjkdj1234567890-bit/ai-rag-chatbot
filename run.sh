#!/bin/bash
# AI RAG 챗봇 (시연용 프로토타입) 실행 스크립트
set -e
cd "$(dirname "$0")"
pip install --quiet --break-system-packages flask 2>/dev/null || pip install --quiet flask
echo "▶ 서버 시작: http://127.0.0.1:5001"
python3 app.py
