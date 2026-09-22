#!/bin/bash

# 设置日志目录
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

# 初始化 conda（若使用 conda 环境需要此行；若已在全局环境可省略）
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "=== 1. 正在启动 RAG API (环境 A) ==="
conda activate rag  # 替换为你的环境 A 名称
nohup python ./rag/rag_api.py > "$LOG_DIR/rag_api.log" 2>&1 &
sleep 5

echo "=== 2. 正在启动 vLLM (环境 B) ==="
conda activate vllm  # 替换为你的环境 B 名称
nohup bash ./start_vllm.sh > "$LOG_DIR/vllm.log" 2>&1 &
sleep 20

echo "=== 3. 正在启动 Streamlit 前端 (环境 C) ==="
conda activate app  # 替换为你的环境 C 名称
streamlit run ./app.py
