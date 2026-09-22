#!/bin/bash

# ==============================================================================
# 脊柱 AI 辅助诊断系统 - 视觉大模型 (Qwen3-VL) API 启动脚本
# ==============================================================================

# 1. 指定使用的显卡编号 (如果只有单卡请保持为 0，多卡可选 0,1,2,3 等)
export CUDA_VISIBLE_DEVICES=0

# 2. 打印启动提示
echo "使用的显卡: GPU $CUDA_VISIBLE_DEVICES"
echo "API 端口: 8000"
echo "--------------------------------------------------------"

# 3. 执行 vLLM 启动命令
python -m vllm.entrypoints.openai.api_server \
    --model ../SpineAgent/model_weights/diagnosis_weights \
    --served-model-name qwen-spine-vlm \
    --dtype bfloat16 \
    --trust-remote-code \
    --limit-mm-per-prompt '{"image": 30}' \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.5 \
    --port 8000