
export CUDA_VISIBLE_DEVICES=4,5

vllm serve QWen/Qwen3-VL-4B-Instruct \
  --port 8081 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 2048 \
  --tensor-parallel-size 4