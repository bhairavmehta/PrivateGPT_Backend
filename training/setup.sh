# CUDA 11.8 / 12.x works
python -m venv qwen-sec-env
source qwen-sec-env/bin/activate

pip install --upgrade "unsloth[torch]" datasets trl bitsandbytes \
             accelerate einops

# For the GGUF tool-chain and inference helpers:
pip install sentencepiece peft llama-cpp-python>=0.2.68 # CPU or GPU wheels are auto-picked


git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -S . -B build -DLLAMA_CUDA=ON -DGGML_CUBLAS=ON      # remove flags for CPU-only
cmake --build build --target server quantize -j
cd ..
