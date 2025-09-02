# merge_and_quantise.py
import subprocess, shutil
from unsloth import FastLanguageModel
from transformers import AutoTokenizer

FINETUNE_DIR = "qwen-sec-dpo"      # output from training step
MERGED_DIR   = "qwen-sec-dpo-fp16"
GGUF_FP16    = "qwen-sec-dpo-fp16.gguf"
GGUF_Q6K     = "qwen-sec-dpo-q6k.gguf"

### 1 — merge LoRA → full-precision fp16
model, tok = FastLanguageModel.from_pretrained(
    FINETUNE_DIR, load_in_4bit=True
)
model.save_pretrained_merged(
    MERGED_DIR, tok, save_method="merged_16bit"
)

### 2 — HuggingFace fp16 → GGUF fp16
subprocess.check_call([
    "python", "llama.cpp/convert-hf-to-gguf.py",
    MERGED_DIR,
    "--outfile", GGUF_FP16,
    "--outtype", "f16"
])

### 3 — GGUF fp16 → q6_k quant
subprocess.check_call([
    "./llama.cpp/build/bin/quantize",
    GGUF_FP16, GGUF_Q6K, "q6_k"
])

print("✅  Done!  Final file:", GGUF_Q6K)
