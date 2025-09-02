# serve.py
from llama_cpp import Llama

MODEL_PATH   = "qwen-sec-dpo-q6k.gguf"
CTX          = 4096        # adjust if VRAM is tight

llm = Llama(
    model_path       = MODEL_PATH,
    n_ctx            = CTX,
    n_gpu_layers     = -1,          # "-1" = all layers on GPU (CUDA build)
    verbose          = False,
)

def chat(user_msg, system_msg="You are a secure-coding assistant."):
    prompt = (f"<|im_start|>system\n{system_msg}<|im_end|>\n"
              f"<|im_start|>user\n{user_msg}<|im_end|>\n"
              f"<|im_start|>assistant\n")
    out = llm(
        prompt,
        temperature = 0.7,
        top_p       = 0.9,
        n_predict   = 512,
        stop        = ["<|im_end|>"]
    )
    return out["choices"][0]["text"]

if __name__ == "__main__":
    while True:
        try:
            q = input("\n👤  ")
            print("\n🤖 ", chat(q).strip(), "\n")
        except (KeyboardInterrupt, EOFError):
            break
