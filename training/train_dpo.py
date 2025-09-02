# train_dpo.py
import os, torch, random
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported, PatchDPOTrainer, get_chat_template
from transformers import AutoTokenizer, TrainingArguments
from trl import DPOTrainer

### 1 — reproducibility
SEED = 42
torch.manual_seed(SEED); random.seed(SEED)

### 2 — model + tokenizer
BASE_MODEL = "unsloth/Qwen2.5-Coder-7B-Instruct"
MAXLEN     = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = BASE_MODEL,
    max_seq_length = MAXLEN,
    load_in_4bit   = True,          # NF4 quant during training
    dtype          = None,          # bf16 if supported, else fp16
)

model = FastLanguageModel.get_peft_model(
    model,
    r                      = 16,
    lora_alpha             = 32,
    lora_dropout           = 0.05,
    target_modules         = ["q_proj","k_proj","v_proj","o_proj",
                              "gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing = "unsloth",
    max_seq_length         = MAXLEN,
)

### 3 — dataset → {prompt, chosen, rejected}
tmpl = get_chat_template(tokenizer)
dataset = load_dataset("CyberNative/Code_Vulnerability_Security_DPO")

def to_dpo(ex):
    sys_msg = ex.get("system", "You are a secure-coding assistant.")
    prompt  = tmpl(
        [{"role":"system","content":sys_msg},
         {"role":"user",  "content":ex["question"]}],
        add_generation_prompt=True,
        tokenize=False
    )
    return {"prompt":prompt, "chosen":ex["chosen"], "rejected":ex["rejected"]}

if all(k in dataset for k in ["train", "validation", "test"]):
    train_ds = dataset["train"]
    val_ds = dataset["validation"]
    test_ds = dataset["test"]
else:
    # If validation/test splits not provided, create them from the train split
    temp_split = dataset["train"].train_test_split(test_size=0.2, seed=SEED)
    train_ds = temp_split["train"]
    remainder = temp_split["test"].train_test_split(test_size=0.5, seed=SEED)
    val_ds = remainder["train"]
    test_ds = remainder["test"]

train_ds = train_ds.map(to_dpo, remove_columns=train_ds.column_names, num_proc=4)
val_ds = val_ds.map(to_dpo, remove_columns=val_ds.column_names, num_proc=4)
test_ds = test_ds.map(to_dpo, remove_columns=test_ds.column_names, num_proc=4)

### 4 — DPO trainer
PatchDPOTrainer()                       # monkey-patch TRL for Unsloth

args = TrainingArguments(
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 8,
    num_train_epochs            = 3,
    learning_rate               = 1e-4,
    lr_scheduler_type           = "cosine",
    warmup_ratio                = 0.1,
    optim                       = "adamw_8bit",
    fp16                        = not is_bfloat16_supported(),
    bf16                        = is_bfloat16_supported(),
    logging_steps               = 10,
    output_dir                  = "qwen-sec-dpo",
    save_total_limit            = 2,
    report_to                   = "none",
    seed                        = SEED,
)

trainer = DPOTrainer(
    model             = model,
    ref_model         = None,        # Unsloth auto-clones a frozen reference
    beta              = 0.1,
    tokenizer         = tokenizer,
    train_dataset     = train_ds,
    eval_dataset      = val_ds,
    max_length        = MAXLEN,
    max_prompt_length = MAXLEN//2,
    args              = args,
)

trainer.train()
trainer.evaluate(eval_dataset=test_ds)
trainer.save_model("qwen-sec-dpo")     # LoRA adapters + 4-bit base
tokenizer.save_pretrained("qwen-sec-dpo")
