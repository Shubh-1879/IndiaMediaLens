import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# ==========================================
# 1. Model & Tokenizer Initialization
# ==========================================
model_id = "mistralai/Mistral-7B-v0.1"

print(f"Loading Tokenizer for {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
# Mistral doesn't have a default pad token, so we map it to the EOS token
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Loading 16-bit Native Model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16, 
    device_map="auto" # Automatically routes to your 32GB V100
)

# Crucial for VRAM management: Disables caching and enables checkpointing
model.config.use_cache = False 
model.gradient_checkpointing_enable()

# ==========================================
# 2. LoRA (Low-Rank Adaptation) Configuration
# ==========================================
# We target the attention and MLP layers to give the model the best
# chance at learning the nuances of laptop aspect extraction.
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model.enable_input_require_grads()

model = get_peft_model(model, lora_config)

for name, param in model.named_parameters():
    if param.requires_grad:
        param.data = param.data.to(torch.float32)

model.print_trainable_parameters()



# ==========================================
# 3. Dataset Loading
# ==========================================
print("Loading laptop ABSA dataset...")
# SFTTrainer expects a dataset with a specific column containing the full prompt.
# We assume your JSONL has a column named "text".
dataset = load_dataset("json", data_files="absa_train.jsonl", split="train")

# ==========================================
# 4. Training Arguments (32GB V100 Optimized)
# ==========================================
training_args = TrainingArguments(
    output_dir="./mistral_laptop_absa_full_checkpoints",
    per_device_train_batch_size=1,       # MUST stay at 1 to prevent OOM
    gradient_accumulation_steps=4,       # Simulates a batch size of 4
    gradient_checkpointing=True,
    fp16=True,                           # Mixed precision training
    optim="adamw_torch",
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=50,
    save_strategy="epoch",
    num_train_epochs=3,
    report_to="none"                     # Change to "wandb" if you set up Weights & Biases
)

# ==========================================
# 5. Initialize SFTTrainer
# ==========================================
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    dataset_text_field="text",           # The key in your JSONL containing the prompt
    max_seq_length=512,                  # Keep <= 1024 to save VRAM on the V100
    tokenizer=tokenizer,
    args=training_args,
)

# ==========================================
# 6. Execute Training
# ==========================================
print("Starting training loop...")
trainer.train()

# Save the final, fully-trained adapter weights
print("Saving final model adapters...")
trainer.model.save_pretrained("./mistral_laptop_absa_full_final")
tokenizer.save_pretrained("./mistral_laptop_absa_full_final")
print("Training Complete!")
