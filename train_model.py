import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2SeqLM,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Load the prepared JSONL data
dataset = load_dataset('json', data_files='processed_data/tinyllama_training_data.jsonl')

# Load the TinyLlama model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Prepare the model for LoRA fine-tuning
model = get_peft_model(model, lora_config)
model.train()                      # Make sure the model is in training mode
model.config.use_cache = False    # Disable caching (required for gradient checkpointing)
model.config.return_dict = True   # Ensure outputs are returned as a dict

# Define a function to tokenize the dataset
def tokenize_function(examples):
    prompts = examples['prompt']
    responses = examples['response']
    full_texts = [p + ' ' + r for p, r in zip(prompts, responses)]
    tokens = tokenizer(full_texts, truncation=True, padding="max_length", max_length=512)
    
    # Set labels to -100 for prompt tokens (we don't want to compute loss on these)
    labels = tokens["input_ids"].copy()
    for i, text in enumerate(full_texts):
        prompt_tokens = tokenizer(prompts[i], truncation=True, padding="max_length", max_length=512)["input_ids"]
        labels[i][:len(prompt_tokens)] = [-100] * len(prompt_tokens)
    
    tokens["labels"] = labels
    return tokens

# Tokenize the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=True,  # Enable mixed precision training
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    report_to="none"
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    data_collator=DataCollatorForSeq2SeqLM(tokenizer=tokenizer, padding=True)
)

# Train the model
trainer.train()

# Merge LoRA weights into the base model
model = model.merge_and_unload()

# Save locally
model.save_pretrained("merged_model")
tokenizer.save_pretrained("merged_model")

# Save the merged model to Hugging Face Hub
# Note: You need to be logged in to Hugging Face Hub
# Run: huggingface-cli login


print("Training complete and model saved locally and to Hugging Face Hub!") 