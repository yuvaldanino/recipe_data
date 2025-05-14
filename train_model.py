import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

# Load the prepared JSONL data
dataset = load_dataset('json', data_files='processed_data/tinyllama_training_data.jsonl')

# Load the TinyLlama model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
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
model.train()  # Make sure the model is in training mode
model.config.use_cache = False  # Disable caching (required for gradient checkpointing)

# Define a function to tokenize the dataset
def tokenize_function(examples):
    prompts = examples['prompt']
    responses = examples['response']
    full_texts = [p + ' ' + r for p, r in zip(prompts, responses)]
    
    # Tokenize with padding and truncation
    model_inputs = tokenizer(
        full_texts,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    # Create labels by copying input_ids
    model_inputs["labels"] = model_inputs["input_ids"].clone()
    
    # Set padding tokens to -100 in labels
    model_inputs["labels"][model_inputs["attention_mask"] == 0] = -100
    
    return model_inputs

# Tokenize the dataset
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

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
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
)

# Train the model
trainer.train()

# Merge LoRA weights into the base model
model = model.merge_and_unload()

# Save locally
model.save_pretrained("merged_model")
tokenizer.save_pretrained("merged_model")

print("Training complete and model saved locally!") 