import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import os

def train_model(test_mode=False):
    try:
        # Check if we're on M1
        is_m1 = torch.backends.mps.is_available()
        device = "mps" if is_m1 else "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # Load the model and tokenizer from local directory
        print("Loading model and tokenizer...")
        model_path = "tinyllama_model"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto" if not is_m1 else None
        )
        
        if is_m1:
            model = model.to(device)

        # Load your dataset
        print("Loading dataset...")
        dataset = load_dataset('json', data_files='processed_data/tinyllama_training_data.jsonl')

        # Configure LoRA
        print("Configuring LoRA...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        # Apply LoRA
        print("Applying LoRA configuration...")
        model = get_peft_model(model, lora_config)
        
        # Enable training mode
        model.train()
        model.config.use_cache = False

        # Print trainable parameters
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params:,d} || "
            f"all params: {all_param:,d} || "
            f"trainable%: {100 * trainable_params / all_param:.2f}%"
        )

        def tokenize_function(examples):
            # Combine prompt and response
            texts = [f"{p} {r}" for p, r in zip(examples['prompt'], examples['response'])]
            
            # Tokenize
            tokenized = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Create labels
            tokenized["labels"] = tokenized["input_ids"].clone()
            
            # Set padding tokens to -100 in labels
            tokenized["labels"][tokenized["attention_mask"] == 0] = -100
            
            return tokenized

        # Tokenize dataset
        print("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir="./training_results",
            per_device_train_batch_size=2 if is_m1 else 4,
            gradient_accumulation_steps=8 if is_m1 else 4,
            learning_rate=2e-4,
            fp16=not is_m1,
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            remove_unused_columns=False,
            report_to="none",
            optim="adamw_torch",
            max_steps=1 if test_mode else -1,  # -1 means run for all epochs
            num_train_epochs=3  # Always set epochs, test mode will stop after 1 step
        )

        # Initialize trainer
        print("Initializing trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False
            )
        )

        # Train
        print("Starting training...")
        trainer.train()

        if not test_mode:
            # Save the final model
            print("Saving final model...")
            model.save_pretrained("final_model")
            tokenizer.save_pretrained("final_model")
        
        print("Setup verification completed successfully!" if test_mode else "Training completed successfully!")

    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        raise

if __name__ == "__main__":
    train_model(test_mode=False) 