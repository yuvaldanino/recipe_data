import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys
from huggingface_hub import snapshot_download

def download_and_save_model():
    try:
        # Check if we're on M1
        is_m1 = torch.backends.mps.is_available()
        device = "mps" if is_m1 else "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # Model name from Hugging Face
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        print("Step 1: Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        print("Step 2: Downloading model...")
        # First download the model files
        model_path = snapshot_download(
            repo_id=model_name,
            local_dir="tinyllama_model",
            local_dir_use_symlinks=False
        )
        
        print("Step 3: Loading model into memory...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if not is_m1 else torch.float32,  # Use float32 on M1
            device_map="auto" if not is_m1 else None,  # Don't use device_map on M1
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        if is_m1:
            model = model.to(device)  # Manually move to M1 device
        
        print("Step 4: Saving model and tokenizer...")
        # Save model and tokenizer
        model.save_pretrained("tinyllama_model")
        tokenizer.save_pretrained("tinyllama_model")
        
        print("Success! Model and tokenizer saved to 'tinyllama_model' directory")
        print("You can now use this model for training or inference")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    download_and_save_model() 