from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

def merge_lora_weights():
    try:
        print("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            "tinyllama_model",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        print("Loading LoRA weights...")
        model = PeftModel.from_pretrained(base_model, "final_model")
        
        print("Merging LoRA weights with base model...")
        merged_model = model.merge_and_unload()
        
        print("Saving merged model...")
        merged_model.save_pretrained("merged_model")
        
        # Also save the tokenizer
        tokenizer = AutoTokenizer.from_pretrained("tinyllama_model")
        tokenizer.save_pretrained("merged_model")
        
        print("Model successfully merged and saved to 'merged_model' directory!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    merge_lora_weights() 