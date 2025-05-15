from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

def merge_model():
    try:
        print("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            "tinyllama_model",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        print("Loading adapter...")
        model = PeftModel.from_pretrained(base_model, "final_model")
        
        print("Merging adapter with base model...")
        merged_model = model.merge_and_unload()
        
        print("Saving merged model...")
        merged_model.save_pretrained("merged_model")
        
        print("Loading and saving tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("final_model")
        tokenizer.save_pretrained("merged_model")
        
        print("Model successfully merged and saved to 'merged_model' directory")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    merge_model() 