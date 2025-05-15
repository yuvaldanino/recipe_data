from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi
import os

def push_to_hub():
    try:
        # Get your Hugging Face username
        api = HfApi()
        user = api.whoami()
        username = user['name']
        
        # Create a unique model name
        model_name = f"{username}/tinyllama-recipe-merged"
        
        print(f"Pushing model to {model_name}...")
        
        # Load the model and tokenizer
        model = AutoModelForCausalLM.from_pretrained("merged_model")
        tokenizer = AutoTokenizer.from_pretrained("merged_model")
        
        # Push to hub
        model.push_to_hub(model_name)
        tokenizer.push_to_hub(model_name)
        
        print(f"Model successfully saved to https://huggingface.co/{model_name}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    push_to_hub() 