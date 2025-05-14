from vllm import LLM, SamplingParams
import torch

def main():
    # Initialize the model
    print("Loading model...")
    model = LLM(
        model="ydanino/tinyllama-recipe-finetuned",  # Replace with your Hugging Face model path
        tensor_parallel_size=1,  # Number of GPUs to use
        gpu_memory_utilization=0.9,  # How much GPU memory to use
        trust_remote_code=True
    )
    
    # Define sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=512
    )
    
    # Test the model with a simple prompt
    print("Testing model with a prompt...")
    prompts = ["Write a recipe for chocolate chip cookies:"]
    
    # Generate
    outputs = model.generate(prompts, sampling_params)
    
    # Print results
    for output in outputs:
        print("\nGenerated text:")
        print(output.outputs[0].text)

if __name__ == "__main__":
    main() 