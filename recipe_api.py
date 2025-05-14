#   pip install fastapi uvicorn vllm
# pip install vllm 

# run the server "   python recipe_api.py"
# health check at Health check: curl http://localhost:8000/health

# curl -X POST http://localhost:8000/generate_recipe -H "Content-Type: application/json" -d '{"prompt": "Write a recipe for chocolate chip cookies:"}'

#get ubuntu 22.04
#update steps tom 

# # 1. System updates
# sudo apt update
# sudo apt upgrade -y

# # 2. Install Python and pip
# sudo apt install -y python3-pip python3-venv

# # 3. Install CUDA toolkit
# sudo apt install -y nvidia-cuda-toolkit

# # 4. Create and activate virtual environment
# python3 -m venv recipe_env
# source recipe_env/bin/activate

# # 5. Install Hugging Face CLI and login
# pip install --upgrade huggingface_hub
# huggingface-cli login
# # (Enter your token from huggingface.co)

# # 6. Install our server requirements
# pip install fastapi uvicorn vllm

# # 7. Run the server
# python recipe_api.py


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import uvicorn
from typing import List, Optional

app = FastAPI(title="Recipe Generation API")

# Initialize model and sampling params
model = None
sampling_params = None

class RecipeRequest(BaseModel):
    prompt: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    top_p: Optional[float] = 0.95

class RecipeResponse(BaseModel):
    recipe: str
    tokens_generated: int

@app.on_event("startup")
async def startup_event():
    global model, sampling_params
    try:
        # Initialize the model
        print("Loading model...")
        model = LLM(
            model="ydanino/tinyllama-recipe-finetuned",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            trust_remote_code=True
        )
        
        # Default sampling parameters
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=512
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

@app.post("/generate_recipe", response_model=RecipeResponse)
async def generate_recipe(request: RecipeRequest):
    try:
        # Update sampling parameters if provided
        current_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens
        )
        
        # Generate recipe
        outputs = model.generate([request.prompt], current_params)
        
        # Get the generated text
        generated_text = outputs[0].outputs[0].text
        
        return RecipeResponse(
            recipe=generated_text,
            tokens_generated=len(outputs[0].outputs[0].token_ids)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 