# Required packages:
# pip install fastapi uvicorn vllm requests

# To run the server:
# python recipe_api.py

# Test the API:
# Health check: curl http://localhost:8000/health
# Generate tip: curl -X POST http://localhost:8000/generate_tip -H "Content-Type: application/json" -d '{"prompt": "How do I keep onions from making me cry?"}'

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import uvicorn
from typing import List, Optional
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Cooking Tips API")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://chefgpt-alb-1911712359.us-east-1.elb.amazonaws.com",  # Correct ALB
        "https://chefgpt-alb-1911712359.us-east-1.elb.amazonaws.com",  # In case you switch to HTTPS
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)


# Global variables for model and sampling parameters
model = None
sampling_params = None

# Define request and response models
class TipRequest(BaseModel):
    prompt: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 150  # Reduced for shorter tips
    top_p: Optional[float] = 0.95

class TipResponse(BaseModel):
    tip: str
    tokens_generated: int

def clean_response(text: str) -> str:
    """Clean the response text by removing prompt structure and formatting."""
    # Remove any text that looks like a prompt or instruction
    text = re.sub(r'(Tip:|Example:|Format:|Requirements:|Structure:).*?$', '', text, flags=re.MULTILINE)
    
    # Remove any numbered items or bullet points
    text = re.sub(r'^\d+\.\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[-•]\s*', '', text, flags=re.MULTILINE)
    
    # Remove any text in quotes
    text = re.sub(r'["\'].*?["\']', '', text)
    
    # Remove any text that looks like a prompt
    text = re.sub(r'Give a single, direct cooking tip about:.*?$', '', text, flags=re.MULTILINE)
    
    # Clean up whitespace
    text = ' '.join(text.split())
    
    # If the text is too short after cleaning, return a default message
    if len(text) < 10:
        return "I apologize, but I couldn't generate a specific tip for that. Could you please try rephrasing your question?"
    
    return text

@app.on_event("startup")
async def startup_event():
    """Initialize the model when the server starts"""
    global model, sampling_params
    try:
        logger.info("Loading cooking tips model...")
        model = LLM(
            model="ydanino/tinyllama-recipe-merged",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            trust_remote_code=True
        )
        
        # Default sampling parameters for generating tips
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=150  # Shorter responses for tips
        )
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

@app.post("/generate_tip", response_model=TipResponse)
async def generate_tip(request: TipRequest):
    """Generate a cooking tip based on the user's prompt"""
    if not model:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")
    
    try:
        # Log the incoming request
        logger.info(f"Received request with prompt: {request.prompt}")
        
        # Format the prompt to focus on cooking tips with strict instructions
        formatted_prompt = f"""Answer this cooking question: {request.prompt}
Give a single, direct answer. No lists, no examples, just one clear tip."""
        
        # Update sampling parameters if provided
        current_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=request.max_tokens
        )
        
        # Generate the tip
        outputs = model.generate([formatted_prompt], current_params)
        
        # Get the generated text and clean it
        generated_text = clean_response(outputs[0].outputs[0].text.strip())
        
        # If the response is empty or too short, try again with a more specific prompt
        if len(generated_text) < 10:
            logger.warning("Received empty or too short response, retrying with more specific prompt")
            formatted_prompt = f"""What is the best cooking tip for: {request.prompt}?
Answer in one sentence."""
            outputs = model.generate([formatted_prompt], current_params)
            generated_text = clean_response(outputs[0].outputs[0].text.strip())
        
        # Log the response
        logger.info(f"Generated tip: {generated_text}")
        
        return TipResponse(
            tip=generated_text,
            tokens_generated=len(outputs[0].outputs[0].token_ids)
        )
    except Exception as e:
        logger.error(f"Error generating tip: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Check if the API is healthy and the model is loaded"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "service": "cooking-tips-api"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 