#make virtual environment before and downloda all the stuff in recipe_api.py
#then run the server with "python recipe_api_flask.py"
 # Hugging Face for model access
# pip install --upgrade huggingface_hub

# # Login to Hugging Face (you'll need your token)
# huggingface-cli login

# # Server and model requirements
# pip install flask
# pip install vllm




from flask import Flask, request, jsonify
from vllm import LLM, SamplingParams
import torch

app = Flask(__name__)

# Initialize model and sampling params
model = None
sampling_params = None

def init_model():
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

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })

@app.route('/generate_recipe', methods=['POST'])
def generate_recipe():
    try:
        # Get request data
        data = request.get_json()
        
        # Validate required fields
        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing 'prompt' in request"}), 400
            
        # Get parameters with defaults
        temperature = data.get('temperature', 0.7)
        max_tokens = data.get('max_tokens', 512)
        top_p = data.get('top_p', 0.95)
        
        # Update sampling parameters
        current_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        
        # Generate recipe
        outputs = model.generate([data['prompt']], current_params)
        
        # Get the generated text
        generated_text = outputs[0].outputs[0].text
        
        return jsonify({
            "recipe": generated_text,
            "tokens_generated": len(outputs[0].outputs[0].token_ids)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Initialize model before starting server
    init_model()
    
    # Run the server
    app.run(host='0.0.0.0', port=8000) 