from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from services.recipe_service import FoodTipsService
import json
import requests

# Initialize the service
food_tips_service = FoodTipsService(base_url="http://your-ec2-ip:8000")

# Recipe API configuration
RECIPE_API_URL = "http://localhost:8000"  # Update this with your actual API URL

@csrf_exempt
@require_http_methods(["POST"])
def chat_with_food_tips(request):
    """
    Handle chat messages and return food tips.
    Expected JSON body: {"message": "user's question about food"}
    """
    try:
        data = json.loads(request.body)
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return JsonResponse({
                "error": "Message is required"
            }, status=400)
        
        # Get response from the food tips service
        response = food_tips_service.get_food_tip(user_message)
        
        return JsonResponse({
            "response": response["response"],
            "tokens_generated": response["tokens_generated"]
        })
        
    except json.JSONDecodeError:
        return JsonResponse({
            "error": "Invalid JSON"
        }, status=400)
    except Exception as e:
        return JsonResponse({
            "error": str(e)
        }, status=500)

@require_http_methods(["GET"])
def check_service_health(request):
    """Check if the recipe API is healthy"""
    try:
        response = requests.get(f"{RECIPE_API_URL}/health")
        is_healthy = response.status_code == 200
        return JsonResponse({
            "status": "healthy" if is_healthy else "unhealthy"
        })
    except:
        return JsonResponse({
            "status": "unhealthy"
        })

@csrf_exempt
@require_http_methods(["POST"])
def vllm_chat_view(request):
    """
    Handle chat messages and return cooking tips from the recipe API.
    Expected JSON body: {"message": "user's question about cooking"}
    """
    try:
        data = json.loads(request.body)
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return JsonResponse({
                "error": "Message is required"
            }, status=400)
        
        # Format the prompt to focus on cooking tips rather than full recipes
        prompt = f"Give me a helpful cooking tip about: {user_message}. Keep it concise and practical."
        
        # Call the recipe API
        response = requests.post(
            f"{RECIPE_API_URL}/generate_recipe",
            json={
                "prompt": prompt,
                "temperature": 0.7,
                "max_tokens": 256,  # Reduced max tokens since we want shorter tips
                "top_p": 0.95
            }
        )
        
        if response.status_code != 200:
            return JsonResponse({
                "error": "Failed to get response from recipe API"
            }, status=500)
        
        recipe_data = response.json()
        
        return JsonResponse({
            "response": recipe_data["recipe"],
            "tokens_generated": recipe_data["tokens_generated"]
        })
        
    except json.JSONDecodeError:
        return JsonResponse({
            "error": "Invalid JSON"
        }, status=400)
    except Exception as e:
        return JsonResponse({
            "error": str(e)
        }, status=500)

@require_http_methods(["GET"])
def check_recipe_service_health(request):
    """Check if the recipe API is healthy"""
    try:
        response = requests.get(f"{RECIPE_API_URL}/health")
        is_healthy = response.status_code == 200
        return JsonResponse({
            "status": "healthy" if is_healthy else "unhealthy"
        })
    except:
        return JsonResponse({
            "status": "unhealthy"
        }) 