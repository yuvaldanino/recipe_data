import requests
from typing import Optional, Dict, Any

class FoodTipsService:
    def __init__(self, base_url: str = "http://your-ec2-ip:8000"):
        self.base_url = base_url

    def get_food_tip(self, user_query: str, temperature: float = 0.7, 
                    max_tokens: int = 512, top_p: float = 0.95) -> Dict[str, Any]:
        """
        Get food tips and cooking advice based on user's query.
        
        Args:
            user_query (str): The user's question about food or cooking
            temperature (float): Controls randomness in generation (0.7 is good for creative responses)
            max_tokens (int): Maximum length of response
            top_p (float): Controls diversity of responses
            
        Returns:
            Dict containing the generated response and token count
        """
        try:
            # Format the prompt to encourage food tip responses
            formatted_prompt = f"User: {user_query}\nAssistant: Let me help you with that cooking tip:"
            
            response = requests.post(
                f"{self.base_url}/generate_recipe",  # We're reusing the endpoint but for chat
                json={
                    "prompt": formatted_prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p
                }
            )
            response.raise_for_status()
            result = response.json()
            
            # Clean up the response to remove the prompt
            if "recipe" in result:
                result["response"] = result["recipe"].replace(formatted_prompt, "").strip()
                del result["recipe"]
            
            return result
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to get food tip: {str(e)}")

    def check_health(self) -> bool:
        """
        Check if the food tips service is healthy.
        
        Returns:
            bool: True if service is healthy, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()["status"] == "healthy"
        except requests.exceptions.RequestException:
            return False 