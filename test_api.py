import requests
import json

def test_api_connection():
    # Replace with your EC2 instance's public IP
    server_ip = "3.230.173.141"  # This is the IP from your workspace path
    api_url = f'http://{server_ip}:8000/generate_tip'
    
    # Test data
    test_prompt = "What's the best way to store fresh herbs?"
    
    try:
        # Make the request
        response = requests.post(
            api_url,
            json={"prompt": test_prompt},
            headers={"Content-Type": "application/json"}
        )
        
        # Check if request was successful
        if response.status_code == 200:
            result = response.json()
            print("✅ API Connection Successful!")
            print("\nResponse:")
            print(f"Tip: {result['tip']}")
            print(f"Tokens generated: {result['tokens_generated']}")
        else:
            print(f"❌ Error: Status code {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Could not connect to the server")
        print("Please check:")
        print("1. The server is running (python recipe_api.py)")
        print("2. The security group allows inbound traffic on port 8000")
        print("3. The IP address is correct")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    test_api_connection() 