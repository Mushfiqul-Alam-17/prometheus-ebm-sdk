"""OpenRouter provider — broad multi-model access via OpenRouter API."""
import requests


class OpenRouterProvider:
    """Adapter for OpenRouter API (access to 100+ models with one API key).
    
    Get your API key at: https://openrouter.ai/keys
    """
    
    BASE_URL = "https://openrouter.ai/api/v1"
    
    def __init__(self, config):
        self.config = config
        self.api_key = config.api_key
        if not self.api_key:
            raise ValueError("OpenRouter requires an API key. Set config.api_key")
    
    def list_models(self):
        """List available models on OpenRouter."""
        resp = requests.get(
            f"{self.BASE_URL}/models",
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        resp.raise_for_status()
        return [m['id'] for m in resp.json().get('data', [])]
    
    def prompt(self, model_name: str, system: str, user: str) -> str:
        """Send a prompt to an OpenRouter model."""
        resp = requests.post(
            f"{self.BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.0,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()['choices'][0]['message']['content']
