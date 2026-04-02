"""OpenAI provider — direct GPT API access."""


class OpenAIProvider:
    """Adapter for OpenAI's GPT API.
    
    Requires: pip install openai
    Get your API key at: https://platform.openai.com/api-keys
    """
    
    def __init__(self, config):
        self.config = config
        self.api_key = config.api_key
        if not self.api_key:
            raise ValueError("OpenAI provider requires an API key. Set config.api_key")
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("Install openai: pip install prometheus-ebm[openai]")
    
    def list_models(self):
        """List common OpenAI models."""
        return [
            "gpt-5.4",
            "gpt-5.4-thinking",
            "o3-mini",
        ]
    
    def prompt(self, model_name: str, system: str, user: str) -> str:
        """Send a prompt to GPT."""
        response = self.client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=2048,
        )
        return response.choices[0].message.content
