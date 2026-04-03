"""OpenAI provider — direct GPT API access."""

import os
from types import SimpleNamespace


class OpenAIProvider:
    """Adapter for OpenAI's GPT API.
    
    Requires: pip install openai
    Get your API key at: https://platform.openai.com/api-keys
    """
    
    def __init__(self, config=None, *, api_key=None, base_url=None, model_id=None):
        # Support both SDK-native construction (OpenAIProvider(config)) and direct
        # construction for OpenAI-compatible endpoints such as Groq.
        if config is None:
            cfg_api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.config = SimpleNamespace(api_key=cfg_api_key, api_base_url=base_url)
        else:
            self.config = config
            if getattr(self.config, "api_base_url", None) is None and base_url is not None:
                self.config.api_base_url = base_url
            if getattr(self.config, "api_key", None) is None and api_key is not None:
                self.config.api_key = api_key

        self.default_model = model_id
        self.api_key = self.config.api_key
        if not self.api_key:
            raise ValueError("OpenAI provider requires an API key. Set config.api_key")
        
        try:
            import openai
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.config.api_base_url
            )
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
        selected_model = model_name or self.default_model
        if not selected_model:
            raise ValueError("No model name supplied. Provide model_name or initialize with model_id.")

        response = self.client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=2048,
        )
        return response.choices[0].message.content
