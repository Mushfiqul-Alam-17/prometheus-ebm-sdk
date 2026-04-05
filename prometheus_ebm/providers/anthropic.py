"""Anthropic provider — direct Claude API access."""


class AnthropicProvider:
    """Adapter for Anthropic's Claude API.
    
    Requires: pip install anthropic
    Get your API key at: https://console.anthropic.com/
    """
    
    def __init__(self, config):
        self.config = config
        self.api_key = config.api_key
        if not self.api_key:
            raise ValueError("Anthropic provider requires an API key. Set config.api_key")
        
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("Install anthropic: pip install prometheus-ebm[anthropic]")
    
    def list_models(self):
        """List common Anthropic models."""
        return [
            "claude-opus-4-6-20260201",
            "claude-sonnet-4-6-20260201",
            "claude-haiku-4-5-20251001",
        ]
    
    def prompt(self, model_name: str, system: str, user: str) -> str:
        """Send a prompt to Claude."""
        message = self.client.messages.create(
            model=model_name,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=0.0,
        )
        return message.content[0].text
