"""Kaggle kbench provider — uses Kaggle's built-in model pool."""

class KaggleProvider:
    """Adapter for Kaggle Benchmarks (kbench) model pool.
    
    No API key needed — models are available directly in the Kaggle environment.
    """
    
    def __init__(self, config):
        self.config = config
        self._pool = None
    
    def list_models(self):
        """List all available models in the Kaggle pool."""
        try:
            import kaggle_benchmarks as kbench
            if isinstance(kbench.llms, dict):
                return list(kbench.llms.keys())
            else:
                return [getattr(m, 'model', str(m)) for m in kbench.llms]
        except ImportError:
            raise RuntimeError("kaggle_benchmarks not available. Run inside a Kaggle notebook.")
    
    def get_model(self, model_name: str):
        """Get a model object from the Kaggle pool."""
        import kaggle_benchmarks as kbench
        if isinstance(kbench.llms, dict):
            return kbench.llms.get(model_name)
        else:
            for m in kbench.llms:
                if getattr(m, 'model', str(m)) == model_name:
                    return m
        raise ValueError(f"Model '{model_name}' not found in Kaggle pool")
    
    def prompt(self, model_name: str, system: str, user: str) -> str:
        """Send a prompt to a Kaggle model and return the response."""
        model = self.get_model(model_name)
        response = model.prompt(system=system, user=user)
        return str(response)
