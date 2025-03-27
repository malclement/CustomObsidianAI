from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import torch
from transformers import AutoConfig
from transformers import AutoModel
from transformers import AutoTokenizer


class ModelService:
    """
    Service for loading and caching Hugging Face models
    """

    def __init__(
        self, default_use_cache: bool = True, huggingface_token: Optional[str] = None
    ):
        """
        Initialize the ModelService

        Args:
        default_use_cache: Whether to use caching by default
        huggingface_token: Optional Hugging Face API token for private models
        """
        self.model_cache = {}
        self.default_use_cache = default_use_cache
        self.huggingface_token = huggingface_token

    async def get_model_with_processing(
        self,
        model_name: str,
        text: Optional[str] = None,
        use_cache: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Get a model and process text if provided

        Args:
        model_name: The name of the model on Hugging Face Hub
        text: Optional text to process with the model
        use_cache: Whether to use cached model if available (defaults to self.default_use_cache)

        Returns:
        Dict containing model information and processing results
        """
        if use_cache is None:
            use_cache = self.default_use_cache

        # Load the model and tokenizer
        model, tokenizer = await self._load_model_and_tokenizer(model_name, use_cache)

        # Get basic model info
        model_info = {
            "model_name": model_name,
            "model_type": model.config.model_type,
            "is_cached": model_name in self.model_cache,
        }

        # Process text if provided
        if text:
            model_info["processed_text"] = await self._process_text(
                model, tokenizer, text
            )

        return model_info

    async def _load_model_and_tokenizer(
        self, model_name: str, use_cache: bool
    ) -> Tuple[AutoModel, AutoTokenizer]:
        """
        Load a model and tokenizer from Hugging Face or cache

        Args:
        model_name: The name of the model on Hugging Face Hub
        use_cache: Whether to use cached model if available

        Returns:
        Tuple of (model, tokenizer)
        """
        # Check if model is already cached
        if use_cache and model_name in self.model_cache:
            return self.model_cache[model_name]

        # Load the config first to check if model exists and get metadata
        config_kwargs = {}
        if self.huggingface_token:
            config_kwargs["token"] = self.huggingface_token

        # Load the model and tokenizer from Hugging Face
        tokenizer = AutoTokenizer.from_pretrained(model_name, **config_kwargs)
        model = AutoModel.from_pretrained(model_name, **config_kwargs)

        # Cache the model
        if use_cache:
            self.model_cache[model_name] = (model, tokenizer)

        return model, tokenizer

    async def _process_text(
        self, model: AutoModel, tokenizer: AutoTokenizer, text: str
    ) -> Dict[str, Any]:
        """
        Process text with the given model and tokenizer

        Args:
        model: The model to use for processing
        tokenizer: The tokenizer to use for processing
        text: The text to process

        Returns:
        Dict containing processing results
        """
        # Tokenize and process the text
        inputs = tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )

        with torch.no_grad():
            outputs = model(**inputs)

        # Extract the last hidden state
        last_hidden_state = outputs.last_hidden_state

        # Get the embedding for the [CLS] token (first token)
        # This is often used as a sentence embedding
        embedding = last_hidden_state[:, 0, :].numpy().tolist()

        return {
            "input_text": text,
            "embedding_dim": len(embedding[0]),
            "embedding_preview": embedding[0][:5],  # First 5 values as preview
        }

    def clear_cache(self) -> None:
        """
        Clear the model cache to free up memory
        """
        self.model_cache = {}
