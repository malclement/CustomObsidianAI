import os
import threading
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import StoppingCriteria
from transformers import StoppingCriteriaList
from transformers import TextIteratorStreamer

from app.core.config import settings
from app.services.model_service import ModelService


class CompletionService:
    """
    Service for generating text completions using Hugging Face models
    """

    def __init__(
        self,
        model_service: ModelService,
        system_prompt_path: str = "prompts/system.txt",
        user_prompt_path: str = "prompts/user.txt",
    ):
        """
        Initialize the CompletionService

        Args:
        model_service: ModelService for loading and caching models
        system_prompt_path: Path to the system prompt template file
        user_prompt_path: Path to the user prompt template file
        """
        self.model_service = model_service
        self.completion_cache = {}

        # Load default prompts
        self.default_system_prompt = self._load_prompt(system_prompt_path)
        self.default_user_prompt = self._load_prompt(user_prompt_path)

    def _load_prompt(self, path: str) -> str:
        """
        Load a prompt template from a file

        Args:
        path: Path to the prompt template file

        Returns:
        The prompt template as a string
        """
        try:
            prompt_path = Path(path)
            if prompt_path.exists():
                with open(prompt_path, "r", encoding="utf-8") as f:
                    return f.read().strip()
            else:
                # Return default prompts if files don't exist
                if "system" in path:
                    return "You are a helpful AI assistant."
                else:
                    return "Process the following markdown content from a webpage and respond accordingly:\n\n{content}"
        except Exception as e:
            print(f"Error loading prompt from {path}: {e}")
            # Return default prompts on error
            if "system" in path:
                return "You are a helpful AI assistant."
            else:
                return "Process the following markdown content from a webpage and respond accordingly:\n\n{content}"

    async def generate_completion(
        self,
        model_name: str,
        user_content: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> Tuple[str, Dict[str, int]]:
        """
        Generate a completion for the given input

        Args:
        model_name: The name of the model on Hugging Face Hub
        user_content: The user's input content to process
        system_prompt: Optional system prompt to override default
        temperature: Sampling temperature (0.0 to 1.0)
        top_p: Nucleus sampling parameter (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate
        stream: Whether to stream the output

        Returns:
        Tuple containing the generated text and token usage statistics
        """
        # Use special loading for chat models
        lm_model, tokenizer = await self._load_completion_model(model_name)

        # Prepare system prompt (use provided or default)
        if system_prompt is None:
            system_prompt = self.default_system_prompt

        # Format the user prompt with the content
        formatted_user_prompt = self.default_user_prompt.format(content=user_content)

        # Create a prompt in the format expected by the model
        # This will adjust based on the model type
        if "llama" in model_name.lower():
            # LLaMA style formatting
            full_prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{formatted_user_prompt} [/INST]"
        elif any(name in model_name.lower() for name in ["mistral", "mixtral"]):
            # Mistral style formatting
            full_prompt = (
                f"<s>[INST] {system_prompt}\n\n{formatted_user_prompt} [/INST]"
            )
        else:
            # Generic formatting
            full_prompt = f"### System:\n{system_prompt}\n\n### User:\n{formatted_user_prompt}\n\n### Assistant:"

        # Tokenize the prompt
        inputs = tokenizer(full_prompt, return_tensors="pt").to(lm_model.device)
        prompt_length = inputs.input_ids.shape[1]

        # Set generation parameters
        gen_kwargs = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.eos_token_id,
        }

        # Add max tokens if specified
        if max_tokens is not None:
            gen_kwargs["max_new_tokens"] = max_tokens
        else:
            # Default to a reasonable value
            gen_kwargs["max_new_tokens"] = 1024

        # Generate text
        with torch.no_grad():
            if stream:
                # Implement streaming logic if needed
                raise NotImplementedError("Streaming not implemented yet")
            else:
                outputs = lm_model.generate(**gen_kwargs)

        # Extract only the generated response (not the prompt)
        generated_text = tokenizer.decode(
            outputs[0][prompt_length:], skip_special_tokens=True
        )

        # Calculate token usage
        completion_length = outputs.shape[1] - prompt_length
        usage = {
            "prompt_tokens": prompt_length,
            "completion_tokens": completion_length,
            "total_tokens": outputs.shape[1],
        }

        return generated_text.strip(), usage

    async def _load_completion_model(
        self, model_name: str
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load a model and tokenizer specifically configured for text generation

        Args:
        model_name: The name of the model on Hugging Face Hub

        Returns:
        Tuple of (model, tokenizer)
        """
        # Check if model is already in completion cache
        if model_name in self.completion_cache:
            return self.completion_cache[model_name]

        # Configure model loading parameters
        model_kwargs = {
            "torch_dtype": torch.float16,  # Use half precision for efficiency
            "low_cpu_mem_usage": True,  # More memory-efficient loading
            "device_map": "auto",  # Automatically determine device placement
        }

        # Add token if available
        if settings.HUGGINGFACE_TOKEN:
            model_kwargs["token"] = settings.HUGGINGFACE_TOKEN

        # Load the model and tokenizer from Hugging Face
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=settings.HUGGINGFACE_TOKEN
        )
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # Cache the model
        self.completion_cache[model_name] = (model, tokenizer)

        return model, tokenizer

    def clear_cache(self) -> None:
        """
        Clear the completion model cache to free up memory
        """
        self.completion_cache = {}
