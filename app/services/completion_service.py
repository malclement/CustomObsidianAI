import os
import threading
import time
import traceback
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import torch
from transformers import AutoModel
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import pipeline
from transformers import StoppingCriteria
from transformers import StoppingCriteriaList
from transformers import TextIteratorStreamer

from app.core.config import settings
from app.services.model_service import ModelService
from app.utils.logger import get_logger

logger = get_logger()


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
            logger.error(f"Error loading prompt from {path}: {e}")
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
        try:
            # Check if model is a text generation model
            is_causal_lm = self._is_causal_lm_model(model_name)

            if is_causal_lm:
                # Use special loading for chat/causal LM models
                return await self._generate_with_causal_lm(
                    model_name=model_name,
                    user_content=user_content,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stream=stream,
                )
            else:
                # Fallback for encoder models like BERT
                return await self._generate_with_encoder(
                    model_name=model_name,
                    user_content=user_content,
                    system_prompt=system_prompt,
                )
        except Exception as e:
            logger.error(
                f"Error generating completion: {str(e)}\n{traceback.format_exc()}"
            )
            raise

    async def _generate_with_causal_lm(
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
        Generate text with a causal language model (GPT, LLaMA, etc)
        """
        try:
            # Log generation start
            logger.info(f"Starting text generation with model: {model_name}")

            # Load model and tokenizer
            lm_model, tokenizer = await self._load_causal_lm_model(model_name)

            # Prepare system prompt (use provided or default)
            if system_prompt is None:
                system_prompt = self.default_system_prompt

            # Format the user prompt with the content
            formatted_user_prompt = self.default_user_prompt.format(
                content=user_content
            )

            # Create a prompt in the format expected by the model
            # This will adjust based on the model type
            if "llama" in model_name.lower() or "tinyllama" in model_name.lower():
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

            logger.debug(f"Using prompt template for model {model_name}")

            # Tokenize the prompt
            inputs = tokenizer(full_prompt, return_tensors="pt").to(lm_model.device)
            prompt_length = inputs.input_ids.shape[1]

            logger.debug(f"Prompt tokenized with {prompt_length} tokens")

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

            logger.debug(
                f"Generation parameters: temp={temperature}, top_p={top_p}, max_tokens={gen_kwargs['max_new_tokens']}"
            )

            # Generate text
            with torch.no_grad():
                if stream:
                    # Implement streaming logic if needed
                    raise NotImplementedError("Streaming not implemented yet")
                else:
                    # Adding safety mechanism for small models like TinyLlama
                    try:
                        outputs = lm_model.generate(**gen_kwargs)
                    except RuntimeError as e:
                        if (
                            "CUDA out of memory" in str(e)
                            or "not enough memory" in str(e).lower()
                        ):
                            logger.warning(
                                f"Memory error with {model_name}, falling back to CPU"
                            )
                            # Try again with CPU
                            lm_model = lm_model.to("cpu")
                            inputs = {k: v.to("cpu") for k, v in inputs.items()}
                            gen_kwargs["input_ids"] = inputs["input_ids"]
                            gen_kwargs["attention_mask"] = inputs["attention_mask"]
                            outputs = lm_model.generate(**gen_kwargs)
                        else:
                            raise

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

            logger.info(f"Generated {completion_length} tokens with model {model_name}")

            return generated_text.strip(), usage

        except Exception as e:
            logger.error(
                f"Error in _generate_with_causal_lm: {str(e)}\n{traceback.format_exc()}"
            )
            raise

    async def _generate_with_encoder(
        self,
        model_name: str,
        user_content: str,
        system_prompt: Optional[str] = None,
    ) -> Tuple[str, Dict[str, int]]:
        """
        Generate a response using an encoder model like BERT
        For encoder-only models, we'll just summarize the content
        """
        try:
            # Use the summarization pipeline as a fallback
            try:
                # Load the model and tokenizer from the model service
                model, tokenizer = await self.model_service._load_model_and_tokenizer(
                    model_name, True
                )

                # For encoder models, we'll create a simple summarization response
                max_length = min(len(user_content.split()), 500)  # Reasonable default

                # Create a basic response
                response = (
                    f"Content analysis from model '{model_name}':\n\n"
                    f"The provided content contains approximately {len(user_content.split())} words. "
                    f"This model ({model.config.model_type}) is an encoder model and "
                    f"doesn't generate text directly, but can be used for analysis and embedding."
                )

                # Estimate token counts
                prompt_tokens = len(tokenizer.encode(user_content))
                completion_tokens = len(tokenizer.encode(response))

                usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }

                return response, usage

            except Exception as e:
                logger.error(
                    f"Error in encoder generation: {str(e)}\n{traceback.format_exc()}"
                )
                # If something goes wrong, return a generic response
                fallback_response = (
                    f"Unable to process content with model '{model_name}'. "
                    f"This model may not support text generation. Error: {str(e)}"
                )

                # Estimate token counts for the fallback
                usage = {
                    "prompt_tokens": len(user_content.split()),
                    "completion_tokens": len(fallback_response.split()),
                    "total_tokens": len(user_content.split())
                    + len(fallback_response.split()),
                }

                return fallback_response, usage
        except Exception as e:
            logger.error(
                f"Error in _generate_with_encoder: {str(e)}\n{traceback.format_exc()}"
            )
            raise

    def _is_causal_lm_model(self, model_name: str) -> bool:
        """
        Check if a model is a causal language model that can generate text

        Args:
        model_name: The name of the model on Hugging Face Hub

        Returns:
        True if the model is a causal LM, False otherwise
        """
        # These model families are known to be causal language models
        causal_lm_families = [
            "gpt",
            "llama",
            "mistral",
            "mixtral",
            "falcon",
            "mpt",
            "opt",
            "bloom",
            "phi",
            "gemma",
            "claude",
            "llm",
            "gptq",
            "starcoder",
            "pythia",
            "vicuna",
            "stablelm",
            "tiny",
        ]

        # Check if model name contains any of the causal LM identifiers
        return any(family in model_name.lower() for family in causal_lm_families)

    async def _load_causal_lm_model(
        self, model_name: str
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load a causal language model and tokenizer specifically configured for text generation

        Args:
        model_name: The name of the model on Hugging Face Hub

        Returns:
        Tuple of (model, tokenizer)
        """
        try:
            # Check if model is already in completion cache
            if model_name in self.completion_cache:
                logger.info(f"Using cached model: {model_name}")
                return self.completion_cache[model_name]

            logger.info(f"Loading model: {model_name}")

            # Check if we should use 8-bit or 4-bit quantization for larger models
            use_quantization = False
            quantization_type = None

            # Size detection for quantization decision
            if any(size in model_name.lower() for size in ["7b", "13b", "70b"]):
                # Larger models should use quantization
                use_quantization = True

                # Choose quantization type based on size
                if "70b" in model_name.lower():
                    quantization_type = "4bit"
                else:
                    quantization_type = "8bit"

            # Configure model loading parameters
            model_kwargs = {
                "device_map": "auto"  # Automatically determine device placement
            }

            # Apply quantization if needed
            if use_quantization:
                if quantization_type == "8bit":
                    model_kwargs["load_in_8bit"] = True
                elif quantization_type == "4bit":
                    model_kwargs["load_in_4bit"] = True
                    model_kwargs["bnb_4bit_compute_dtype"] = torch.float16
                else:
                    # Default to FP16 for smaller models
                    model_kwargs["torch_dtype"] = torch.float16
            else:
                # For small models like TinyLlama, use regular loading
                pass

            # Add token if available - using proper token handling
            token = settings.HUGGINGFACE_TOKEN or os.environ.get(
                "HUGGINGFACE_TOKEN", None
            )

            # Handle authentication properly
            use_auth = False

            # Check if token is provided
            if token:
                model_kwargs["token"] = token
                use_auth = True
                logger.info(f"Using Hugging Face token for model: {model_name}")

            # Check if model requires authentication
            if "tinyllama" in model_name.lower():
                # TinyLlama doesn't require authentication, so remove token if it's causing issues
                if "token" in model_kwargs:
                    logger.info(f"Removing token for public model: {model_name}")
                    del model_kwargs["token"]
                use_auth = False

            # Log what we're doing
            if use_quantization:
                logger.info(
                    f"Loading model {model_name} with {quantization_type} quantization"
                )
            else:
                logger.info(f"Loading model {model_name} without quantization")

            # Create tokenizer arguments
            tokenizer_kwargs = {}
            if use_auth and token:
                tokenizer_kwargs["token"] = token

            # Try loading with authentication first
            try:
                logger.debug(f"Loading tokenizer for {model_name}")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, **tokenizer_kwargs
                )

                # Handle special case for TinyLlama which might need padding token
                if "tinyllama" in model_name.lower():
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token

                logger.debug(f"Loading model for {model_name}")
                model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

            except Exception as e:
                logger.warning(f"Error loading model with auth: {str(e)}")

                # If unauthorized error and we were using a token, try without token
                if (
                    "401" in str(e)
                    or "unauthorized" in str(e).lower()
                    or "Invalid credentials" in str(e)
                ):
                    logger.info(
                        f"Authentication failed, trying without token for model: {model_name}"
                    )
                    # Remove token from kwargs
                    if "token" in model_kwargs:
                        del model_kwargs["token"]

                    # Try loading without token
                    tokenizer = AutoTokenizer.from_pretrained(model_name)

                    # Handle special case for TinyLlama which might need padding token
                    if "tinyllama" in model_name.lower():
                        if tokenizer.pad_token is None:
                            tokenizer.pad_token = tokenizer.eos_token

                    model = AutoModelForCausalLM.from_pretrained(
                        model_name, **model_kwargs
                    )
                else:
                    # For other errors, re-raise
                    raise

            # Make sure the model has padding token set properly
            if (
                hasattr(model.config, "pad_token_id")
                and model.config.pad_token_id is None
            ):
                if tokenizer.pad_token_id is not None:
                    model.config.pad_token_id = tokenizer.pad_token_id
                else:
                    model.config.pad_token_id = tokenizer.eos_token_id

            # Cache the model
            self.completion_cache[model_name] = (model, tokenizer)

            logger.info(f"Successfully loaded model {model_name}")
            return model, tokenizer

        except Exception as e:
            logger.error(
                f"Error in _load_causal_lm_model: {str(e)}\n{traceback.format_exc()}"
            )
            raise

    def clear_cache(self) -> None:
        """
        Clear the completion model cache to free up memory
        """
        logger.info(
            f"Clearing completion cache with {len(self.completion_cache)} models"
        )
        self.completion_cache = {}
