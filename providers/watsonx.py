#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WatsonX.AI provider for the drift runner experiment.

Supports both SDK and REST fallback for maximum compatibility.
"""
import os
import time
import logging
from typing import Dict, List, Optional, Any
import httpx


class WatsonxProvider:
    """WatsonX.AI provider with SDK-first, REST fallback strategy."""

    # Model aliases for common name variations
    MODEL_ALIASES = {
        # Llama aliases
        "meta-llama/llama-3-70b-instruct": "meta-llama/llama-3-3-70b-instruct",
        "llama-3-70b-instruct": "meta-llama/llama-3-3-70b-instruct",
        "meta-llama/llama-3.1-70b-instruct": "meta-llama/llama-3-3-70b-instruct",

        # Granite shorthand
        "granite-3-8b-instruct": "ibm/granite-3-8b-instruct",
        "granite-3-2-8b-instruct": "ibm/granite-3-2-8b-instruct",

        # Mistral shorthand
        "mistral-large": "mistralai/mistral-large",
    }

    def __init__(self, api_key: Optional[str] = None, url: Optional[str] = None,
                 project_id: Optional[str] = None, logger = None):
        self.api_key = api_key or os.getenv("WATSONX_API_KEY")
        self.url = url or os.getenv("WATSONX_URL")
        self.project_id = project_id or os.getenv("WATSONX_PROJECT_ID")
        self.logger = logger or logging.getLogger(__name__)

        if not all([self.api_key, self.url, self.project_id]):
            missing = []
            if not self.api_key: missing.append("WATSONX_API_KEY")
            if not self.url: missing.append("WATSONX_URL")
            if not self.project_id: missing.append("WATSONX_PROJECT_ID")
            raise ValueError(f"Missing required watsonx environment variables: {', '.join(missing)}")

        # Clean URL - safe to access since we checked above
        self.url = self.url.rstrip('/') if self.url else ""

        # Try SDK import
        self._use_sdk = False
        self._fm_class = None
        self._credentials = None
        try:
            from ibm_watsonx_ai import Credentials
            from ibm_watsonx_ai.foundation_models import ModelInference
            self._fm_class = ModelInference
            self._credentials = Credentials(api_key=self.api_key, url=self.url)
            self._use_sdk = True
            self.logger.info("WatsonX: Using SDK path")
        except ImportError:
            self.logger.info("WatsonX: SDK not available, using REST fallback")

        # For REST fallback, get IAM token
        self._iam_token = None
        self._token_expires = 0
        if not self._use_sdk:
            self._refresh_iam_token()

    def _normalize_model_id(self, model: str) -> str:
        """Normalize model ID using aliases."""
        return self.MODEL_ALIASES.get(model, model)

    def supports_listing(self) -> bool:
        """Returns True if model listing is supported."""
        return self._use_sdk

    def list_models(self) -> List[str]:
        """List available foundation models."""
        if not self._use_sdk:
            return []  # REST fallback doesn't implement model listing

        try:
            from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
            # Return real tenant models based on the supported list
            return [
                "google/flan-t5-xl",
                "ibm/granite-13b-instruct-v2",
                "ibm/granite-3-2-8b-instruct",
                "ibm/granite-3-2b-instruct",
                "ibm/granite-3-3-8b-instruct",
                "ibm/granite-3-8b-instruct",
                "ibm/granite-4-small-instruct-preview-rc",
                "ibm/granite-8b-code-instruct",
                "ibm/granite-guardian-3-2b",
                "ibm/granite-guardian-3-8b",
                "ibm/granite-vision-3-2-2b",
                "meta-llama/llama-2-13b-chat",
                "meta-llama/llama-3-2-11b-vision-instruct",
                "meta-llama/llama-3-2-90b-vision-instruct",
                "meta-llama/llama-3-3-70b-instruct",
                "meta-llama/llama-3-405b-instruct",
                "meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
                "meta-llama/llama-guard-3-11b-vision",
                "mistralai/mistral-large",
                "mistralai/mistral-medium-2505",
                "mistralai/mistral-small-3-1-24b-instruct-2503",
                "mistralai/pixtral-12b"
            ]
        except Exception as e:
            self.logger.warning(f"Could not list models: {e}")
            return []

    def _refresh_iam_token(self):
        """Get IAM token for REST API calls."""
        if time.time() < self._token_expires - 300:  # 5min buffer
            return

        data = {
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": self.api_key
        }

        try:
            response = httpx.post(
                "https://iam.cloud.ibm.com/identity/token",
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30
            )
            response.raise_for_status()
            token_data = response.json()
            self._iam_token = token_data["access_token"]
            self._token_expires = time.time() + token_data["expires_in"]
        except Exception as e:
            raise RuntimeError(f"Failed to get IAM token: {e}")

    def generate(self, *, model: str, prompt: str, temperature: float, top_p: float,
                 seed: Optional[int] = None, max_new_tokens: Optional[int] = None,
                 stream: bool = False, extra: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate text using watsonx.ai model.

        Returns:
            {
                "text": str,
                "model": str,
                "tokens_prompt": int|None,
                "tokens_completion": int|None,
                "latency_s": float,
                "cost_per_1k_tok": float|None,
                "raw": dict
            }
        """
        start_time = time.time()

        try:
            if self._use_sdk:
                result = self._generate_sdk(model, prompt, temperature, top_p, seed, max_new_tokens, stream, extra)
            else:
                result = self._generate_rest(model, prompt, temperature, top_p, seed, max_new_tokens, stream, extra)
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            result = {
                "text": "",
                "model": model,
                "tokens_prompt": None,
                "tokens_completion": None,
                "latency_s": time.time() - start_time,
                "cost_per_1k_tok": None,
                "raw": {"error": str(e)}
            }

        result["latency_s"] = time.time() - start_time
        return result

    def _generate_sdk(self, model: str, prompt: str, temperature: float, top_p: float,
                      seed: Optional[int], max_new_tokens: Optional[int],
                      stream: bool, extra: Optional[Dict]) -> Dict[str, Any]:
        """Generate using the watsonx SDK."""
        model = self._normalize_model_id(model)
        params = {
            "temperature": temperature,
            "top_p": top_p,
            "decoding_method": "greedy" if temperature == 0.0 else "sample",
            "return_options": {"input_tokens": True, "generated_tokens": True}
        }

        if seed is not None:
            params["random_seed"] = seed
        if max_new_tokens is not None:
            params["max_new_tokens"] = max_new_tokens

        if extra:
            params.update(extra)

        if not self._credentials:
            raise RuntimeError("Watson SDK credentials not initialized")

        fm = self._fm_class(
            model_id=model,
            credentials=self._credentials,
            project_id=self.project_id,
            params=params
        )

        if stream:
            # For streaming, collect all chunks
            text_parts = []
            raw_response = {}
            try:
                for chunk in fm.generate_text_stream(prompt):
                    if isinstance(chunk, dict):
                        text_parts.append(chunk.get('generated_text', ''))
                        raw_response = chunk
                    else:
                        text_parts.append(str(chunk))
                text = ''.join(text_parts)
            except Exception as e:
                text = ""
                raw_response = {"error": str(e)}
        else:
            response = fm.generate_text(prompt)
            if isinstance(response, dict):
                text = response.get('generated_text', '')
                raw_response = response
            else:
                text = str(response)
                raw_response = {"generated_text": text}

        # Extract token counts if available
        tokens_prompt = None
        tokens_completion = None
        if isinstance(raw_response, dict):
            if 'usage' in raw_response and isinstance(raw_response['usage'], dict):
                usage = raw_response['usage']
                tokens_prompt = usage.get('prompt_tokens')
                tokens_completion = usage.get('completion_tokens')
            elif 'token_usage' in raw_response and isinstance(raw_response['token_usage'], dict):
                usage = raw_response['token_usage']
                tokens_prompt = usage.get('prompt_token_count')
                tokens_completion = usage.get('generated_token_count')

        return {
            "text": text,
            "model": model,
            "tokens_prompt": tokens_prompt,
            "tokens_completion": tokens_completion,
            "cost_per_1k_tok": None,  # Not available in SDK response
            "raw": raw_response if isinstance(raw_response, dict) else {"generated_text": text}
        }

    def _generate_rest(self, model: str, prompt: str, temperature: float, top_p: float,
                       seed: Optional[int], max_new_tokens: Optional[int],
                       stream: bool, extra: Optional[Dict]) -> Dict[str, Any]:
        """Generate using REST API fallback."""
        model = self._normalize_model_id(model)
        self._refresh_iam_token()

        headers = {
            "Authorization": f"Bearer {self._iam_token}",
            "Content-Type": "application/json"
        }

        parameters = {
            "temperature": temperature,
            "top_p": top_p,
            "decoding_method": "greedy" if temperature == 0.0 else "sample",
            "return_options": {"input_tokens": True, "generated_tokens": True}
        }

        if seed is not None:
            parameters["random_seed"] = seed
        if max_new_tokens is not None:
            parameters["max_new_tokens"] = max_new_tokens

        if extra:
            parameters.update(extra)

        payload = {
            "model_id": model,
            "input": prompt,
            "parameters": parameters,
            "project_id": self.project_id
        }

        url = f"{self.url}/ml/v1/text/generation"
        params = {"version": "2023-05-29"}

        try:
            response = httpx.post(url, headers=headers, json=payload, params=params, timeout=180)
            response.raise_for_status()
            data = response.json()

            text = ""
            tokens_prompt = None
            tokens_completion = None

            if "results" in data and data["results"]:
                result = data["results"][0]
                text = result.get("generated_text", "")

                # Extract token usage
                if "token_usage" in result:
                    usage = result["token_usage"]
                    tokens_prompt = usage.get("prompt_token_count")
                    tokens_completion = usage.get("generated_token_count")

            return {
                "text": text,
                "model": model,
                "tokens_prompt": tokens_prompt,
                "tokens_completion": tokens_completion,
                "cost_per_1k_tok": None,  # Not available
                "raw": data
            }

        except Exception as e:
            raise RuntimeError(f"REST API call failed: {e}")


# For backward compatibility with existing runner.py
def create_watsonx_provider(logger=None) -> WatsonxProvider:
    """Factory function to create a WatsonX provider."""
    return WatsonxProvider(logger=logger)