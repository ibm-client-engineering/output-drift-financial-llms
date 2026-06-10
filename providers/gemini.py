#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Google Gemini provider for the drift runner experiment.

Supports Gemini 3.0, 2.5, and 2.0 models via the Generative AI API.
Tests generational drift across model versions.
"""
import os
import logging
from typing import Dict, List, Optional, Any
import httpx

from providers import retry_on_transient, async_retry_on_transient


class GeminiProvider:
    """Google Gemini provider using the Generative Language API."""

    # Model aliases for convenience
    MODEL_ALIASES = {
        # Gemini 3 (Nov 2025) - Bleeding Edge / Challenger
        "gemini-3": "gemini-3-pro-preview",
        "gemini-3-pro": "gemini-3-pro-preview",
        # Gemini 2.5 (Mid 2025) - Champion / Current Standard
        "gemini-2.5": "gemini-2.5-pro",
        "gemini-2.5-pro": "gemini-2.5-pro",
        "gemini-2.5-flash": "gemini-2.5-flash",
        # Gemini 2.0 (Early 2025) - Historical Ancestor
        "gemini-2.0": "gemini-2.0-flash",
        "gemini-2.0-flash": "gemini-2.0-flash",
        # Gemma 3 (Nov 2025) - Open source GCP model
        "gemma-3": "gemma-3-27b-it",
        "gemma-3-27b": "gemma-3-27b-it",
        # Default to latest stable
        "gemini": "gemini-2.5-pro",
        "gemini-pro": "gemini-2.5-pro",
        "gemini-flash": "gemini-2.5-flash",
    }

    # Available models for generational drift testing
    AVAILABLE_MODELS = [
        # Gemini 3 (Nov 2025) - Bleeding Edge Challenger
        "gemini-3-pro-preview",
        # Gemini 2.5 (Mid 2025) - Production Champion
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        # Gemini 2.0 (Early 2025) - Historical Ancestor
        "gemini-2.0-flash",
        # Gemma 3 (Nov 2025) - Open Source GCP (27B params)
        "gemma-3-27b-it",
    ]

    def __init__(self, api_key: Optional[str] = None, logger=None):
        self.name = "gemini"
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.logger = logger or logging.getLogger(__name__)
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

        if not self.api_key:
            raise ValueError("Missing GEMINI_API_KEY environment variable")

        self.logger.info("Gemini provider initialized")

    def _normalize_model_id(self, model: str) -> str:
        """Normalize model ID using aliases."""
        return self.MODEL_ALIASES.get(model, model)

    def supports_listing(self) -> bool:
        """Returns True - we maintain a list of available models."""
        return True

    def list_models(self) -> List[str]:
        """Return list of available Gemini models."""
        return self.AVAILABLE_MODELS.copy()

    @retry_on_transient()
    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: Optional[int] = None,  # Gemini supports seed for some models
        max_new_tokens: int = 4096,
        stream: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Synchronous generation using Gemini Generative Language API.
        """
        model_id = self._normalize_model_id(model)

        # Build the full model path
        model_path = f"models/{model_id}"

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "topP": top_p,
                "maxOutputTokens": max_new_tokens,
            },
        }

        # Add seed if provided (Gemini 2.0+ supports this)
        if seed is not None:
            payload["generationConfig"]["seed"] = seed

        # Add any extra parameters
        if extra:
            payload["generationConfig"].update(extra)

        url = f"{self.base_url}/{model_path}:generateContent?key={self.api_key}"

        try:
            with httpx.Client(timeout=180) as client:
                response = client.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                data = response.json()

                # Extract text from response
                text = ""
                if "candidates" in data and len(data["candidates"]) > 0:
                    candidate = data["candidates"][0]
                    if "content" in candidate and "parts" in candidate["content"]:
                        for part in candidate["content"]["parts"]:
                            if "text" in part:
                                text += part["text"]

                # Extract usage info
                usage = {}
                if "usageMetadata" in data:
                    usage = {
                        "prompt_tokens": data["usageMetadata"].get("promptTokenCount", 0),
                        "completion_tokens": data["usageMetadata"].get("candidatesTokenCount", 0),
                        "total_tokens": data["usageMetadata"].get("totalTokenCount", 0),
                    }

                return {
                    "text": text,
                    "model": model_id,
                    "usage": usage,
                    "stop_reason": data.get("candidates", [{}])[0].get("finishReason", ""),
                }

        except httpx.HTTPStatusError as e:
            self.logger.error(f"Gemini API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            self.logger.error(f"Gemini generation failed: {e}")
            raise

    @async_retry_on_transient()
    async def agenerate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: Optional[int] = None,
        max_new_tokens: int = 4096,
        stream: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Async version of generate."""
        model_id = self._normalize_model_id(model)
        model_path = f"models/{model_id}"

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "topP": top_p,
                "maxOutputTokens": max_new_tokens,
            },
        }

        if seed is not None:
            payload["generationConfig"]["seed"] = seed

        if extra:
            payload["generationConfig"].update(extra)

        url = f"{self.base_url}/{model_path}:generateContent?key={self.api_key}"

        try:
            async with httpx.AsyncClient(timeout=180) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                data = response.json()

                text = ""
                if "candidates" in data and len(data["candidates"]) > 0:
                    candidate = data["candidates"][0]
                    if "content" in candidate and "parts" in candidate["content"]:
                        for part in candidate["content"]["parts"]:
                            if "text" in part:
                                text += part["text"]

                usage = {}
                if "usageMetadata" in data:
                    usage = {
                        "prompt_tokens": data["usageMetadata"].get("promptTokenCount", 0),
                        "completion_tokens": data["usageMetadata"].get("candidatesTokenCount", 0),
                        "total_tokens": data["usageMetadata"].get("totalTokenCount", 0),
                    }

                return {
                    "text": text,
                    "model": model_id,
                    "usage": usage,
                    "stop_reason": data.get("candidates", [{}])[0].get("finishReason", ""),
                }

        except httpx.HTTPStatusError as e:
            self.logger.error(f"Gemini API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            self.logger.error(f"Gemini async generation failed: {e}")
            raise
