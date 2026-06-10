#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anthropic/Claude provider for the drift runner experiment.

Supports Claude Sonnet, Haiku, and Opus models via the Anthropic API.
"""
import os
import logging
from typing import Dict, List, Optional, Any
import httpx


class AnthropicProvider:
    """Anthropic Claude provider using the Messages API."""

    # Model aliases for convenience
    MODEL_ALIASES = {
        # Claude 4.5 aliases (current)
        "claude-opus-4.5": "claude-opus-4-5-20251101",
        "claude-sonnet-4.5": "claude-sonnet-4-5-20250929",
        "claude-haiku-4.5": "claude-haiku-4-5-20251001",
        "opus-4.5": "claude-opus-4-5-20251101",
        "sonnet-4.5": "claude-sonnet-4-5-20250929",
        "haiku-4.5": "claude-haiku-4-5-20251001",
        # Claude 4 aliases
        "claude-sonnet-4": "claude-sonnet-4-20250514",
        "claude-opus-4": "claude-opus-4-20250514",
        "sonnet-4": "claude-sonnet-4-20250514",
        "opus-4": "claude-opus-4-20250514",
        # Claude 3.5 aliases
        "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
        "claude-3.5-haiku": "claude-3-5-haiku-20241022",
        "sonnet": "claude-sonnet-4-5-20250929",  # default to latest
        "haiku": "claude-haiku-4-5-20251001",
    }

    # Available models for listing
    AVAILABLE_MODELS = [
        # Claude 4.5 (Nov 2025) - Latest
        "claude-opus-4-5-20251101",
        "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5-20251001",
        # Claude 4.0 (May 2025) - For generational comparison
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
        # Claude 3.5 (Oct 2024) - Historical baseline
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
    ]

    def __init__(self, api_key: Optional[str] = None, logger=None):
        self.name = "anthropic"
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.logger = logger or logging.getLogger(__name__)
        self.base_url = "https://api.anthropic.com/v1"

        if not self.api_key:
            raise ValueError("Missing ANTHROPIC_API_KEY environment variable")

        self.logger.info("Anthropic provider initialized")

    def _normalize_model_id(self, model: str) -> str:
        """Normalize model ID using aliases."""
        return self.MODEL_ALIASES.get(model, model)

    def supports_listing(self) -> bool:
        """Returns True - we maintain a list of available models."""
        return True

    def list_models(self) -> List[str]:
        """Return list of available Claude models."""
        return self.AVAILABLE_MODELS.copy()

    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: Optional[int] = None,  # Not supported by Anthropic
        max_new_tokens: int = 4096,
        stream: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Synchronous generation using Anthropic Messages API.

        Note: Anthropic does not support seed parameter for reproducibility.
        """
        model_id = self._normalize_model_id(model)

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        payload = {
            "model": model_id,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

        # Add top_p if not default
        if top_p < 1.0:
            payload["top_p"] = top_p

        # Add any extra parameters
        if extra:
            payload.update(extra)

        try:
            with httpx.Client(timeout=180) as client:
                response = client.post(
                    f"{self.base_url}/messages",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

                # Extract text from response
                text = ""
                if "content" in data and len(data["content"]) > 0:
                    for block in data["content"]:
                        if block.get("type") == "text":
                            text += block.get("text", "")

                return {
                    "text": text,
                    "model": data.get("model", model_id),
                    "usage": data.get("usage", {}),
                    "stop_reason": data.get("stop_reason"),
                }

        except httpx.HTTPStatusError as e:
            self.logger.error(f"Anthropic API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            self.logger.error(f"Anthropic generation failed: {e}")
            raise

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

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        payload = {
            "model": model_id,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

        if top_p < 1.0:
            payload["top_p"] = top_p

        if extra:
            payload.update(extra)

        try:
            async with httpx.AsyncClient(timeout=180) as client:
                response = await client.post(
                    f"{self.base_url}/messages",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

                text = ""
                if "content" in data and len(data["content"]) > 0:
                    for block in data["content"]:
                        if block.get("type") == "text":
                            text += block.get("text", "")

                return {
                    "text": text,
                    "model": data.get("model", model_id),
                    "usage": data.get("usage", {}),
                    "stop_reason": data.get("stop_reason"),
                }

        except httpx.HTTPStatusError as e:
            self.logger.error(f"Anthropic API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            self.logger.error(f"Anthropic async generation failed: {e}")
            raise
