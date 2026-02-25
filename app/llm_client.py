"""LLM Client — Groq primary, OpenRouter fallback, with streaming support."""

from __future__ import annotations

import json
from typing import AsyncGenerator

import httpx
from groq import Groq, AsyncGroq

from app.utils import get_settings, setup_logger

logger = setup_logger(__name__)


class LLMClient:
    """Dual-provider LLM client with automatic fallback and streaming."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._groq: Groq | None = None
        self._async_groq: AsyncGroq | None = None
        self._http: httpx.AsyncClient | None = None

    # ─── Lazy Initialization ─────────────────────────────

    @property
    def groq(self) -> Groq:
        if self._groq is None:
            self._groq = Groq(api_key=self.settings.GROQ_API_KEY)
        return self._groq

    @property
    def async_groq(self) -> AsyncGroq:
        if self._async_groq is None:
            self._async_groq = AsyncGroq(api_key=self.settings.GROQ_API_KEY)
        return self._async_groq

    @property
    def http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=60.0)
        return self._http

    # ═══════════════════════════════════════════════════════
    # Synchronous (used during ingestion)
    # ═══════════════════════════════════════════════════════

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Synchronous generation with Groq → OpenRouter fallback."""
        try:
            return self._gen_groq_sync(prompt, system_prompt)
        except Exception as exc:
            logger.warning("Groq sync failed (%s). Falling back to OpenRouter.", exc)
            return self._gen_openrouter_sync(prompt, system_prompt)

    def _gen_groq_sync(self, prompt: str, system_prompt: str) -> str:
        logger.info("🔗 LLM API: Groq (sync) | model: %s", self.settings.GROQ_MODEL)
        msgs = self._build_messages(system_prompt, prompt)
        resp = self.groq.chat.completions.create(
            model=self.settings.GROQ_MODEL,
            messages=msgs,
            temperature=0.3,
            max_tokens=2048,
        )
        return resp.choices[0].message.content or ""

    def _gen_openrouter_sync(self, prompt: str, system_prompt: str) -> str:
        logger.info("🔗 LLM API: OpenRouter (sync) | model: %s", self.settings.OPENROUTER_MODEL)
        msgs = self._build_messages(system_prompt, prompt)
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.settings.OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.settings.OPENROUTER_MODEL,
                    "messages": msgs,
                    "temperature": 0.3,
                    "max_tokens": 2048,
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    # ═══════════════════════════════════════════════════════
    # Async (used during API requests)
    # ═══════════════════════════════════════════════════════

    async def agenerate(self, prompt: str, system_prompt: str = "") -> str:
        """Async generation with fallback."""
        try:
            return await self._agen_groq(prompt, system_prompt)
        except Exception as exc:
            logger.warning("Groq async failed (%s). Falling back.", exc)
            return await self._agen_openrouter(prompt, system_prompt)

    async def _agen_groq(self, prompt: str, system_prompt: str) -> str:
        logger.info("🔗 LLM API: Groq (async) | model: %s", self.settings.GROQ_MODEL)
        msgs = self._build_messages(system_prompt, prompt)
        resp = await self.async_groq.chat.completions.create(
            model=self.settings.GROQ_MODEL,
            messages=msgs,
            temperature=0.3,
            max_tokens=2048,
        )
        return resp.choices[0].message.content or ""

    async def _agen_openrouter(self, prompt: str, system_prompt: str) -> str:
        logger.info("🔗 LLM API: OpenRouter (async) | model: %s", self.settings.OPENROUTER_MODEL)
        msgs = self._build_messages(system_prompt, prompt)
        resp = await self.http.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.settings.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.settings.OPENROUTER_MODEL,
                "messages": msgs,
                "temperature": 0.3,
                "max_tokens": 2048,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    # ═══════════════════════════════════════════════════════
    # Streaming (Gemini-like typing effect)
    # ═══════════════════════════════════════════════════════

    async def stream_generate(
        self, prompt: str, system_prompt: str = ""
    ) -> AsyncGenerator[str, None]:
        """Stream tokens with Groq → OpenRouter fallback."""
        try:
            async for token in self._stream_groq(prompt, system_prompt):
                yield token
        except Exception as exc:
            logger.warning("Groq stream failed (%s). Falling back.", exc)
            async for token in self._stream_openrouter(prompt, system_prompt):
                yield token

    async def _stream_groq(
        self, prompt: str, system_prompt: str
    ) -> AsyncGenerator[str, None]:
        logger.info("🔗 LLM API: Groq (stream) | model: %s", self.settings.GROQ_MODEL)
        msgs = self._build_messages(system_prompt, prompt)
        stream = await self.async_groq.chat.completions.create(
            model=self.settings.GROQ_MODEL,
            messages=msgs,
            temperature=0.3,
            max_tokens=2048,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    async def _stream_openrouter(
        self, prompt: str, system_prompt: str
    ) -> AsyncGenerator[str, None]:
        logger.info("🔗 LLM API: OpenRouter (stream) | model: %s", self.settings.OPENROUTER_MODEL)
        msgs = self._build_messages(system_prompt, prompt)
        async with self.http.stream(
            "POST",
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.settings.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.settings.OPENROUTER_MODEL,
                "messages": msgs,
                "temperature": 0.3,
                "max_tokens": 2048,
                "stream": True,
            },
        ) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    content = json.loads(payload)["choices"][0]["delta"].get(
                        "content", ""
                    )
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    # ─── Helpers ────────────────────────────────────────

    @staticmethod
    def _build_messages(system_prompt: str, user_prompt: str) -> list[dict]:
        msgs: list[dict] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": user_prompt})
        return msgs

    async def close(self) -> None:
        """Release async resources."""
        if self._http:
            await self._http.aclose()
