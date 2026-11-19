"""
OpenAI SDK runtime — dùng cho:

  Blue Team → OpenRouter liquid/lfm-3-1.7b:free (create_blue_pair)
  Red Team  → OpenAI gpt-4o-mini (create_openai_pair) khi RED_TEAM_PROVIDER=openai

Gemini Red Team dùng Google ADK trong agents/*.py — không đi qua file này.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from inspect import isawaitable
from typing import Any, Callable

from core.config import (
    PROVIDER_OPENROUTER,
    get_red_model,
    get_red_provider,
    get_blue_model,
    get_blue_provider,
    openrouter_client_kwargs,
    red_openai_client_kwargs,
)


@dataclass
class OpenAIAgent:
    name: str
    instruction: str
    provider: str = "openai"


@dataclass
class OpenAIInvocationContext:
    """Per-request user identity and decisions from the runtime plugins."""

    user_id: str = "student"
    blocked_by: str | None = None
    output_modified_by: str | None = None
    judge_checked: bool = False


@dataclass
class OpenAIRunner:
    """Optional ADK-style plugins + Chat Completions."""

    app_name: str
    model: str
    plugins: list = field(default_factory=list)
    provider: str = "openai"
    temperature: float | None = 0.4
    client_kwargs: dict = field(default_factory=dict)
    input_hooks: list[Callable[[str], str | None]] = field(default_factory=list)
    output_hooks: list[Callable[[str], str]] = field(default_factory=list)

    def _client(self):
        from openai import OpenAI

        return OpenAI(**(self.client_kwargs or {}))

    async def chat(
        self,
        agent: OpenAIAgent,
        user_message: str,
        *,
        invocation_context: OpenAIInvocationContext | None = None,
    ) -> str:
        ctx = invocation_context or OpenAIInvocationContext()
        for hook in self.input_hooks:
            blocked = hook(user_message)
            if blocked:
                ctx.blocked_by = "input_hook"
                return blocked

        block_msg = await self._run_input_plugins(user_message, ctx)
        if block_msg is not None:
            return block_msg

        client = self._client()
        sampling = {} if self.temperature is None else {"temperature": self.temperature}
        completion = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": agent.instruction},
                {"role": "user", "content": user_message},
            ],
            **sampling,
        )
        text = (completion.choices[0].message.content or "").strip()

        for hook in self.output_hooks:
            text = hook(text)

        text = await self._run_output_plugins(text, ctx)
        return text

    async def _run_input_plugins(self, user_message: str, ctx: OpenAIInvocationContext) -> str | None:
        if not self.plugins:
            return None
        try:
            from google.genai import types
        except ImportError:
            return None

        user_content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)],
        )
        for plugin in self.plugins:
            cb = getattr(plugin, "on_user_message_callback", None)
            if cb is None:
                continue
            result = cb(invocation_context=ctx, user_message=user_content)
            if isawaitable(result):
                result = await result
            if result is None:
                continue
            ctx.blocked_by = getattr(plugin, "name", type(plugin).__name__)
            return _content_to_text(result)
        return None

    async def _run_output_plugins(self, text: str, ctx: OpenAIInvocationContext) -> str:
        if not self.plugins or not text:
            return text
        try:
            from google.genai import types
        except ImportError:
            return text

        content = types.Content(
            role="model", parts=[types.Part.from_text(text=text)]
        )

        class _Resp:
            pass

        llm_response = _Resp()
        llm_response.content = content

        for plugin in self.plugins:
            cb = getattr(plugin, "after_model_callback", None)
            if cb is None:
                continue
            out = cb(callback_context=ctx, llm_response=llm_response)
            if isawaitable(out):
                out = await out
            if out is not None and getattr(out, "content", None) is not None:
                ctx.output_modified_by = getattr(plugin, "name", type(plugin).__name__)
                llm_response = out
        return _content_to_text(llm_response.content)


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = getattr(content, "parts", None) or []
    chunks = []
    for part in parts:
        t = getattr(part, "text", None)
        if t:
            chunks.append(t)
    return "".join(chunks)


def _make_pair(
    *,
    name: str,
    instruction: str,
    app_name: str,
    model: str,
    provider: str,
    client_kwargs: dict,
    plugins: list | None = None,
    input_hooks: list | None = None,
    output_hooks: list | None = None,
    temperature: float = 0.4,
) -> tuple[OpenAIAgent, OpenAIRunner]:
    agent = OpenAIAgent(name=name, instruction=instruction, provider=provider)
    runner = OpenAIRunner(
        app_name=app_name,
        model=model,
        provider=provider,
        client_kwargs=client_kwargs,
        plugins=list(plugins or []),
        input_hooks=list(input_hooks or []),
        output_hooks=list(output_hooks or []),
        temperature=temperature,
    )
    return agent, runner


def create_blue_pair(
    *,
    name: str,
    instruction: str,
    app_name: str,
    plugins: list | None = None,
    input_hooks: list | None = None,
    output_hooks: list | None = None,
    temperature: float = 0.4,
) -> tuple[OpenAIAgent, OpenAIRunner]:
    """Blue Team — always OpenRouter liquid/lfm-3-1.7b:free."""
    return _make_pair(
        name=name,
        instruction=instruction,
        app_name=app_name,
        model=get_blue_model(),
        provider=get_blue_provider(),
        client_kwargs=openrouter_client_kwargs(),
        plugins=plugins,
        input_hooks=input_hooks,
        output_hooks=output_hooks,
        temperature=temperature,
    )


def create_openai_pair(
    *,
    name: str,
    instruction: str,
    app_name: str,
    plugins: list | None = None,
    input_hooks: list | None = None,
    output_hooks: list | None = None,
    temperature: float = 0.4,
    model: str | None = None,
    provider: str | None = None,
) -> tuple[OpenAIAgent, OpenAIRunner]:
    """Build a Red Team runner for an OpenAI-compatible endpoint."""
    provider = provider or get_red_provider()
    client_kwargs = (
        openrouter_client_kwargs() if provider == PROVIDER_OPENROUTER else red_openai_client_kwargs()
    )
    return _make_pair(
        name=name,
        instruction=instruction,
        app_name=app_name,
        model=model or get_red_model(),
        provider=provider,
        client_kwargs=client_kwargs,
        plugins=plugins,
        input_hooks=input_hooks,
        output_hooks=output_hooks,
        temperature=temperature,
    )
