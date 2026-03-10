"""Unified LLM interface.

All LLM calls go through this module.  To adapt to a different provider,
edit only this file.

Public API (only 2 functions):
    generate_stream()  — 异步流式，用于报告生成 / 讨论 / 润色（core/llm_service.py 调用）
    generate_sync()    — 同步调用，用于 PDF 图表识别 / 图表质量审核（processor/ & tools/ 调用）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator

from openai import AsyncOpenAI, AsyncAzureOpenAI, OpenAI, AzureOpenAI

from config import (
    LLM_API_BASE, LLM_API_KEY, LLM_API_VERSION,
    LLM_MODEL, LLM_TEMPERATURE, LLM_TIMEOUT,
    VISION_TIMEOUT,
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ToolCallDelta:
    index: int
    id: str | None = None
    name: str | None = None
    arguments: str | None = None


@dataclass
class StreamChunk:
    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[ToolCallDelta] | None = None
    finish_reason: str | None = None


# ---------------------------------------------------------------------------
# Client singletons
# ---------------------------------------------------------------------------

_async_client: AsyncOpenAI | None = None
_sync_client: OpenAI | None = None


def _get_async_client() -> AsyncOpenAI:
    """被 generate_stream() 使用。"""
    global _async_client
    if _async_client is None:
        if LLM_API_VERSION:
            _async_client = AsyncAzureOpenAI(
                azure_endpoint=LLM_API_BASE,
                api_version=LLM_API_VERSION,
                api_key=LLM_API_KEY,
                timeout=LLM_TIMEOUT,
            )
        else:
            _async_client = AsyncOpenAI(
                base_url=LLM_API_BASE or "https://api.deepseek.com",
                api_key=LLM_API_KEY,
                timeout=LLM_TIMEOUT,
            )
    return _async_client


def _get_sync_client(timeout: int = LLM_TIMEOUT) -> OpenAI:
    """被 generate_sync() 使用。"""
    global _sync_client
    if _sync_client is None:
        if LLM_API_VERSION:
            _sync_client = AzureOpenAI(
                azure_endpoint=LLM_API_BASE,
                api_version=LLM_API_VERSION,
                api_key=LLM_API_KEY,
                timeout=timeout,
            )
        else:
            _sync_client = OpenAI(
                base_url=LLM_API_BASE or "https://api.deepseek.com",
                api_key=LLM_API_KEY,
                timeout=timeout,
            )
    return _sync_client


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------

def _build_messages(
    text: str,
    *,
    system: str | None = None,
    images: list[str] | None = None,
) -> list[dict]:
    """Build an OpenAI-format messages list."""
    msgs: list[dict] = []
    if system:
        msgs.append({"role": "system", "content": system})

    if images:
        content: list[dict] = []
        for b64 in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
        content.append({"type": "text", "text": text})
        msgs.append({"role": "user", "content": content})
    else:
        msgs.append({"role": "user", "content": text})

    return msgs


# ---------------------------------------------------------------------------
# generate_stream
# ---------------------------------------------------------------------------
# 调用方: core/llm_service.py
# 场景:
#   - 报告生成（tool-calling loop，带 tools 参数）
#   - 讨论 Q&A（Reader/Writer Agent 多轮对话）
#   - 报告润色（Writer Agent 根据讨论内容改写报告）
# 输入: OpenAI 格式 messages list + 可选 tools/model/temperature 等 kwargs
# 输出: 逐块 yield StreamChunk（content / reasoning / tool_calls / finish_reason）
# ---------------------------------------------------------------------------

async def generate_stream(
    messages: list[dict],
    **kwargs,
) -> AsyncIterator[StreamChunk]:
    client = _get_async_client()
    model = kwargs.pop("model", LLM_MODEL)
    temperature = kwargs.pop("temperature", LLM_TEMPERATURE)
    max_tokens = kwargs.pop("max_tokens", 32784)
    tools = kwargs.pop("tools", None)
    timeout = kwargs.pop("timeout", None)

    create_kwargs = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
        **kwargs,
    )
    if tools is not None:
        create_kwargs["tools"] = tools
    if timeout is not None:
        create_kwargs["timeout"] = timeout

    stream = await client.chat.completions.create(**create_kwargs)

    async for chunk in stream:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        delta = choice.delta

        sc = StreamChunk()

        if choice.finish_reason:
            sc.finish_reason = choice.finish_reason

        # Reasoning content (DeepSeek R1)
        rc = getattr(delta, "reasoning_content", None)
        if rc:
            sc.reasoning = rc

        # Text content
        if delta.content:
            sc.content = delta.content

        # Tool calls
        if delta.tool_calls:
            sc.tool_calls = []
            for tc in delta.tool_calls:
                tcd = ToolCallDelta(index=tc.index)
                if tc.id:
                    tcd.id = tc.id
                if tc.function:
                    if tc.function.name:
                        tcd.name = tc.function.name
                    if tc.function.arguments:
                        tcd.arguments = tc.function.arguments
                sc.tool_calls.append(tcd)

        yield sc


# ---------------------------------------------------------------------------
# generate_sync
# ---------------------------------------------------------------------------
# 调用方:
#   - processor/vision_extractor.py — PDF 图表识别（传入 messages，含多张页面图片）
#   - tools/figure_reviewer.py     — 生成图表质量审核（传入 text + images）
# 输入: 两种模式
#   1) text + images: 自动构建 messages（figure_reviewer 使用）
#   2) messages: 预构建的 messages list（vision_extractor 使用）
# 输出: str，模型的完整回复文本
# ---------------------------------------------------------------------------

def generate_sync(
    text: str | None = None,
    *,
    messages: list[dict] | None = None,
    images: list[str] | None = None,
    system: str | None = None,
    **kwargs,
) -> str:
    model = kwargs.pop("model", LLM_MODEL)
    temperature = kwargs.pop("temperature", LLM_TEMPERATURE)
    max_tokens = kwargs.pop("max_tokens", 32784)
    timeout = kwargs.pop("timeout", None)

    if messages is None:
        if text is None:
            raise ValueError("Either text or messages must be provided")
        messages = _build_messages(text, system=system, images=images)

    client = _get_sync_client(timeout=timeout or VISION_TIMEOUT)
    create_kwargs = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )

    response = client.chat.completions.create(**create_kwargs)
    return response.choices[0].message.content or ""
