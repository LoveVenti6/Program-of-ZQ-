"""Claude (Anthropic) 最小调用封装。

用法：
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 claude_client.py "你好"

可选环境变量：
    ANTHROPIC_BASE_URL  中转/代理地址（默认官方 https://api.anthropic.com）
    CLAUDE_MODEL        默认模型，默认 claude-sonnet-4-5
"""

import os

from anthropic import Anthropic

DEFAULT_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5")


def get_client() -> Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("请先设置环境变量 ANTHROPIC_API_KEY")
    base_url = os.getenv("ANTHROPIC_BASE_URL") or None
    return Anthropic(api_key=api_key, base_url=base_url)


def chat(
    prompt: str,
    system: str | None = None,
    history: list[dict] | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
) -> str:
    """单轮调用，返回纯文本。history 为之前的多轮消息列表。"""
    messages = [*(history or []), {"role": "user", "content": prompt}]
    resp = get_client().messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
        **({"system": system} if system else {}),
    )
    return "".join(block.text for block in resp.content if block.type == "text")


def stream_chat(
    prompt: str,
    system: str | None = None,
    history: list[dict] | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
):
    """流式调用，逐段 yield 文本。"""
    messages = [*(history or []), {"role": "user", "content": prompt}]
    with get_client().messages.stream(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
        **({"system": system} if system else {}),
    ) as stream:
        for text in stream.text_stream:
            yield text


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "用一句话介绍你自己"
    for chunk in stream_chat(question, system="你是一个简洁的中文助手。"):
        print(chunk, end="", flush=True)
    print()
