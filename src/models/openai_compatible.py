"""OpenAI-compatible 多模态接口封装，统一处理 provider、图片载入和重试。"""

from __future__ import annotations

import base64
import mimetypes
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import requests


_ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
_DEFAULT_MAX_IMAGE_BYTES = 20 * 1024 * 1024


@dataclass(frozen=True)
class ProviderConfig:
    """一个 OpenAI-compatible provider 的地址、密钥环境变量和默认模型。"""

    name: str
    base_url: str
    api_key_env: str
    default_model: str
    model_type: Literal["closed", "open", "unknown"]
    base_url_env: str | None = None


PROVIDERS: dict[str, ProviderConfig] = {
    "qwen": ProviderConfig(
        name="qwen",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
        default_model="qwen-vl-plus",
        model_type="open",
    ),
    "gemini_local": ProviderConfig(
        name="gemini_local",
        base_url="http://127.0.0.1:8317/v1",
        api_key_env="GEMINI_LOCAL_API_KEY",
        default_model="gemini-2.5-flash",
        model_type="closed",
    ),
    "gpt54_local": ProviderConfig(
        name="gpt54_local",
        base_url="http://127.0.0.1:8317/v1",
        api_key_env="CHATGPT_LOCAL_KEY",
        default_model="gpt-5.4-mini",
        model_type="closed",
        base_url_env="GPT54_LOCAL_BASE_URL",
    ),
    "openrouter_qwen3_vl_instruct": ProviderConfig(
        name="openrouter_qwen3_vl_instruct",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        default_model="qwen/qwen3-vl-8b-instruct",
        model_type="open",
    ),
    "openrouter_qwen3_vl_thinking": ProviderConfig(
        name="openrouter_qwen3_vl_thinking",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        default_model="qwen/qwen3-vl-8b-thinking",
        model_type="open",
    ),
}


class OpenAICompatibleClient:
    def __init__(
        self,
        config: ProviderConfig,
        timeout: int = 120,
        max_retries: int = 2,
        retry_delay_seconds: float = 1.0,
    ) -> None:
        self.config = config
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds

    def chat_completion(
        self,
        *,
        prompt: str,
        image_path: str | Path,
        model: str | None = None,
        temperature: float = 0,
        max_tokens: int = 512,
        allowed_image_root: str | Path | None = None,
        max_image_bytes: int = _DEFAULT_MAX_IMAGE_BYTES,
        native_reasoning: bool = False,
        reasoning_max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """发送单次多模态 chat completion 请求。"""

        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing API key environment variable: {self.config.api_key_env}"
            )

        payload = build_chat_payload(
            model=model or self.config.default_model,
            prompt=prompt,
            image_path=image_path,
            temperature=temperature,
            max_tokens=max_tokens,
            allowed_image_root=allowed_image_root,
            max_image_bytes=max_image_bytes,
            native_reasoning=native_reasoning,
            reasoning_max_tokens=reasoning_max_tokens,
            provider_name=self.config.name,
            response_format=response_format,
        )
        base_url = (
            os.environ.get(self.config.base_url_env)
            if self.config.base_url_env is not None
            else None
        ) or self.config.base_url
        response = self._post_with_retries(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            payload=payload,
        )
        return response.json()

    def _post_with_retries(
        self, url: str, *, headers: dict[str, str], payload: dict[str, Any]
    ) -> requests.Response:
        """对超时、连接错误、429 和 5xx 做有限重试。"""

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise
                time.sleep(self.retry_delay_seconds)
                continue

            if response.status_code < 400:
                return response
            if response.status_code not in {429} and response.status_code < 500:
                raise RuntimeError(_response_error_message(response))
            if attempt >= self.max_retries:
                raise RuntimeError(_response_error_message(response))
            time.sleep(self.retry_delay_seconds)

        if last_error is not None:
            raise last_error
        raise RuntimeError("Model API request failed")


def _response_error_message(response: requests.Response) -> str:
    message = f"Model API request failed with status {response.status_code}"
    body = getattr(response, "text", "")
    if body:
        return f"{message}: {str(body)[:500]}"
    return message


def build_chat_payload(
    *,
    model: str,
    prompt: str,
    image_path: str | Path,
    temperature: float = 0,
    max_tokens: int = 512,
    allowed_image_root: str | Path | None = None,
    max_image_bytes: int = _DEFAULT_MAX_IMAGE_BYTES,
    native_reasoning: bool = False,
    reasoning_max_tokens: int | None = None,
    provider_name: str | None = None,
    response_format: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_to_data_url(
                                image_path,
                                allowed_root=allowed_image_root,
                                max_bytes=max_image_bytes,
                            )
                        },
                    },
                ],
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format is not None:
        payload["response_format"] = response_format
    if native_reasoning:
        _add_native_reasoning_options(payload, reasoning_max_tokens, provider_name)
    return payload


def _add_native_reasoning_options(
    payload: dict[str, Any],
    reasoning_max_tokens: int | None,
    provider_name: str | None,
) -> None:
    if provider_name == "gemini_local":
        payload["google"] = {"thinking_config": {"include_thoughts": True}}
        return
    if provider_name is None or not provider_name.startswith("openrouter_"):
        raise ValueError(
            f"Native reasoning is not supported for provider: {provider_name}"
        )

    payload["reasoning"] = {"enabled": True, "exclude": False}
    if reasoning_max_tokens is not None:
        payload["reasoning"]["max_tokens"] = reasoning_max_tokens


def extract_message_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if (
                isinstance(item, dict)
                and item.get("type") == "text"
                and item.get("thought") is not True
            ):
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return str(content)


def extract_native_reasoning(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices") or []
    if not choices:
        return {}
    message = choices[0].get("message", {})
    reasoning: dict[str, Any] = {}
    for key in ("reasoning", "reasoning_content", "reasoning_details"):
        value = message.get(key)
        if value not in (None, "", []):
            reasoning[key] = value
    thought_summary = _extract_gemini_thought_summary(message.get("content"))
    if thought_summary:
        reasoning["thought_summary"] = thought_summary
    return reasoning


def _extract_gemini_thought_summary(content: Any) -> str:
    if not isinstance(content, list):
        return ""
    parts = []
    for item in content:
        if isinstance(item, dict) and item.get("thought") is True:
            parts.append(str(item.get("text", "")))
    return "".join(parts).strip()


def get_provider_config(provider: str) -> ProviderConfig:
    try:
        return PROVIDERS[provider]
    except KeyError as exc:
        available = ", ".join(sorted(PROVIDERS))
        raise ValueError(
            f"Unknown provider: {provider}. Available providers: {available}"
        ) from exc


def image_to_data_url(
    image_path: str | Path,
    *,
    allowed_root: str | Path | None = None,
    max_bytes: int = _DEFAULT_MAX_IMAGE_BYTES,
) -> str:
    """校验图片路径和大小后转成 OpenAI-compatible image_url data URL。"""

    path = Path(image_path).resolve()
    if allowed_root is not None:
        root = Path(allowed_root).resolve()
        if path != root and root not in path.parents:
            raise ValueError(f"Image path is outside allowed root: {path.name}")
    if path.suffix.lower() not in _ALLOWED_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {path.suffix}")
    if not path.is_file():
        raise ValueError(f"Image path is not a regular file: {path.name}")
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"Image file is too large: {size} bytes")
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"
