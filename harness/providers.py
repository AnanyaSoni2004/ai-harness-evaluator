"""Provider knowledge: which env var holds each provider's key, key-format detection, local endpoints."""
from __future__ import annotations

import os
from urllib.parse import urlparse

# LiteLLM provider -> environment variables that may hold its key (checked after AI_API_KEY).
PROVIDER_KEY_ENV: dict[str, tuple[str, ...]] = {
    "groq": ("GROQ_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
    "dashscope": ("DASHSCOPE_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
}
# Every variable that may hold a model key: redacted from logs and removed from target-repo processes.
KEY_ENV_VARS: tuple[str, ...] = ("AI_API_KEY",) + tuple(v for names in PROVIDER_KEY_ENV.values() for v in names)

# Providers that run on the evaluator's machine and need no key.
LOCAL_PROVIDERS = frozenset({"ollama", "ollama_chat", "lm_studio", "hosted_vllm", "llamafile"})
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "0.0.0.0", "::1", "host.docker.internal"})
LOCAL_PLACEHOLDER_KEY = "no-key-needed"  # sent to local servers, which ignore it

# Key prefix -> preset. Plain "sk-" keys are shared by OpenAI, DashScope and DeepSeek, so they are not guessed.
_KEY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("gsk_", "groq"),
    ("AIza", "gemini"),
    ("sk-or-", "openrouter"),
    ("sk-ant-", "anthropic"),  # a prefix only (example key shape), not a secret
    ("sk-proj-", "openai"),
    ("sk-svcacct-", "openai"),
)
# Provider-native variable -> preset, in the order auto-detection tries them.
_ENV_PRESETS: tuple[tuple[str, str], ...] = (
    ("GROQ_API_KEY", "groq"),
    ("GEMINI_API_KEY", "gemini"),
    ("GOOGLE_API_KEY", "gemini"),
    ("OPENAI_API_KEY", "openai"),
    ("DASHSCOPE_API_KEY", "qwen"),
    ("OPENROUTER_API_KEY", "openrouter"),
    ("DEEPSEEK_API_KEY", "deepseek"),
    ("ANTHROPIC_API_KEY", "anthropic"),
)
_KEY_KIND = {"groq": "Groq", "gemini": "Gemini", "openrouter": "OpenRouter", "anthropic": "Anthropic",
             "openai": "OpenAI"}


def env(name: str) -> str:
    """The stripped value of an environment variable ('' when unset)."""
    return os.environ.get(name, "").strip()


def provider_of(model: str) -> str:
    """LiteLLM's provider for a model string ('gemini/x' -> 'gemini'), or its prefix when LiteLLM does not know it."""
    import litellm

    litellm.suppress_debug_info = True  # no "Provider List" banner for an unknown provider
    try:
        return str(litellm.get_llm_provider(model)[1])
    except Exception:  # noqa: BLE001 - unknown provider strings are reported later, by the call itself
        return model.split("/", 1)[0] if "/" in model else ""


def is_local(provider: str, api_base: str | None) -> bool:
    """True for a model served on this machine (Ollama, LM Studio, vLLM, or any localhost endpoint)."""
    if provider in LOCAL_PROVIDERS:
        return True
    host = urlparse(api_base or "").hostname or ""
    return host in LOCAL_HOSTS


def preset_for_key(key: str) -> str | None:
    """The preset a key's format points to (gsk_ -> groq, AIza -> gemini, ...), or None when ambiguous."""
    for prefix, preset in _KEY_PREFIXES:
        if key.startswith(prefix):
            return preset
    return None


def preset_from_env() -> tuple[str, str] | None:
    """(preset, variable) for the first provider-native key variable that is set, or None."""
    for name, preset in _ENV_PRESETS:
        if env(name):
            return preset, name
    return None


def key_mismatch(key: str, provider: str) -> str | None:
    """A hint when the key's format belongs to a different provider than the configured model, else None."""
    preset = preset_for_key(key)
    if preset is None or preset == provider or (preset == "gemini" and provider == "vertex_ai"):
        return None
    return (f"AI_API_KEY looks like a {_KEY_KIND.get(preset, preset)} key, but the model is a {provider} model. "
            f"Try: make run MODEL={preset}")


def context_window_for(model: str) -> int | None:
    """The model's input context window from LiteLLM's model table, or None when unknown."""
    import litellm

    try:
        info = litellm.get_model_info(model)
    except Exception:  # noqa: BLE001 - unknown models (local, custom endpoints) fall back to a default
        return None
    value = info.get("max_input_tokens") or info.get("max_tokens")
    return int(value) if value else None


def redact_values() -> list[str]:
    """Current values of every key variable, longest first (so no key is replaced only partly)."""
    values = {env(n) for n in KEY_ENV_VARS}
    return sorted((v for v in values if v), key=len, reverse=True)

