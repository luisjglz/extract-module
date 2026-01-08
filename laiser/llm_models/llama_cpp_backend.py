import os
import re
from pathlib import Path
from typing import Optional, List
from llama_cpp import Llama


def _strip_fences(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    # mimic gemini.py behavior
    text = re.sub(r"```(?:json)?|```", "", text).strip()
    return text


class LlamaCppBackend:
    """
    Wrapper around llama-cpp-python supporting:
    - text mode (for cleaning prompt)
    - json mode via create_chat_completion + response_format (for extraction)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: int = 4096,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = 0,
        temperature: float = 0.2,
        chat_format: str = "chatml",
    ):
        model_path = model_path or os.getenv("LAISER_LLAMA_CPP_MODEL_PATH")
        if not model_path:
            raise ValueError("Set LAISER_LLAMA_CPP_MODEL_PATH or pass model_path to LlamaCppBackend.")

        model_path = str(Path(model_path).expanduser().resolve())
        if not Path(model_path).exists():
            raise ValueError(f"Model path does not exist: {model_path}")

        self.temperature = temperature

        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads or None,
            n_gpu_layers=n_gpu_layers,
            logits_all=False,
            chat_format=chat_format,
        )

    def generate(
        self,
        prompt: str,
        *,
        json_mode: bool = False,
        system: str = "You are a helpful assistant.",
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        temperature: Optional[float] = None,
    ) -> str:
        temperature = self.temperature if temperature is None else temperature

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        # Always use chat completion
        kwargs = {
            "messages": messages,
            "temperature": temperature,
        }

        # Optional knobs (these are safe to omit if unsupported in your version)
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if stop is not None:
            kwargs["stop"] = stop

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self.llm.create_chat_completion(**kwargs)
        print("LLAMA_CPP RESPONSE:::::::",resp)
        return _strip_fences(resp["choices"][0]["message"]["content"])


# ✅ Gemini-style entry point used by the router
def llama_cpp_generate(prompt: str, llm: "LlamaCppBackend", *, json_mode: bool = False) -> str:
    if llm is None:
        raise ValueError("llm is None; expected an initialized LlamaCppBackend instance.")
    return llm.generate(
        prompt,
        json_mode=json_mode,
        system="You are a helpful assistant that outputs in JSON." if json_mode else "You are a helpful assistant.",
    )
