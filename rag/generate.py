"""LLM generation via Hugging Face transformers (Qwen 2.5 in 4-bit NF4)."""

from __future__ import annotations

import logging
from typing import Protocol

from .config import settings

log = logging.getLogger(__name__)


class LLM(Protocol):
    name: str

    def complete(self, system: str, messages: list[dict]) -> str: ...


# --- Real LLM (lazy) --------------------------------------------------------


class TransformersLLM:
    """Qwen2.5 via Hugging Face transformers. 4-bit NF4 by default; BF16 fallback."""

    def __init__(self) -> None:
        self.name = settings.llm_model
        self._tokenizer = None
        self._model = None
        self._device = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._device = (
            "cuda" if (settings.llm_device == "auto" and torch.cuda.is_available())
            else settings.llm_device
        )
        if self._device == "auto":
            self._device = "cpu"

        log.info("Loading %s on %s (4bit=%s)", settings.llm_model, self._device, settings.use_4bit)

        self._tokenizer = AutoTokenizer.from_pretrained(settings.llm_model)

        kwargs: dict = {"torch_dtype": torch.bfloat16}
        if settings.use_4bit and self._device == "cuda":
            try:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
                kwargs["device_map"] = "auto"
            except ImportError:
                log.warning("bitsandbytes unavailable — falling back to bf16 full precision")
                kwargs["device_map"] = "auto"
        else:
            kwargs["device_map"] = self._device

        self._model = AutoModelForCausalLM.from_pretrained(settings.llm_model, **kwargs)
        self._model.eval()
        self._loaded = True

    def complete(self, system: str, messages: list[dict]) -> str:
        if not self._loaded:
            self.load()
        import torch

        chat = [{"role": "system", "content": system}, *messages]
        prompt = self._tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=settings.llm_max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                repetition_penalty=1.05,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def make_llm() -> LLM:
    """Construct and eagerly load the configured LLM. Raises on failure."""
    llm = TransformersLLM()
    llm.load()
    return llm
