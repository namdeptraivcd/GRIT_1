from __future__ import annotations

from dataclasses import dataclass

import torch


_RISK_SCORE = {"Safe": 1.0, "Controversial": 0.5, "Unsafe": 0.0}


@dataclass(frozen=True)
class PrefixScores:
    values: list[float]


class SafetyStreamPRM:
    """Frozen Qwen3Guard-Stream adapter producing prefix safety potentials."""

    def __init__(self, model_path: str, device: torch.device) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        if device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        if not hasattr(self.model, "stream_moderate_from_ids"):
            raise TypeError(f"{model_path} does not expose the Qwen3Guard stream API")

    @torch.inference_mode()
    def score_one(self, prompt: str, response: str) -> PrefixScores:
        rendered = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        ids = self.tokenizer(rendered, return_tensors="pt").input_ids[0].to(self.device)
        values = ids.tolist()
        im_start = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        user = self.tokenizer.convert_tokens_to_ids("user")
        im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        user_start = next(
            i for i in range(len(values) - 1) if values[i : i + 2] == [im_start, user]
        )
        user_end = next(i for i in range(user_start + 2, len(values)) if values[i] == im_end)

        _query_result, state = self.model.stream_moderate_from_ids(
            ids[: user_end + 1], role="user", stream_state=None
        )
        scores: list[float] = []
        for token_id in ids[user_end + 1 :]:
            result, state = self.model.stream_moderate_from_ids(
                token_id, role="assistant", stream_state=state
            )
            label = result["risk_level"][-1]
            scores.append(_RISK_SCORE[label])
        self.model.close_stream(state)
        return PrefixScores(values=scores or [0.5])

    def score_batch(self, prompts: list[str], responses: list[str]) -> list[PrefixScores]:
        return [self.score_one(prompt, response) for prompt, response in zip(prompts, responses)]
