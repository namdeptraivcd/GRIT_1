from __future__ import annotations

import concurrent.futures
import re
from dataclasses import dataclass

from openai import OpenAI


@dataclass(frozen=True)
class SafetyScore:
    reward: float
    label: str
    refusal: bool


class VLLMRolloutClient:
    def __init__(self, base_url: str, model: str, *, timeout: float = 180.0) -> None:
        self.model = model
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=base_url.rstrip("/") + "/v1",
            timeout=timeout,
            max_retries=2,
        )

    def generate(
        self,
        prompts: list[str],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        workers: int = 8,
    ) -> list[str]:
        def one(prompt: str) -> str:
            result = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            return result.choices[0].message.content or ""

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            return list(pool.map(one, prompts))


class VLLMSafetyClient:
    _label = re.compile(r"^Safety:\s*(Safe|Unsafe|Controversial)\s*$", re.MULTILINE)
    _refusal = re.compile(r"^Refusal:\s*(Yes|No)\s*$", re.MULTILINE)

    def __init__(self, base_url: str, model: str, *, timeout: float = 180.0) -> None:
        self.model = model
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=base_url.rstrip("/") + "/v1",
            timeout=timeout,
            max_retries=2,
        )

    def score_one(self, prompt: str, response: str) -> SafetyScore:
        result = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
            temperature=0.0,
            max_tokens=96,
        )
        content = result.choices[0].message.content or ""
        label_match = self._label.search(content)
        refusal_match = self._refusal.search(content)
        if label_match is None:
            raise RuntimeError(f"guard returned an invalid response: {content!r}")
        label = label_match.group(1)
        reward = {"Safe": 1.0, "Controversial": 0.25, "Unsafe": -1.0}[label]
        return SafetyScore(
            reward=reward,
            label=label,
            refusal=bool(refusal_match and refusal_match.group(1) == "Yes"),
        )

    def score_batch(
        self, prompts: list[str], responses: list[str], *, workers: int = 8
    ) -> list[SafetyScore]:
        if len(prompts) != len(responses):
            raise ValueError("prompt and response counts differ")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            return list(pool.map(lambda pair: self.score_one(*pair), zip(prompts, responses)))
