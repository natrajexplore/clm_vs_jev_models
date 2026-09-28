"""Minimal client for TypeSafe's Jev (https://docs.typesafe.ai/api).

Raw HTTP instead of the typesafe-sdk package so latency and retries are measured and controlled here.
Responses are cached on disk (keyed by the full request payload) so reruns do not re-bill.
"""

import asyncio
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

RETRY_STATUS = {429, 500, 502, 503, 504, 529}


@dataclass
class JevAnswer:
    choice: str
    probabilities: dict[str, float]
    confidence: float | None
    latency_s: float  # of the original (uncached) request
    input_tokens: int | None
    model: str | None
    cached: bool


class JevClient:
    def __init__(
        self,
        model: str,
        base_url: str,
        max_concurrency: int = 8,
        timeout_s: float = 30.0,
        max_retries: int = 5,
        backoff_s: float = 1.0,
        cache_path: str | Path | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        api_key = os.environ.get("TYPESAFE_API_KEY")
        if not api_key:
            raise RuntimeError("Set the TYPESAFE_API_KEY environment variable (key from console.typesafe.ai/keys).")
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self.model = model
        self.base_url = base_url
        self.max_concurrency = max_concurrency
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.transport = transport
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, dict[str, Any]] = {}
        if self.cache_path and self.cache_path.exists():
            with open(self.cache_path) as f:
                for line in f:
                    entry = json.loads(line)
                    self.cache[entry["key"]] = entry

    def choice(self, states: list[str], instructions: str, criteria: dict[str, str]) -> list[JevAnswer]:
        """Ask one Choice question per state. Returns answers in input order."""
        payloads = [
            {
                "state": s,
                "model": self.model,
                "questions": {"label": {"type": "choice", "instructions": instructions, "criteria": criteria}},
            }
            for s in states
        ]
        return asyncio.run(self._run_all(payloads))

    async def _run_all(self, payloads: list[dict[str, Any]]) -> list[JevAnswer]:
        sem = asyncio.Semaphore(self.max_concurrency)
        async with httpx.AsyncClient(timeout=self.timeout_s, transport=self.transport) as client:
            return await asyncio.gather(*(self._one(client, sem, p) for p in payloads))

    async def _one(self, client: httpx.AsyncClient, sem: asyncio.Semaphore, payload: dict[str, Any]) -> JevAnswer:
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if key in self.cache:
            entry = self.cache[key]
            return _parse(entry["response"], entry["latency_s"], cached=True)

        async with sem:
            for attempt in range(self.max_retries + 1):
                t0 = time.perf_counter()
                try:
                    resp = await client.post(self.base_url, headers=self.headers, json=payload)
                except httpx.TransportError:
                    if attempt == self.max_retries:
                        raise
                else:
                    latency = time.perf_counter() - t0
                    if resp.status_code == 200:
                        break
                    if resp.status_code not in RETRY_STATUS or attempt == self.max_retries:
                        raise RuntimeError(f"Jev API error {resp.status_code}: {resp.text[:500]}")
                await asyncio.sleep(self.backoff_s * 2**attempt * (1 + random.random()))

        data = resp.json()
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "a") as f:
                f.write(json.dumps({"key": key, "response": data, "latency_s": latency}) + "\n")
        return _parse(data, latency, cached=False)


def _parse(data: dict[str, Any], latency_s: float, cached: bool) -> JevAnswer:
    answer = data["answers"]["label"]
    # The docs place `usage` per answer or per response; accept either.
    usage = answer.get("usage") or data.get("usage") or {}
    return JevAnswer(
        choice=answer["choice"],
        probabilities=answer["probabilities"],
        confidence=answer.get("confidence"),
        latency_s=latency_s,
        input_tokens=usage.get("input_tokens"),
        model=data.get("model"),
        cached=cached,
    )
