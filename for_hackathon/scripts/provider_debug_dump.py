#!/usr/bin/env python3
"""Dump provider compatibility info for debugging empty chat content.

Collects:
1) GET /v1/models output
2) Raw request payload for /v1/chat/completions
3) Raw response JSON for /v1/chat/completions
4) Extracted fields summary useful for parser fixes

Usage:
  uv run python scripts/provider_debug_dump.py \
    --api-key <token> \
    --base-url https://provider.intellemma.ru/v1 \
    --model openai/gpt-oss-20b
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import httpx


DEFAULT_BASE_URL = "https://provider.intellemma.ru/v1"
DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_API_KEY = "0726dd7e0fff495120de20e640be4cd341bb7720fdb9d77101e88975e2870e82"


def _safe_get(dct: Dict[str, Any], *path: str) -> Any:
    cur: Any = dct
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def build_chat_payload(model: str, prompt: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a concise and helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump provider models and raw chat/completions response")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Provider base URL")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="Provider API token")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name")
    parser.add_argument("--prompt", default="Return exactly one short sentence about apples.", help="User prompt")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--output",
        default="provider_debug_dump.json",
        help="Path to output JSON file",
    )
    args = parser.parse_args()

    headers = {
        "Authorization": f"Bearer {args.api_key}",
        "Content-Type": "application/json",
    }

    base_url = args.base_url.rstrip("/")
    models_url = f"{base_url}/models"
    chat_url = f"{base_url}/chat/completions"

    payload = build_chat_payload(
        model=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    report: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "request": {
            "models_url": models_url,
            "chat_url": chat_url,
            "headers": {"Authorization": "Bearer ***", "Content-Type": "application/json"},
            "chat_payload": payload,
        },
        "responses": {},
        "parsed_summary": {},
    }

    chat_json: Dict[str, Any] = {}
    with httpx.Client(timeout=60.0) as client:
        try:
            models_resp = client.get(models_url, headers=headers)
            report["responses"]["models_status"] = models_resp.status_code
            try:
                report["responses"]["models_json"] = models_resp.json()
            except Exception:
                report["responses"]["models_text"] = models_resp.text
        except Exception as e:
            report["responses"]["models_error"] = repr(e)

        try:
            chat_resp = client.post(chat_url, headers=headers, json=payload)
            report["responses"]["chat_status"] = chat_resp.status_code
            try:
                chat_json = chat_resp.json()
                report["responses"]["chat_json"] = chat_json
            except Exception:
                report["responses"]["chat_text"] = chat_resp.text
        except Exception as e:
            report["responses"]["chat_error"] = repr(e)

    choice0 = None
    if isinstance(chat_json, dict):
        choices = chat_json.get("choices")
        if isinstance(choices, list) and choices:
            choice0 = choices[0]

    message = choice0.get("message") if isinstance(choice0, dict) else None
    report["parsed_summary"] = {
        "id": _safe_get(chat_json, "id"),
        "object": _safe_get(chat_json, "object"),
        "model": _safe_get(chat_json, "model"),
        "finish_reason": choice0.get("finish_reason") if isinstance(choice0, dict) else None,
        "message_role": message.get("role") if isinstance(message, dict) else None,
        "message_content_type": type(message.get("content")).__name__ if isinstance(message, dict) else None,
        "message_content": message.get("content") if isinstance(message, dict) else None,
        "message_tool_calls": message.get("tool_calls") if isinstance(message, dict) else None,
        "usage": chat_json.get("usage") if isinstance(chat_json, dict) else None,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved debug report to: {out_path}")
    print(f"models status: {report['responses'].get('models_status')}")
    print(f"chat status: {report['responses'].get('chat_status')}")
    print("parsed summary:")
    print(json.dumps(report["parsed_summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
