#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Collect a bounded, repeatable expert-routing corpus from an eager server.

Run each split in a fresh server with its own R9V_ROUTE_PROFILE_DIR. These are
routing measurements, never throughput results. No production prompts are used.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

try:
    from tools.observability import Reporter
    from tools.rank_experts import counts
    from tools.runtime_workload import call, red_image
except ModuleNotFoundError:
    from observability import Reporter
    from rank_experts import counts
    from runtime_workload import call, red_image


TOPICS = {
    "train": [
        "Explain photosynthesis and the carbon cycle with concrete examples.",
        "Write Python code to merge overlapping intervals and explain edge cases.",
        "Compare TCP and UDP and describe an appropriate use for each.",
        "Explain how to solve two simultaneous linear equations with an example.",
        "Write a short story about a gardener repairing an abandoned greenhouse.",
        "Translate a paragraph about a railway journey into French and Spanish.",
        "Describe how to organize a small community science fair.",
        "Explain the differences between igneous, sedimentary, and metamorphic rocks.",
    ],
    "holdout": [
        "Explain the water cycle and how groundwater replenishes a river.",
        "Write Python code for breadth-first graph search and discuss disconnected nodes.",
        "Explain DNS resolution and caching with a worked example.",
        "Explain conditional probability using colored marbles in two bags.",
        "Write a short story about an astronomer repairing a mountain telescope.",
        "Translate a paragraph about a coastal village into German and Italian.",
        "Describe how to organize a neighborhood book exchange.",
        "Explain the differences between conductors, insulators, and semiconductors.",
    ],
}


def corpus(split):
    prompts = [
        {"messages": [{"role": "user", "content": topic}]} for topic in TOPICS[split]
    ]
    code = "ROUTE-TRAIN-613" if split == "train" else "ROUTE-HOLDOUT-927"
    prompts.append(
        {
            "messages": [
                {"role": "user", "content": f"Call record_code with code {code}."}
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "record_code",
                        "parameters": {
                            "type": "object",
                            "properties": {"code": {"type": "string"}},
                            "required": ["code"],
                        },
                    },
                }
            ],
            "tool_choice": "auto",
        }
    )
    prompts.append(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image and suggest three objects of the same color.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": red_image(
                                    512 if split == "train" else 1024,
                                    512 if split == "train" else 256,
                                )
                            },
                        },
                    ],
                }
            ]
        }
    )
    prompts.append(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Summarize the following observations and identify recurring themes. "
                    + (TOPICS[split][0] + " ") * 512,
                }
            ]
        }
    )
    return prompts


def durable_write(path, value):
    """Persist evidence and its directory entry before advancing the workload."""
    with path.open("xb") as stream:
        os.chmod(path, 0o600)
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    sync_directory(path.parent)


def sync_directory(directory):
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


_reporter = None


def event(directory, name, **fields):
    global _reporter
    if _reporter is None:
        _reporter = Reporter()
    _reporter.send("r9v.event", event=name, **fields)
    record = {
        "event": name,
        "time": time.time(),
        "monotonic_ns": time.monotonic_ns(),
        **fields,
    }
    with (directory / "events.jsonl").open("a") as stream:
        os.chmod(stream.name, 0o600)
        stream.write(json.dumps(record) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    sync_directory(directory)


def run(url, model, directory, split, *, collect=True, limit=None, identity=None):
    if limit is not None and (
        type(limit) is not int or not 1 <= limit <= len(corpus(split))
    ):
        raise ValueError("limit must be 1..11")
    if identity is not None and (not isinstance(identity, dict) or any(
        not isinstance(identity.get(key), str) or not identity[key]
        for key in ('model_package', 'model_hash', 'runtime_hash', 'config_hash'))):
        raise ValueError('Route identity requires model_package, model_hash, runtime_hash and config_hash')
    directory.mkdir(parents=True, exist_ok=True)
    if any(
        (directory / name).exists()
        for name in (
            "enable",
            "dump",
            "histogram-rank0.json",
            "corpus.json",
            "events.jsonl",
        )
    ):
        raise ValueError(
            "route directory must be unused; restart with a fresh directory for each split"
        )
    requests = corpus(split)[:limit]
    raw = json.dumps(requests, sort_keys=True).encode()
    durable_write(directory / "corpus.json", raw)
    event(
        directory, "session_start", collect=collect, split=split, requests=len(requests)
    )
    if collect:
        durable_write(directory / "enable", b"")
        event(directory, "histogram_enabled")
    results = []
    try:
        for index, request in enumerate(requests):
            body = {
                "model": model,
                "temperature": 0,
                "max_tokens": 192,
                "chat_template_kwargs": {"enable_thinking": False},
                **request,
            }
            started = time.monotonic()
            durable_write(
                directory / f"request-{index}.json", json.dumps(body).encode()
            )
            event(directory, "request_start", index=index,
                  request_sha256=hashlib.sha256(json.dumps(body).encode()).hexdigest())
            response = call(url, "/v1/chat/completions", body)
            durable_write(
                directory / f"response-{index}.json", json.dumps(response).encode()
            )
            event(directory, "response_saved", index=index)
            if not response.get("choices") or response["choices"][0].get(
                "finish_reason"
            ) not in ("stop", "length", "tool_calls"):
                raise ValueError(f"corpus request {index} failed")
            results.append(
                {
                    "index": index,
                    "seconds": time.monotonic() - started,
                    "usage": response.get("usage"),
                }
            )
        if collect:
            durable_write(directory / "dump", b"")
            event(directory, "histogram_dump_start")
            # The next complete layer group flushes the fixed-size device counters.
            call(
                url,
                "/v1/chat/completions",
                {
                    "model": model,
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "Say ready."}],
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            histogram = json.loads((directory / "histogram-rank0.json").read_text())
            counts(histogram)
            if identity is not None:
                histogram['metadata'] = dict(identity)
                histogram['capture'] = {'split': split, 'corpus_sha256': hashlib.sha256(raw).hexdigest(),
                    'raw_histogram_sha256': hashlib.sha256((directory / 'histogram-rank0.json').read_bytes()).hexdigest()}
                durable_write(directory / 'histogram-bound.json', json.dumps(histogram).encode())
            event(directory, "histogram_saved")
        durable_write(
            directory / "result.json",
            json.dumps(
                {
                    "passed": True,
                    "split": split,
                    "collection_enabled": collect,
                    "corpus_complete": limit is None or limit == len(corpus(split)),
                    "corpus_sha256": hashlib.sha256(raw).hexdigest(),
                    "requests": results,
                    "limitations": "Synthetic bounded corpus, eager execution; held-out hit rate and compiled serving qualification still required.",
                },
                indent=2,
            ).encode(),
        )
        event(directory, "session_complete")
    except Exception as error:
        event(
            directory,
            "session_error",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
    finally:
        (directory / "enable").unlink(missing_ok=True)
        sync_directory(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8004")
    parser.add_argument("--model", default="qwen3.8-flash-next")
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--identity", type=Path, help="supervisor-verified model/package/runtime/workload fingerprint JSON")
    parser.add_argument("--split", required=True, choices=tuple(TOPICS))
    parser.add_argument(
        "--no-collect",
        action="store_true",
        help="Replay the same corpus without enabling GPU histograms",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Replay only the first N requests (1..11); not full corpus qualification",
    )
    args = parser.parse_args()
    try:
        run(
            args.url,
            args.model,
            args.directory,
            args.split,
            collect=not args.no_collect,
            limit=args.limit,
            identity=json.loads(args.identity.read_text()) if args.identity else None,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Route collection failed: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
