"""Record the served model and version of a local OpenAI-compatible vLLM instance."""

import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import requests


def normalize_endpoint(endpoint):
    url = urlsplit(endpoint)
    if (url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost"}
            or url.username is not None or url.password is not None
            or url.path not in {"", "/"} or url.query or url.fragment):
        raise ValueError("expected a local HTTP origin such as http://127.0.0.1:18471")
    return endpoint.rstrip("/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18471")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    endpoint = normalize_endpoint(args.endpoint)
    with requests.Session() as session:
        session.trust_env = False
        models = session.get(f"{endpoint}/v1/models", timeout=10, allow_redirects=False)
        models.raise_for_status()
        if args.model not in {model["id"] for model in models.json()["data"]}:
            parser.error(f"the service does not advertise {args.model!r}")
        version = session.get(f"{endpoint}/version", timeout=10, allow_redirects=False)
        version.raise_for_status()
        receipt = {
            "served_model": args.model,
            "server_version": version.json(),
            "receipt_id": uuid4().hex,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(receipt, output, indent=2, ensure_ascii=False)
        output.write("\n")


if __name__ == "__main__":
    main()
