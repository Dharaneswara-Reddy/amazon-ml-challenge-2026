"""Run a shell command on the SageMaker notebook instance through its Jupyter terminal API.

Usage (no extra deps in the pipeline venv):
  uv run --with requests --with websocket-client scripts/notebook_exec.py 'df -h ~/SageMaker'

Auth: a short-lived presigned URL from `aws sagemaker create-presigned-notebook-instance-url` using $AWS_PROFILE
(default org-admin). The auth token is never printed. Creates and deletes one Jupyter terminal; nothing billable.
"""

import json
import os
import subprocess
import sys
import time
import uuid
from urllib.parse import urlparse

import requests
import websocket

NOTEBOOK = os.environ.get("NOTEBOOK_NAME", "ml-challenge-2026")
PROFILE = os.environ.get("AWS_PROFILE", "org-admin")
REGION = os.environ.get("AWS_REGION", "us-east-1")


def presigned_url() -> str:
    """Create a presigned notebook URL via the AWS CLI."""
    return subprocess.check_output(
        ["aws", "sagemaker", "create-presigned-notebook-instance-url", "--notebook-instance-name", NOTEBOOK,
         "--profile", PROFILE, "--region", REGION, "--query", "AuthorizedUrl", "--output", "text"],
        text=True,
    ).strip()


def login() -> tuple[requests.Session, str]:
    """Follow the presigned URL to obtain Jupyter session cookies; return the session and base URL."""
    url = presigned_url()
    host = urlparse(url).netloc
    s = requests.Session()
    r = s.get(url, timeout=60)
    r.raise_for_status()
    return s, f"https://{host}"


def run(command: str, timeout: float) -> int:
    """Run `command` in a new Jupyter terminal, stream output, and return its exit code."""
    s, base = login()
    xsrf = s.cookies.get("_xsrf", "")
    r = s.post(f"{base}/api/terminals", headers={"X-XSRFToken": xsrf}, timeout=60)
    r.raise_for_status()
    name = r.json()["name"]
    cookie = "; ".join(f"{c.name}={c.value}" for c in s.cookies)
    ws = websocket.create_connection(
        f"wss://{urlparse(base).netloc}/terminals/websocket/{name}", header=[f"Cookie: {cookie}"], timeout=30
    )
    tag = uuid.uuid4().hex[:8]
    marker = f"__DONE_{tag}__"
    # the marker is split in the command text so the terminal's echo of the command never matches it
    ws.send(json.dumps(["stdin", f"stty -echo; {{ {command} ; }} 2>&1; echo \"__DO\"\"NE_{tag}__$?\"\r"]))
    buf, code, deadline = "", 1, time.time() + timeout
    try:
        while time.time() < deadline:
            try:
                msg = json.loads(ws.recv())
            except websocket.WebSocketTimeoutException:
                continue
            if msg[0] != "stdout":
                continue
            buf += msg[1]
            if marker in buf:
                out, _, rest = buf.partition(marker)
                sys.stdout.write(out)
                code = int("".join(ch for ch in rest.split()[0] if ch.isdigit()) or 1) if rest.split() else 1
                break
            lines = buf.split("\n")
            sys.stdout.write("\n".join(lines[:-1]) + ("\n" if len(lines) > 1 else ""))
            buf = lines[-1]
        else:
            sys.stdout.write(buf + f"\n[timeout after {timeout}s]\n")
    finally:
        ws.close()
        s.delete(f"{base}/api/terminals/{name}", headers={"X-XSRFToken": xsrf}, timeout=60)
    return code


if __name__ == "__main__":
    sys.exit(run(sys.argv[1], float(os.environ.get("NB_TIMEOUT", "900"))))
