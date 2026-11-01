# SPDX-License-Identifier: Apache-2.0
"""Higgsfield transport: every call goes through the signed-in `higgsfield` CLI, never an HTTP key of its own.

One job is three steps, each a CLI call (higgsfield 1.2.2, measured 2026-10-30):

1. `generate cost <job_type> ...` prices the job without creating it, `{"credits": N}` with `--json`. A parameter
   the model does not take is refused here (exit 4, "Unknown params: ..."), so a bad flag never costs credits.
2. `generate create <job_type> ... --json` (no `--wait`) answers the job ids, a JSON array of strings. From here
   the credits are spent: nothing after this step creates the job again.
3. `generate wait <id> --json --quiet --timeout ...` answers the job object, `status` and `result_url`. A wait that
   gets no answer ("request failed (no response received)", seen on the first probe) is asked again for the same
   id until the deadline; a job still unfinished at the deadline is named by its id, not resubmitted.

Media flags take a local path and the CLI uploads it (`--image-references`, `--start-image`, `--end-image`).
The result is fetched from `result_url` (a CDN URL with no credential on it).

Billing is Higgsfield credits on the user's plan, said on stderr before the job is created (`CREDIT_WORDS`), with
the price `generate cost` gave.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from .base import HIGGSFIELD_CREDIT_WORDS, PROVIDER_HIGGSFIELD, provider_binary, provider_subprocess_env

BINARY = "higgsfield"
# Replaces the binary (the test suite points it at nothing, so no test can spend credits).
BINARY_ENV = "SPRITE_GEN_HIGGSFIELD_BIN"
# npm puts a `.cmd` shim on PATH on Windows, and cmd.exe cuts an argument at its first line break: a prompt of
# several lines lost everything after its first line, `--json` too ("generate cost answered no JSON", 2026-10-30).
# The npm package carries the CLI's own executable at this path beside the shim; it is called instead.
SHIM_SUFFIXES = (".cmd", ".bat")
NATIVE_BESIDE_SHIM = Path("node_modules") / "@higgsfield" / "cli" / "vendor" / "hf.exe"
LOGIN_HINT = "`higgsfield auth login` signs your Higgsfield account in"
# The line a batch verb's log forwards to the terminal (`base.run_logged`), like a per-call API charge.
CREDIT_WORDS = HIGGSFIELD_CREDIT_WORDS
# A job that answers neither completed nor one of these is still running.
DONE_STATUSES = ("completed",)
FAILED_STATUSES = ("failed", "nsfw", "canceled", "cancelled", "error", "rejected")
# A CLI call that is not a job wait (cost, create, account) answers in seconds.
CALL_TIMEOUT_SECONDS = 120
DOWNLOAD_TIMEOUT_SECONDS = 300
_NO_ANSWER = re.compile(r"no response received|timeout|timed out|connection (reset|refused)|EOF", re.I)

Run = Callable[..., subprocess.CompletedProcess]


class HiggsfieldError(SystemExit):
    """A Higgsfield call failed. `job_id` is set once a job was created (its credits may be spent)."""

    def __init__(self, message: str, *, job_id: str | None = None) -> None:
        super().__init__(message)
        self.job_id = job_id


@dataclass
class Job:
    job_type: str
    params: dict[str, Any] = field(default_factory=dict)
    # Media flag (without dashes) -> local paths, e.g. {"image-references": [ref]} or {"start-image": [still]}.
    media: dict[str, list[Path]] = field(default_factory=dict)

    def args(self) -> list[str]:
        args = [self.job_type]
        for name, value in self.params.items():
            if value is None:
                continue
            if isinstance(value, bool):
                value = "true" if value else "false"
            args += [f"--{name}", str(value)]
        for flag, paths in self.media.items():
            for path in paths:
                args += [f"--{flag}", str(Path(path).resolve())]
        return args


def _said(done: subprocess.CompletedProcess) -> str:
    return " ".join((done.stderr or done.stdout or "").split())[:400]


def binary() -> str:
    """The CLI to call: `BINARY_ENV` when set, else `higgsfield` on PATH, or, where that is npm's cmd.exe shim,
    the executable the npm package carries beside it (`NATIVE_BESIDE_SHIM`)."""
    named = os.environ.get(BINARY_ENV)
    if named:
        return provider_binary(named)
    found = provider_binary(BINARY)
    if Path(found).suffix.lower() in SHIM_SUFFIXES:
        native = Path(found).parent / NATIVE_BESIDE_SHIM
        if native.is_file():
            return str(native)
    return found


def call(args: list[str], *, timeout: float, run: Run | None = None) -> subprocess.CompletedProcess:
    """One CLI call with `--json`; a binary that is missing or does not answer in `timeout` is a HiggsfieldError.
    A cmd.exe shim is refused an argument of several lines, which it would cut short, before it is run."""
    program = binary()
    if Path(program).suffix.lower() in SHIM_SUFFIXES and any("\n" in arg or "\r" in arg for arg in args):
        raise HiggsfieldError(f"higgsfield: {program} is a cmd.exe shim, which cuts an argument at its first line "
                              f"break, and this call carries a prompt of several lines; set {BINARY_ENV} to the CLI's "
                              f"own executable (npm installs it as {NATIVE_BESIDE_SHIM} beside the shim)")
    command = [program, *args, "--json"]
    try:
        # Late-bound, so a test that monkeypatches subprocess.run is followed.
        return (run or subprocess.run)(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                   env=provider_subprocess_env(), timeout=timeout, stdin=subprocess.DEVNULL)
    except FileNotFoundError as exc:
        raise HiggsfieldError(f"higgsfield: the `{BINARY}` CLI is not on PATH; install it and "
                              f"{LOGIN_HINT}") from exc
    except subprocess.TimeoutExpired as exc:
        raise HiggsfieldError(f"higgsfield: `{BINARY} {args[0]} {args[1] if len(args) > 1 else ''}` gave no "
                              f"answer in {timeout:g}s") from exc


def _json(done: subprocess.CompletedProcess, what: str) -> Any:
    try:
        return json.loads(done.stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        raise HiggsfieldError(f"higgsfield: {what} answered no JSON: {_said(done)}") from exc


def account(*, run: Run | None = None) -> dict[str, Any] | None:
    """The signed-in account (`credits`, `email`, `subscription_plan_type`), or None when not signed in."""
    done = call(["account", "status"], timeout=CALL_TIMEOUT_SECONDS, run=run)
    if done.returncode:
        return None
    reply = _json(done, "account status")
    return reply if isinstance(reply, dict) else None


def cost(job: Job, *, run: Run | None = None) -> float:
    """The job's price in credits. Validates every parameter against the model; nothing is created."""
    done = call(["generate", "cost", *job.args()], timeout=CALL_TIMEOUT_SECONDS, run=run)
    if done.returncode:
        raise HiggsfieldError(f"higgsfield: {job.job_type} refused before anything was spent: {_said(done)}")
    reply = _json(done, "generate cost")
    credits = reply.get("credits") if isinstance(reply, dict) else None
    if not isinstance(credits, (int, float)):
        raise HiggsfieldError(f"higgsfield: generate cost answered no price: {_said(done)}")
    return float(credits)


def announce(who: str, job: Job, credits: float, detail: str = "") -> None:
    print(f"[{who}] higgsfield: {job.job_type} {CREDIT_WORDS}: {credits:g} credits{detail}", file=sys.stderr,
          flush=True)


def create(job: Job, *, run: Run | None = None) -> str:
    """Create the job and return its id. Credits are spent from here."""
    done = call(["generate", "create", *job.args()], timeout=CALL_TIMEOUT_SECONDS, run=run)
    if done.returncode:
        raise HiggsfieldError(f"higgsfield: {job.job_type} was not created: {_said(done)}")
    reply = _json(done, "generate create")
    ids = [item.get("id") if isinstance(item, dict) else item for item in reply] if isinstance(reply, list) else []
    if len(ids) != 1 or not isinstance(ids[0], str) or not ids[0]:
        raise HiggsfieldError(f"higgsfield: expected one job id from generate create, got {done.stdout.strip()[:200]!r}")
    return ids[0]


def wait(job_id: str, *, deadline: float, interval: str = "3s", run: Run | None = None,
         clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    """The finished job. A wait with no answer is asked again for the same id; a failed job, or one unfinished
    at `deadline`, is a HiggsfieldError naming the id (`higgsfield generate get <id>` reads it later)."""
    later = f"; nothing was resubmitted — `{BINARY} generate get {job_id}` reads it later"
    while True:
        left = deadline - clock()
        if left <= 0:
            raise HiggsfieldError(f"higgsfield: job {job_id} did not finish in time{later}", job_id=job_id)
        try:
            done = call(["generate", "wait", job_id, "--quiet", "--timeout", f"{max(1, int(left))}s",
                         "--interval", interval], timeout=left + 30, run=run)
        except HiggsfieldError as exc:
            raise HiggsfieldError(f"{exc}{later}", job_id=job_id) from exc
        if done.returncode:
            said = _said(done)
            if _NO_ANSWER.search(said) and deadline - clock() > 5:
                print(f"[higgsfield] wait for {job_id} got no answer ({said}); asking again", file=sys.stderr)
                sleep(3)
                continue
            raise HiggsfieldError(f"higgsfield: waiting for job {job_id} failed: {said}{later}", job_id=job_id)
        reply = _json(done, "generate wait")
        if isinstance(reply, list) and len(reply) == 1:
            reply = reply[0]
        if not isinstance(reply, dict):
            raise HiggsfieldError(f"higgsfield: generate wait answered no job for {job_id}{later}", job_id=job_id)
        status = str(reply.get("status") or "")
        if status in DONE_STATUSES:
            return reply
        if status in FAILED_STATUSES:
            raise HiggsfieldError(f"higgsfield: job {job_id} ended {status}", job_id=job_id)
        # The CLI returned before the job ended (its own timeout): ask again while time is left.


def result_url(reply: Mapping[str, Any]) -> str:
    url = reply.get("result_url")
    if not isinstance(url, str) or not url.startswith("https://"):
        raise HiggsfieldError(f"higgsfield: job {reply.get('id')} completed with no result_url",
                              job_id=reply.get("id"))
    return url


def download(url: str, *, timeout: float = DOWNLOAD_TIMEOUT_SECONDS, attempts: int = 3) -> bytes:
    """The result's bytes. A transport failure is asked again; the job is never recreated for it."""
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "sprite-gen"}),
                                        timeout=timeout) as response:
                data = response.read()
            if data:
                return data
            last = ValueError("empty body")
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last = exc
        time.sleep(2 * (attempt + 1))
    raise HiggsfieldError(f"higgsfield: the result could not be downloaded ({type(last).__name__}: {last}); "
                          f"it is still at {url}")


def run_job(job: Job, *, who: str, timeout: float, detail: str = "",
            run: Run | None = None) -> tuple[bytes, dict[str, Any]]:
    """Price, create, wait for and download one job: its bytes and what the report keeps."""
    credits = cost(job, run=run)
    announce(who, job, credits, detail)
    started = time.monotonic()
    job_id = create(job, run=run)
    reply = wait(job_id, deadline=started + timeout, run=run)
    url = result_url(reply)
    data = download(url)
    return data, {"provider": PROVIDER_HIGGSFIELD, "model": job.job_type, "job_id": job_id, "credits": credits,
                  "auth_source": "higgsfield-login", "transport": "higgsfield-cli",
                  "elapsed_seconds": round(time.monotonic() - started, 3)}
