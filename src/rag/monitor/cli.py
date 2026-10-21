"""Command line of the monitoring helpers (spec P5-02).

    uv run python -m rag.monitor probe      # the scheduled probe: runs until Ctrl-C
    uv run python -m rag.monitor receiver   # the local alert receiver: runs until Ctrl-C
    uv run python -m rag.monitor prune      # delete old trace files once (--dry-run to look)
    uv run python -m rag.monitor langfuse-env   # add the missing Langfuse secrets to .env

Settings come from `ServiceSettings` (`probe_*`, `alert_*`, `trace_retention_*`; env or `.env`).
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from collections.abc import Sequence

import httpx

from rag.config import ServiceSettings
from rag.config.paths import repo_path
from rag.monitor.alert_receiver import AlertReceiver, serve
from rag.monitor.langfuse_env import ensure_env
from rag.monitor.probe import ProbeConfig, Prober, run_forever
from rag.monitor.prune import prune_files
from rag.observe import init_observability, shutdown_observability

logger = logging.getLogger(__name__)

PROBE_SERVICE_NAME = "rag-probe"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m rag.monitor", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("probe", help="send the fixed requests on a schedule and export the results")
    sub.add_parser("receiver", help="receive Prometheus alerts into data/observe/alerts.jsonl")
    p_prune = sub.add_parser("prune", help="delete old trace files")
    p_prune.add_argument("--dry-run", action="store_true", help="list, do not delete")
    p_env = sub.add_parser(
        "langfuse-env", help="write the Langfuse secrets that are missing to .env"
    )
    p_env.add_argument("--env-file", default=".env", help="default: .env at the repo root")
    return parser


def _wait_for_interrupt(stop: threading.Event) -> None:
    for name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, name, None)
        if sig is not None:
            signal.signal(sig, lambda *_: stop.set())
    stop.wait()


def main(argv: Sequence[str] | None = None, *, settings: ServiceSettings | None = None) -> int:
    settings = settings if settings is not None else ServiceSettings()
    args = build_parser().parse_args(argv)
    if args.command == "prune":
        removed = prune_files(
            repo_path(settings.trace_retention_dir),
            settings.trace_retention_pattern,
            max_age_s=settings.trace_retention_days * 86400.0,
            max_total_bytes=settings.trace_retention_max_mb * 1_000_000,
            now=time.time(),
            dry_run=args.dry_run,
        )
        verb = "would remove" if args.dry_run else "removed"
        print(f"{verb} {len(removed)} file(s)")
        for path in removed:
            print(f"  {path}")
        return 0
    if args.command == "langfuse-env":
        added = ensure_env(repo_path(args.env_file))
        print("added to the env file: " + (", ".join(added) if added else "nothing, all present"))
        return 0
    stop = threading.Event()
    if args.command == "receiver":
        receiver = AlertReceiver(
            repo_path(settings.alert_log_path),
            webhook_url=settings.alert_webhook_url.get_secret_value(),
        )
        server, thread = serve(receiver, settings.alert_receiver_host, settings.alert_receiver_port)
        print(
            f"alert receiver on http://{settings.alert_receiver_host}:"
            f"{settings.alert_receiver_port}, lines in {settings.alert_log_path}"
            + (", webhook set" if settings.alert_webhook_url.get_secret_value() else ""),
            file=sys.stderr,
        )
        try:
            _wait_for_interrupt(stop)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            receiver.flush(10.0)  # posts still queued for the webhook
        return 0
    # probe: its own OTLP metrics, exported often so an alert sees a failure within a minute
    probe_settings = settings.model_copy(
        update={
            "otel_metric_export_interval_s": settings.probe_metric_export_interval_s,
            "otel_auto_instrument": False,
        }
    )
    init_observability(probe_settings, PROBE_SERVICE_NAME)
    config = ProbeConfig.from_settings(settings)
    print(
        f"probe: {config.kb} at {config.base_url} every {config.interval_s:g} s"
        + (f", ask check at most {config.ask_daily_cap} a day" if config.ask_enabled else ""),
        file=sys.stderr,
    )
    try:
        with httpx.Client() as client:
            prober = Prober(config, client=client)
            worker = threading.Thread(target=run_forever, args=(prober, stop), daemon=True)
            worker.start()
            _wait_for_interrupt(stop)
            stop.set()
            worker.join(timeout=config.ask_timeout_s + 5)
    finally:
        shutdown_observability()
    return 0
