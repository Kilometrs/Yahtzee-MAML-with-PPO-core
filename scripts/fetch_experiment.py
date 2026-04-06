#!/usr/bin/env python3
"""Fetch ClearML experiment data (logs, config, scalars, artifacts) by task ID.

Usage:
    python scripts/fetch_experiment.py <task_id> [--output ./output]
"""

import argparse
import json
import os
import sys
from pathlib import Path


def _load_dotenv(path: str = ".env") -> None:
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())
    except FileNotFoundError:
        pass


_load_dotenv()


def fetch_experiment(task_id: str, output_dir: Path) -> None:
    from clearml import Task

    print(f"Connecting to ClearML task: {task_id}")
    task = Task.get_task(task_id=task_id)

    run_dir = output_dir / task_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- Config / hyperparameters ---
    params = task.get_parameters()
    if params:
        config_path = run_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(params, f, indent=2)
        print(f"  Saved config -> {config_path}")

    # --- Scalar metrics ---
    scalars = task.get_reported_scalars()
    if scalars:
        scalars_path = run_dir / "scalars.json"
        with open(scalars_path, "w") as f:
            json.dump(scalars, f, indent=2)
        print(f"  Saved scalars -> {scalars_path}")

    # --- Console logs ---
    logs = task.get_reported_console_output(number_of_reports=10_000)
    if logs:
        logs_path = run_dir / "console.log"
        with open(logs_path, "w") as f:
            f.write("\n".join(logs))
        print(f"  Saved console log -> {logs_path}")

    # --- Artifacts ---
    artifacts = task.artifacts
    if artifacts:
        artifacts_dir = run_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        for name, artifact in artifacts.items():
            print(f"  Downloading artifact: {name}")
            local_path = artifact.get_local_copy()
            if local_path:
                src = Path(local_path)
                dest = artifacts_dir / src.name
                src.replace(dest)
                # Convert parquet to JSON for easier inspection
                if dest.suffix == ".parquet":
                    try:
                        import pandas as pd

                        df = pd.read_parquet(dest)
                        json_dest = dest.with_suffix(".json")
                        df.to_json(json_dest, orient="records", indent=2)
                        print(f"    -> {dest} (+ {json_dest.name})")
                    except Exception as e:
                        print(f"    -> {dest} (parquet->json failed: {e})")
                else:
                    print(f"    -> {dest}")

    # --- Task metadata summary ---
    meta = {
        "id": task.id,
        "name": task.name,
        "project": task.get_project_name(),
        "status": str(task.get_status()),
        "tags": task.get_tags(),
        "started": str(task.data.started) if hasattr(task.data, "started") else None,
        "completed": str(task.data.completed) if hasattr(task.data, "completed") else None,
    }
    meta_path = run_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved metadata -> {meta_path}")

    print(f"\nDone. Output in: {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch ClearML experiment data by task ID.")
    parser.add_argument("task_id", help="ClearML task/experiment ID (hash)")
    parser.add_argument(
        "--output",
        default="./output",
        help="Output base directory (default: ./output)",
    )
    args = parser.parse_args()

    required = ["CLEARML_API_HOST", "CLEARML_API_ACCESS_KEY", "CLEARML_API_SECRET_KEY"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        print(f"Missing env vars: {', '.join(missing)}. Set them or add to .env", file=sys.stderr)
        sys.exit(1)

    fetch_experiment(args.task_id, Path(args.output))


if __name__ == "__main__":
    main()
