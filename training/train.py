"""Fine-tune NetVision's YOLO models on Roboflow datasets.

Run this on a machine with a GPU (Google Colab works well, see
``NetVision_Training_Colab.ipynb``). The trained weights are copied into
``models/`` with the file names the pipeline looks for.

Examples::

    export ROBOFLOW_API_KEY=...            # https://app.roboflow.com/settings/api
    python training/train.py player --epochs 100
    python training/train.py ball --epochs 250 --imgsz 1280
    python training/train.py court --epochs 300
    python training/train.py rim --workspace <ws> --project <project> --version 1

Default datasets are the public Roboflow Universe datasets used in the
tutorial this project is based on. The player dataset also labels the hoop,
so a player model trained on it gives NetVision a rim detector for free.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import yaml

TASKS = {
    "player": {
        "workspace": "workspace-5ujvu", "project": "basketball-players-fy4c2-vfsuv", "version": 17,
        "format": "yolov11", "base": "yolo11s.pt", "imgsz": 640, "epochs": 100, "output": "player_detector.pt",
    },
    "ball": {
        "workspace": "workspace-5ujvu", "project": "basketball-players-fy4c2-vfsuv", "version": 17,
        "format": "yolov11", "base": "yolo11s.pt", "imgsz": 1280, "epochs": 250, "output": "ball_detector_model.pt",
        # motion blur style augmentation helps with a fast moving ball
        "extra": {"degrees": 5.0, "scale": 0.6, "mosaic": 1.0, "mixup": 0.1},
    },
    "court": {
        "workspace": "fyp-3bwmg", "project": "reloc2-den7l", "version": 1,
        "format": "yolov8", "base": "yolo11s-pose.pt", "imgsz": 640, "epochs": 300, "output": "court_keypoint_detector.pt",
        # flipping left/right would swap keypoint meanings, so turn it off
        "extra": {"fliplr": 0.0},
    },
    "rim": {
        "workspace": None, "project": None, "version": None,
        "format": "yolov11", "base": "yolo11s.pt", "imgsz": 640, "epochs": 100, "output": "rim_detector.pt",
    },
}


def download_dataset(workspace: str, project: str, version: int, fmt: str, dest: Path) -> Path:
    try:
        from roboflow import Roboflow
    except ImportError:
        sys.exit("Install roboflow first: pip install roboflow")
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        sys.exit("Set the ROBOFLOW_API_KEY environment variable (https://app.roboflow.com/settings/api)")
    dest.mkdir(parents=True, exist_ok=True)
    current = os.getcwd()
    os.chdir(dest)
    try:
        dataset = Roboflow(api_key=api_key).workspace(workspace).project(project).version(version).download(fmt)
    finally:
        os.chdir(current)
    return Path(dest / dataset.location if not os.path.isabs(dataset.location) else dataset.location)


def fix_data_yaml(dataset_dir: Path) -> Path:
    """Roboflow exports use relative paths that often break; rewrite them as absolute paths."""
    data_file = dataset_dir / "data.yaml"
    data = yaml.safe_load(data_file.read_text())
    for split, folder in (("train", "train"), ("val", "valid"), ("test", "test")):
        images = dataset_dir / folder / "images"
        if images.exists():
            data[split] = str(images.resolve())
        else:
            data.pop(split, None)
    data_file.write_text(yaml.safe_dump(data, sort_keys=False))
    return data_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", choices=sorted(TASKS))
    parser.add_argument("--workspace")
    parser.add_argument("--project")
    parser.add_argument("--version", type=int)
    parser.add_argument("--base", help="Starting weights, e.g. yolo11m.pt")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int, default=-1, help="-1 lets Ultralytics pick the batch size")
    parser.add_argument("--device", default=None)
    parser.add_argument("--data-dir", default="datasets")
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()

    spec = dict(TASKS[args.task])
    for key in ("workspace", "project", "version", "base", "epochs", "imgsz"):
        if getattr(args, key) is not None:
            spec[key] = getattr(args, key)
    if not (spec["workspace"] and spec["project"] and spec["version"]):
        sys.exit(f"Task '{args.task}' needs --workspace, --project and --version for its Roboflow dataset")

    from ultralytics import YOLO

    dataset_dir = download_dataset(spec["workspace"], spec["project"], spec["version"], spec["format"], Path(args.data_dir))
    data_yaml = fix_data_yaml(dataset_dir)
    model = YOLO(spec["base"])
    results = model.train(
        data=str(data_yaml), epochs=spec["epochs"], imgsz=spec["imgsz"], batch=args.batch, device=args.device,
        project="runs", name=args.task, plots=True, **spec.get("extra", {}),
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    target = Path(args.models_dir) / spec["output"]
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, target)
    print(f"Saved {target}")


if __name__ == "__main__":
    main()
