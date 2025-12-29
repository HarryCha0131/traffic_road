#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一鍵跑 YOLO 訓練流程（Ultralytics CLI），並做到：
1) 終端機即時顯示訓練進度
2) 同時把訓練/驗證/推論 log 存成檔案
3) 保存系統資訊、套件版本、data.yaml 快照
4) 將 Ultralytics 產出的 artifacts（results.csv, plots, best.pt...）複製到本次 run 資料夾
5) 最後把整個 run 資料夾打包成 .zip 方便搬回自己電腦
6) 參數：用來標記本次實驗（baseline / lighting / mix / yolo11n...）
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


# ----------------------------
# Utilities
# ----------------------------
def run_capture(cmd: list[str]) -> str:
    """Run command, capture stdout (best-effort)."""
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception as e:
        return f"[command failed] {' '.join(cmd)}\n{e}"


def write_text(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")


def write_json(p: Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def system_info() -> dict:
    return {
        "timestamp": datetime.now().isoformat(),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "uname": run_capture(["uname", "-a"]),
        "pip_freeze": run_capture([sys.executable, "-m", "pip", "freeze"]),
        "nvidia_smi": run_capture(["bash", "-lc", "nvidia-smi"]),
        "nvcc": run_capture(["bash", "-lc", "nvcc --version"]),
        "git_head": run_capture(["bash", "-lc", "git rev-parse HEAD"]),
        "git_status": run_capture(["bash", "-lc", "git status --porcelain"]),
    }


def zip_dir(src_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zf:
        for p in src_dir.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(src_dir))


def newest_run_dir(parent: Path, prefix: str) -> Path | None:
    """Pick newest directory matching prefix (e.g. train*, val*)"""
    cand = [p for p in parent.glob(f"{prefix}*") if p.is_dir()]
    if not cand:
        return None
    cand.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cand[0]


def sh_quote(path: Path) -> str:
    """Simple shell quoting for paths with spaces."""
    s = str(path)
    return "'" + s.replace("'", "'\"'\"'") + "'"


def sanitize_tag(tag: str) -> str:
    """
    Make tag filename-safe-ish:
    - keep letters/numbers/._- only
    - replace spaces with _
    """
    tag = tag.strip().replace(" ", "_")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    return "".join(ch if ch in allowed else "_" for ch in tag)[:60]  # cap length


def run_with_tee(cmd: str, log_path: Path) -> int:
    """
    Run shell command with:
    - print to terminal
    - append full output to log_path
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd_tee = f"{cmd} 2>&1 | tee -a {sh_quote(log_path)}"
    return subprocess.call(cmd_tee, shell=True)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data", type=str, required=True, help="path to data.yaml")
    ap.add_argument("--model", type=str, default="yolo11n.pt", help="e.g. yolo11n.pt / yolov8n.pt")
    ap.add_argument("--task", type=str, default="detect", choices=["detect", "segment", "pose"])

    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", type=str, default="0", help='e.g. "0" / "0,1" / "cpu"')
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--project", type=str, default="runs_local", help="output root folder")
    ap.add_argument("--name", type=str, default="", help="custom run name (optional)")
    ap.add_argument("--tag", type=str, default="", help="experiment tag, e.g. baseline / lighting / mix / yolo11n")
    ap.add_argument("--extra", type=str, default="", help="extra ultralytics args, e.g. 'cos_lr=True lr0=0.01'")

    ap.add_argument("--do_val", action="store_true", help="run val after training (uses best.pt)")
    ap.add_argument("--do_predict", action="store_true", help="run predict after training (uses best.pt)")
    ap.add_argument("--predict_source", type=str, default="", help="path to images folder for predict")
    ap.add_argument("--conf", type=float, default=0.25)

    args = ap.parse_args()

    data_yaml = Path(args.data).resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    # --- Create our run folder ---
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = args.name.strip() or f"{Path(args.model).stem}_{args.task}_{stamp}"

    tag = sanitize_tag(args.tag) if args.tag.strip() else ""
    run_name = f"{tag}__{base_name}" if tag else base_name

    out_root = Path(args.project).resolve() / run_name
    out_root.mkdir(parents=True, exist_ok=True)

    # --- Save metadata ---
    write_json(out_root / "meta" / "run_args.json", {
        "run_name": run_name,
        "tag": tag,
        "args": vars(args),
        "data_yaml": str(data_yaml),
    })
    write_json(out_root / "meta" / "system_info.json", system_info())

    # Copy data.yaml snapshot
    shutil.copy2(data_yaml, out_root / "meta" / "data.yaml")

    # Ultralytics output directory (nested inside our run so everything packs nicely)
    ul_project = out_root / "ultralytics_runs"
    ul_project.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Train
    # ----------------------------
    train_log = out_root / "logs" / "train.log"

    train_cmd = (
        f"yolo {args.task} train "
        f"model={args.model} "
        f"data={sh_quote(data_yaml)} "
        f"imgsz={args.imgsz} "
        f"epochs={args.epochs} "
        f"batch={args.batch} "
        f"device={args.device} "
        f"workers={args.workers} "
        f"seed={args.seed} "
        f"project={sh_quote(ul_project)} "
        f"name=train "
        f"exist_ok=True "
    )
    if args.extra.strip():
        train_cmd += " " + args.extra.strip()

    write_text(train_log, f"[CMD]\n{train_cmd}\n\n")

    t0 = time.time()
    ret = run_with_tee(train_cmd, train_log)
    elapsed = time.time() - t0
    write_json(out_root / "meta" / "timing.json", {"train_seconds": elapsed, "train_return_code": ret})

    if ret != 0:
        print(f"\n❌ Training failed. See log: {train_log}")
        sys.exit(ret)

    # Find ultralytics train run folder
    ul_train_run = ul_project / "train"
    if not ul_train_run.exists():
        newest = newest_run_dir(ul_project, "train")
        if newest is None:
            raise RuntimeError(f"Cannot find ultralytics train run under: {ul_project}")
        ul_train_run = newest

    # Copy artifacts (train)
    artifacts_dir = out_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ul_train_run, artifacts_dir / "train_run", dirs_exist_ok=True)

    # Determine best weights path
    best_pt = artifacts_dir / "train_run" / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = artifacts_dir / "train_run" / "weights" / "last.pt"

    # ----------------------------
    # Val (optional)
    # ----------------------------
    if args.do_val:
        val_log = out_root / "logs" / "val.log"
        val_cmd = (
            f"yolo {args.task} val "
            f"model={sh_quote(best_pt)} "
            f"data={sh_quote(data_yaml)} "
            f"imgsz={args.imgsz} "
            f"device={args.device} "
            f"project={sh_quote(ul_project)} "
            f"name=val "
            f"exist_ok=True "
        )
        write_text(val_log, f"[CMD]\n{val_cmd}\n\n")
        ret2 = run_with_tee(val_cmd, val_log)
        write_json(out_root / "meta" / "val_status.json", {"val_return_code": ret2})

        ul_val_run = ul_project / "val"
        if not ul_val_run.exists():
            newest = newest_run_dir(ul_project, "val")
            if newest is not None:
                ul_val_run = newest
        if ul_val_run.exists():
            shutil.copytree(ul_val_run, artifacts_dir / "val_run", dirs_exist_ok=True)

    # ----------------------------
    # Predict (optional)
    # ----------------------------
    if args.do_predict:
        pred_source = Path(args.predict_source).resolve() if args.predict_source.strip() else None
        if not pred_source or not pred_source.exists():
            print("\n⚠️ --do_predict enabled but --predict_source not found. Skipping predict.")
        else:
            pred_log = out_root / "logs" / "predict.log"
            pred_cmd = (
                f"yolo {args.task} predict "
                f"model={sh_quote(best_pt)} "
                f"source={sh_quote(pred_source)} "
                f"conf={args.conf} "
                f"device={args.device} "
                f"project={sh_quote(ul_project)} "
                f"name=predict "
                f"exist_ok=True "
            )
            write_text(pred_log, f"[CMD]\n{pred_cmd}\n\n")
            ret3 = run_with_tee(pred_cmd, pred_log)
            write_json(out_root / "meta" / "predict_status.json", {"predict_return_code": ret3})

            ul_pred_run = ul_project / "predict"
            if not ul_pred_run.exists():
                newest = newest_run_dir(ul_project, "predict")
                if newest is not None:
                    ul_pred_run = newest
            if ul_pred_run.exists():
                shutil.copytree(ul_pred_run, artifacts_dir / "predict_run", dirs_exist_ok=True)

    # ----------------------------
    # Package ZIP
    # ----------------------------
    zip_path = out_root.parent / f"{run_name}.zip"
    zip_dir(out_root, zip_path)

    print("\n✅ DONE")
    print(f"Run folder : {out_root}")
    print(f"ZIP file   : {zip_path}")
    print("（有 tag 的話，檔名會長得像 baseline__yolo11n_detect_YYYYMMDD_HHMMSS.zip）")


if __name__ == "__main__":
    main()