"""Google Colab / Cloud GPU dataset normalizer, SAM 2 polygon upgrader, and 2-stage trainer.

Supports all three YOLO export layouts automatically:
1. Pre-split Ultralytics layout: `images/{train,val|valid,test}` + `labels/{train,val|valid,test}`
2. Pre-split Roboflow layout: `{train,valid|val,test}/images` + `{train,valid|val,test}/labels`
   (including Roboflow exports where only `train/` is populated and `valid/`/`test/` are missing!)
3. Flat standardized export: `images/` + `labels/` + optional `manifest.jsonl`
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from leukerbad.dataset.sam_masker import IMAGE_EXTENSIONS, upgrade_bbox_labels_with_sam
from leukerbad.dataset.validate import validate_yolo_seg_dataset, write_yolo_dataset_yaml

SPLITS: tuple[str, str, str] = ("train", "val", "test")
_SPLIT_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("train", ("train",)),
    ("val", ("val", "valid", "validation")),
    ("test", ("test",)),
)


def _list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _detect_split_dirs(root: Path) -> dict[str, tuple[Path, Path]]:
    """Detect existing `(images_dir, labels_dir)` pairs in Ultralytics or Roboflow layout."""
    found: dict[str, tuple[Path, Path]] = {}
    for canonical_split, aliases in _SPLIT_ALIASES:
        for alias in aliases:
            # 1. Ultralytics layout: images/<alias> + labels/<alias>
            u_img = root / "images" / alias
            u_lbl = root / "labels" / alias
            if _list_images(u_img):
                found[canonical_split] = (u_img, u_lbl)
                break

            # 2. Roboflow layout: <alias>/images + <alias>/labels
            r_img = root / alias / "images"
            r_lbl = root / alias / "labels"
            if _list_images(r_img):
                found[canonical_split] = (r_img, r_lbl)
                break
    return found


def find_yolo_dataset_root(search_root: Path) -> Path:
    """Locate a YOLO dataset directory (flat or split, Ultralytics or Roboflow) inside `search_root`."""
    root = Path(search_root)
    if _detect_split_dirs(root):
        return root
    if _list_images(root / "images") and (root / "labels").is_dir():
        return root

    # Search subdirectories (ignoring macOS resource forks)
    candidates: list[Path] = []
    for sub in sorted(p for p in root.rglob("*") if p.is_dir() and "__MACOSX" not in p.parts):
        if _detect_split_dirs(sub) or (_list_images(sub / "images") and (sub / "labels").is_dir()):
            candidates.append(sub)

    if not candidates:
        raise FileNotFoundError(
            f"Could not locate any YOLO dataset (flat or split) inside {root}"
        )
    # Prefer shallowest matching root
    return min(candidates, key=lambda p: len(p.parts))


def _canonical_split_name(raw_split: object) -> str | None:
    if not isinstance(raw_split, str):
        return None
    cleaned = raw_split.strip().lower()
    for canonical, aliases in _SPLIT_ALIASES:
        if cleaned in aliases:
            return canonical
    return None


def assign_vape_manifest_splits(
    manifest_entries: list[dict[str, Any]],
    seed: int = 42,
) -> dict[str, str]:
    """Assign leak-free `train`/`val`/`test` splits for `public_data/exports/vape_yolo/manifest.jsonl`.

    Honors all pre-designated test/valid splits while fixing provenance traps:
    - `kaggle_test_manual` -> `test` (even though Roboflow exported it with `source_split="train"`).
    - `source_split == "test"` (`roboflow_primary`, `roboflow_additional`, `manual_photos`) -> `test`.
    - `source_split == "valid"` (`roboflow_primary`, `roboflow_additional`) -> `val`.
    - `roboflow_primary` cross-split `origin_group` siblings in `train` -> moved to `test`/`val`.
    - `manual_photos` (`train`, 3x augmented) -> split by `origin_group` (88% `train`, 12% `val`).
    - `roboflow_additional` (`train` negatives) -> split by `origin_group` (85% `train`, 10% `val`, 5% `test`).
    - `kaggle_train` -> stratified by `(negative, battery_removed)` (85% `train`, 15% `val`).
    - `open_images_negatives` (`source_split="validation_source"`) -> stratified by lookalike category
      (`eraser`, `pen`, `perfume`, `spice_rack`, `tin_can`, `torch`) into 80% `train`, 10% `val`, 10% `test`.
    """
    rng = random.Random(seed)
    assigned: dict[str, str] = {}

    def _img_keys(entry: dict[str, Any]) -> tuple[str, ...]:
        keys: list[str] = []
        for fname in ("image", "file_name", "filename", "image_path", "export_image"):
            val = entry.get(fname)
            if isinstance(val, str) and val:
                p = Path(val)
                keys.extend([val, p.name, p.stem])
        return tuple(dict.fromkeys(keys))

    def _set_split(entry: dict[str, Any], split_name: str) -> None:
        for k in _img_keys(entry):
            assigned[k] = split_name

    def _is_neg(entry: dict[str, Any]) -> bool:
        return bool(entry.get("negative", entry.get("is_negative", False)))

    # 1. If explicit `split` is already populated on rows, honor it first
    unassigned_entries: list[dict[str, Any]] = []
    for entry in manifest_entries:
        explicit = _canonical_split_name(entry.get("split"))
        if explicit is not None:
            _set_split(entry, explicit)
        else:
            unassigned_entries.append(entry)

    if not unassigned_entries:
        return assigned

    # Track `roboflow_primary` held-out origin_groups so burst/flip variants don't stay in `train`
    rf_primary_test_groups = {
        str(e.get("origin_group"))
        for e in unassigned_entries
        if e.get("source_id") == "roboflow_primary"
        and _canonical_split_name(e.get("source_split")) == "test"
        and e.get("origin_group")
    }
    rf_primary_val_groups = {
        str(e.get("origin_group"))
        for e in unassigned_entries
        if e.get("source_id") == "roboflow_primary"
        and _canonical_split_name(e.get("source_split")) == "val"
        and e.get("origin_group")
    }

    rf_add_neg_groups: dict[str, list[dict[str, Any]]] = {}
    manual_train_groups: dict[str, list[dict[str, Any]]] = {}
    kaggle_train_strata: dict[tuple[bool, bool], list[dict[str, Any]]] = {}
    open_images_cats: dict[str, list[dict[str, Any]]] = {}
    generic_unassigned: dict[str, list[dict[str, Any]]] = {}

    for entry in unassigned_entries:
        src = str(
            entry.get("source_id")
            or entry.get("source")
            or entry.get("dataset")
            or entry.get("source_dataset")
            or "default"
        )
        src_split = _canonical_split_name(entry.get("source_split"))
        neg = _is_neg(entry)
        ogroup = str(entry.get("origin_group") or _img_keys(entry)[0])

        if src == "kaggle_test_manual":
            _set_split(entry, "test")
        elif src == "roboflow_primary":
            if src_split == "test" or ogroup in rf_primary_test_groups:
                _set_split(entry, "test")
            elif src_split == "val" or ogroup in rf_primary_val_groups:
                _set_split(entry, "val")
            else:
                _set_split(entry, "train")
        elif src == "roboflow_additional":
            if src_split == "test":
                _set_split(entry, "test")
            elif src_split == "val":
                _set_split(entry, "val")
            elif neg:
                rf_add_neg_groups.setdefault(ogroup, []).append(entry)
            else:
                _set_split(entry, "train")
        elif src == "manual_photos":
            if src_split == "test":
                _set_split(entry, "test")
            elif src_split == "val":
                _set_split(entry, "val")
            else:
                manual_train_groups.setdefault(ogroup, []).append(entry)
        elif src == "kaggle_train":
            if src_split in ("val", "test"):
                _set_split(entry, src_split)
            else:
                removed = entry.get("removed_annotations") or {}
                has_bat = bool(isinstance(removed, dict) and removed.get("battery"))
                kaggle_train_strata.setdefault((neg, has_bat), []).append(entry)
        elif src == "open_images_negatives":
            src_img = str(entry.get("source_image") or "")
            cat = Path(src_img).parent.name if src_img else "default"
            open_images_cats.setdefault(cat, []).append(entry)
        else:
            if src_split in ("val", "test"):
                _set_split(entry, src_split)
            else:
                generic_unassigned.setdefault(f"{src}|neg={int(neg)}", []).append(entry)

    # Partition `roboflow_additional` train negatives by origin_group (85/10/5)
    if rf_add_neg_groups:
        gkeys = sorted(rf_add_neg_groups)
        rng.shuffle(gkeys)
        n_val = max(1, round(len(gkeys) * 0.10)) if len(gkeys) >= 4 else 0
        n_test = max(1, round(len(gkeys) * 0.05)) if len(gkeys) >= 6 else 0
        for idx, gk in enumerate(gkeys):
            target_sp = "val" if idx < n_val else ("test" if idx < n_val + n_test else "train")
            for e in rf_add_neg_groups[gk]:
                _set_split(e, target_sp)

    # Partition `manual_photos` train by origin_group (88% train / 12% val, stratified by neg)
    if manual_train_groups:
        for neg_flag in (False, True):
            sub_g = sorted(
                gk for gk, es in manual_train_groups.items() if _is_neg(es[0]) == neg_flag
            )
            if not sub_g:
                continue
            rng.shuffle(sub_g)
            n_val = max(1, round(len(sub_g) * 0.12)) if len(sub_g) >= 4 else 0
            for idx, gk in enumerate(sub_g):
                target_sp = "val" if idx < n_val else "train"
                for e in manual_train_groups[gk]:
                    _set_split(e, target_sp)

    # Partition `kaggle_train` (85% train / 15% val, stratified by (neg, battery_removed))
    for skey in sorted(kaggle_train_strata):
        bucket = sorted(kaggle_train_strata[skey], key=lambda x: _img_keys(x)[0])
        rng.shuffle(bucket)
        n_val = max(1, round(len(bucket) * 0.15)) if len(bucket) >= 4 else 0
        for idx, e in enumerate(bucket):
            _set_split(e, "val" if idx < n_val else "train")

    # Partition `open_images_negatives` by lookalike category (80% train / 10% val / 10% test)
    for cat in sorted(open_images_cats):
        bucket = sorted(open_images_cats[cat], key=lambda x: _img_keys(x)[0])
        rng.shuffle(bucket)
        n_val = max(1, round(len(bucket) * 0.10)) if len(bucket) >= 4 else 0
        n_test = max(1, round(len(bucket) * 0.10)) if len(bucket) >= 6 else 0
        for idx, e in enumerate(bucket):
            target_sp = "val" if idx < n_val else ("test" if idx < n_val + n_test else "train")
            _set_split(e, target_sp)

    # Partition any other generic unlisted/custom manifest sources (85/10/5)
    for skey in sorted(generic_unassigned):
        bucket = sorted(generic_unassigned[skey], key=lambda x: _img_keys(x)[0])
        rng.shuffle(bucket)
        n = len(bucket)
        n_val = max(1, round(n * 0.10)) if n >= 4 else 0
        n_test = round(n * 0.05)
        if n_val + n_test >= n:
            n_test = max(0, n - 1 - n_val)
        for idx, e in enumerate(bucket):
            target_sp = "val" if idx < n_val else ("test" if idx < n_val + n_test else "train")
            _set_split(e, target_sp)

    return assigned


def _load_manifest_info(
    export_root: Path,
    seed: int = 42,
) -> tuple[dict[str, str], dict[str, str], list[dict[str, Any]]]:
    """Load `(strata_by_key, split_by_key, raw_entries)` from `manifest.jsonl` if present."""
    manifest_path = export_root / "manifest.jsonl"
    strata: dict[str, str] = {}
    entries: list[dict[str, Any]] = []
    if not manifest_path.is_file():
        return strata, {}, entries

    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            entry = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        entries.append(entry)

        source = str(
            entry.get("source_id")
            or entry.get("source")
            or entry.get("dataset")
            or entry.get("source_dataset")
            or "default"
        )
        is_neg = bool(entry.get("negative", entry.get("is_negative", False)))
        key_stratum = f"{source}|neg={int(is_neg)}"

        for field_name in ("file_name", "filename", "image", "image_path", "export_image"):
            val = entry.get(field_name)
            if isinstance(val, str) and val:
                p = Path(val)
                strata[p.name] = key_stratum
                strata[p.stem] = key_stratum

    splits = assign_vape_manifest_splits(entries, seed=seed) if entries else {}
    return strata, splits, entries


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _is_empty_label(lbl_path: Path) -> bool:
    return (not lbl_path.is_file()) or (lbl_path.read_text(encoding="utf-8").strip() == "")


def _partition_items_stratified(
    items: list[tuple[Path, Path, str]],
    ratios: dict[str, float],
    seed: int = 42,
) -> dict[str, list[tuple[Path, Path]]]:
    """Deterministically partition `(img_path, lbl_path, stratum)` items across target splits."""
    target_splits = list(ratios.keys())
    grouped: dict[str, list[tuple[Path, Path]]] = {}
    for img_p, lbl_p, stratum in items:
        grouped.setdefault(stratum, []).append((img_p, lbl_p))

    rng = random.Random(seed)
    assigned: dict[str, list[tuple[Path, Path]]] = {sp: [] for sp in target_splits}

    for stratum_key in sorted(grouped):
        bucket = list(grouped[stratum_key])
        rng.shuffle(bucket)
        n = len(bucket)
        if len(target_splits) == 1:
            assigned[target_splits[0]].extend(bucket)
            continue

        # Allocate secondary splits first (val/test), remainder goes to primary split (train)
        primary = target_splits[0]
        secondary = target_splits[1:]
        sec_counts: dict[str, int] = {}
        used = 0
        for sp in secondary:
            cnt = round(n * ratios[sp])
            if sp == "val" and n >= 4 and ratios[sp] > 0 and cnt == 0:
                cnt = 1
            if used + cnt >= n:
                cnt = max(0, n - 1 - used)
            sec_counts[sp] = cnt
            used += cnt

        n_primary = max(0, n - used)
        assigned[primary].extend(bucket[:n_primary])
        cursor = n_primary
        for sp in secondary:
            cnt = sec_counts[sp]
            assigned[sp].extend(bucket[cursor : cursor + cnt])
            cursor += cnt

    return assigned


def normalize_yolo_dataset_splits(
    raw_root: Path,
    output_dataset_root: Path,
    default_ratios: tuple[float, float, float] = (0.85, 0.10, 0.05),
    seed: int = 42,
) -> dict[str, int]:
    """Normalize any flat, Roboflow, or Ultralytics dataset into `images/{train,val,test}`.

    - Preserves existing `train`, `val` (`valid`), and `test` assignments whenever present!
    - Deduplicates identical images by SHA-256 across splits (priority: `test` > `val` > `train`).
    - If `val` is missing (e.g. a Roboflow export with only `train/` populated), carves out a
      stratified `val` (and `test` if also missing) subset from `train` so YOLO training works.
    """
    train_r, val_r, test_r = default_ratios
    if train_r <= 0 or val_r <= 0 or test_r < 0 or abs((train_r + val_r + test_r) - 1.0) > 1e-5:
        raise ValueError(f"default_ratios must sum to 1.0 with train,val > 0, got {default_ratios}")

    ds_root = find_yolo_dataset_root(raw_root)
    out_root = Path(output_dataset_root)
    existing_splits = _detect_split_dirs(ds_root)
    manifest_strata, manifest_splits, manifest_entries = _load_manifest_info(ds_root, seed=seed)

    split_records: dict[str, list[tuple[Path, Path]]] = {sp: [] for sp in SPLITS}

    if existing_splits:
        if "train" not in existing_splits:
            raise ValueError(
                f"Dataset at {ds_root} has split dirs {list(existing_splits)} but no train/"
            )

        for sp, (img_dir, lbl_dir) in existing_splits.items():
            for img_p in _list_images(img_dir):
                lbl_p = lbl_dir / f"{img_p.stem}.txt"
                split_records[sp].append((img_p, lbl_p))
    else:
        # Flat `images/` + `labels/` layout (e.g. `public_data/exports/vape_yolo/`)
        images_dir = ds_root / "images"
        labels_dir = ds_root / "labels"
        flat_imgs = _list_images(images_dir)
        if not flat_imgs:
            raise FileNotFoundError(f"No images found in {images_dir}")

        unassigned: list[tuple[Path, Path, str]] = []
        for img_p in flat_imgs:
            lbl_p = labels_dir / f"{img_p.stem}.txt"
            assigned_sp = manifest_splits.get(img_p.name) or manifest_splits.get(img_p.stem)
            if assigned_sp is not None:
                split_records[assigned_sp].append((img_p, lbl_p))
            else:
                stratum = manifest_strata.get(img_p.name) or manifest_strata.get(
                    img_p.stem, f"unlisted|neg={int(_is_empty_label(lbl_p))}"
                )
                unassigned.append((img_p, lbl_p, stratum))

        if unassigned:
            part = _partition_items_stratified(
                unassigned,
                {"train": train_r, "val": val_r, "test": test_r},
                seed=seed,
            )
            for sp in SPLITS:
                split_records[sp].extend(part[sp])

    # If `val` is empty (e.g. train-only export), carve `val` (and `test` if empty) from `train`
    if not split_records["val"]:
        if not split_records["train"]:
            raise ValueError(f"Dataset at {ds_root} produced no train or val samples.")
        tagged_train = [
            (
                img_p,
                lbl_p,
                manifest_strata.get(img_p.name)
                or manifest_strata.get(img_p.stem, f"neg={int(_is_empty_label(lbl_p))}"),
            )
            for img_p, lbl_p in split_records["train"]
        ]
        if split_records["test"]:
            total_tv = train_r + val_r
            carved = _partition_items_stratified(
                tagged_train,
                {"train": train_r / total_tv, "val": val_r / total_tv},
                seed=seed,
            )
            split_records["train"] = carved["train"]
            split_records["val"] = carved["val"]
        else:
            split_records = _partition_items_stratified(
                tagged_train,
                {"train": train_r, "val": val_r, "test": test_r},
                seed=seed,
            )

    for sp in SPLITS:
        (out_root / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / sp).mkdir(parents=True, exist_ok=True)

    # Deduplicate exact SHA-256 matches across splits (giving priority to test > val > train)
    seen_hashes: set[str] = set()
    final_counts: dict[str, int] = {sp: 0 for sp in SPLITS}
    final_split_by_name: dict[str, str] = {}

    for sp in ("test", "val", "train"):
        for img_p, lbl_p in split_records[sp]:
            digest = hashlib.sha256(img_p.read_bytes()).hexdigest()
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            final_split_by_name[img_p.name] = sp

            dst_img = out_root / "images" / sp / img_p.name
            _link_or_copy(img_p, dst_img)

            dst_lbl = out_root / "labels" / sp / f"{img_p.stem}.txt"
            if lbl_p.is_file():
                _link_or_copy(lbl_p, dst_lbl)
            else:
                dst_lbl.write_text("", encoding="utf-8")
            final_counts[sp] += 1

    # Write updated `manifest.jsonl` with assigned `split` populated (and exact-dup rows omitted)
    if manifest_entries:
        out_lines: list[str] = []
        for entry in manifest_entries:
            img_field = str(entry.get("image") or entry.get("filename") or entry.get("file_name") or "")
            img_name = Path(img_field).name
            assigned_sp = final_split_by_name.get(img_name)
            if assigned_sp is None:
                continue
            updated = dict(entry)
            updated["split"] = assigned_sp
            out_lines.append(json.dumps(updated, sort_keys=True))
        (out_root / "manifest.jsonl").write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    return final_counts


def _parallel_copy_tree(src_dir: Path, dst_dir: Path, max_workers: int = 16) -> int:
    """Copy thousands of small files off Google Drive FUSE quickly with progress logging."""
    files = [p for p in src_dir.rglob("*") if p.is_file()]
    total = len(files)
    for parent in {dst_dir / p.relative_to(src_dir).parent for p in files}:
        parent.mkdir(parents=True, exist_ok=True)

    def _copy_one(src_file: Path) -> None:
        rel = src_file.relative_to(src_dir)
        shutil.copyfile(src_file, dst_dir / rel)

    copied = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for _ in pool.map(_copy_one, files):
            copied += 1
            if copied % 1000 == 0 or copied == total:
                print(f"  [Drive -> Local SSD] Copied {copied}/{total} files...")
    return copied


def _safe_cache_slug(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", Path(name).stem).strip("_")
    return cleaned or "dataset"


def _select_gpu_train_params() -> dict[str, Any]:
    """Pick optimal batch size, workers, and caching based on the Colab GPU tier."""
    import torch

    if not torch.cuda.is_available():
        return {"device": "cpu", "batch": 8, "workers": 2, "cache": False}

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Detected CUDA GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")

    if vram_gb >= 35.0:  # A100 40GB / 80GB
        return {"device": 0, "batch": 64, "workers": 8, "cache": "ram"}
    if vram_gb >= 20.0:  # L4 24GB
        return {"device": 0, "batch": 48, "workers": 8, "cache": "disk"}
    return {"device": 0, "batch": 32, "workers": 4, "cache": False}  # T4 16GB


def prepare_dataset_from_drive_source(
    drive_dir: Path,
    work_dir: Path,
    source_name: str | None = None,
    sam_weights: str = "sam2.1_t.pt",
    sam_model: Any | None = None,
) -> Path:
    """Prepare one zip/folder from `drive_dir` into a validated polygon YOLO-seg dataset."""
    drive_root = Path(drive_dir)
    if not drive_root.exists():
        raise FileNotFoundError(f"Drive directory not found: {drive_root}")

    work_root = Path(work_dir)
    work_root.mkdir(parents=True, exist_ok=True)

    # Resolve source folder or zip inside drive_root (prefer `vape_yolo` directory if present!)
    chosen_source: Path | None = None
    if source_name is not None:
        candidate = drive_root / source_name
        if not candidate.exists():
            # Search recursively in case e.g. `public_data/exports/vape_yolo` is nested
            matches = [p for p in drive_root.rglob(source_name) if "__MACOSX" not in p.parts]
            if not matches:
                raise FileNotFoundError(f"Dataset source '{source_name}' not found in {drive_root}")
            candidate = min(matches, key=lambda p: len(p.parts))
        chosen_source = candidate
    else:
        vape_yolo_dirs = [
            p for p in drive_root.rglob("vape_yolo") if p.is_dir() and "__MACOSX" not in p.parts
        ]
        if vape_yolo_dirs:
            chosen_source = min(vape_yolo_dirs, key=lambda p: len(p.parts))
        else:
            try:
                chosen_source = find_yolo_dataset_root(drive_root)
            except FileNotFoundError:
                non_cache_zips = sorted(
                    p
                    for p in drive_root.glob("*.zip")
                    if p.is_file() and not p.name.endswith("_seg_cache.zip")
                )
                if not non_cache_zips:
                    raise
                chosen_source = max(non_cache_zips, key=lambda p: p.stat().st_size)

    slug = _safe_cache_slug(chosen_source.name)
    cache_zip = drive_root / f"{slug}_seg_cache.zip"
    prepared_dir = work_root / f"{slug}_seg"

    if cache_zip.is_file() and not prepared_dir.exists():
        print(f"[{slug}] Restoring cached SAM-2 polygon dataset from {cache_zip} ...")
        prepared_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(cache_zip, "r") as zf:
            zf.extractall(prepared_dir)
    elif not (prepared_dir / "images" / "train").is_dir():
        unpack_dir = work_root / f"{slug}_raw"
        shutil.rmtree(unpack_dir, ignore_errors=True)
        unpack_dir.mkdir(parents=True, exist_ok=True)

        if chosen_source.is_file() and chosen_source.suffix.lower() == ".zip":
            local_zip = work_root / chosen_source.name
            print(f"[{slug}] Copying {chosen_source.name} to local SSD ...")
            shutil.copy2(chosen_source, local_zip)
            with zipfile.ZipFile(local_zip, "r") as zf:
                zf.extractall(unpack_dir)
            local_zip.unlink(missing_ok=True)
        else:
            src_ds = find_yolo_dataset_root(chosen_source)
            print(f"[{slug}] Parallel-copying unzipped dataset {src_ds} to local SSD ...")
            _parallel_copy_tree(src_ds, unpack_dir / "copied")

        split_counts = normalize_yolo_dataset_splits(unpack_dir, prepared_dir)
        print(f"[{slug}] Normalized splits (existing splits preserved): {split_counts}")
        shutil.rmtree(unpack_dir, ignore_errors=True)

        for sp in SPLITS:
            img_dir = prepared_dir / "images" / sp
            lbl_dir = prepared_dir / "labels" / sp
            if _list_images(img_dir):
                print(f"[{slug}:{sp}] Upgrading any 4-value bboxes to SAM 2 ({sam_weights}) polygons...")
                stats = upgrade_bbox_labels_with_sam(
                    images_dir=img_dir,
                    labels_dir=lbl_dir,
                    output_labels_dir=lbl_dir,
                    sam_weights=sam_weights,
                    sam_model=sam_model,
                )
                print(f"[{slug}:{sp}] SAM 2 stats: {stats}")

        print(f"[{slug}] Saving polygon-ready cache back to Drive: {cache_zip} ...")
        tmp_archive_base = work_root / f"{slug}_cache_tmp"
        archive_path = Path(shutil.make_archive(str(tmp_archive_base), "zip", root_dir=prepared_dir))
        shutil.copy2(archive_path, cache_zip)
        archive_path.unlink(missing_ok=True)

    yaml_path = write_yolo_dataset_yaml(prepared_dir, ["vape"])
    # If `test` split is empty (e.g. dataset only had pre-assigned `train` + `val`), point `test:` to `val`
    if not _list_images(prepared_dir / "images" / "test"):
        txt = yaml_path.read_text(encoding="utf-8").replace("test: images/test", "test: images/val")
        yaml_path.write_text(txt, encoding="utf-8")

    report = validate_yolo_seg_dataset(prepared_dir, num_classes=1)
    print(
        f"[{slug}] Validation: valid={report.is_valid} | splits={report.split_counts} | "
        f"polygons={report.polygon_count} | near_duplicates={len(report.near_duplicates)}"
    )
    if not report.is_valid:
        raise RuntimeError(f"[{slug}] Dataset validation failed: {report.errors[:5]}")

    return yaml_path


def prepare_and_train_from_drive(
    drive_dir: Path,
    work_dir: Path = Path("/content/datasets"),
    zip_name: str | None = None,
    stage2_zip: str | None = None,
    sam_weights: str = "sam2.1_t.pt",
    epochs: int = 50,
    stage2_epochs: int = 25,
    imgsz: int = 640,
    sam_model: Any | None = None,
    skip_train: bool = False,
) -> Path:
    """Prepare Stage 1 (and optional Stage 2) datasets from Drive and train `yolo11s-seg`."""
    drive_root = Path(drive_dir)
    stage1_yaml = prepare_dataset_from_drive_source(
        drive_dir=drive_root,
        work_dir=work_dir,
        source_name=zip_name,
        sam_weights=sam_weights,
        sam_model=sam_model,
    )

    stage2_yaml: Path | None = None
    if stage2_zip is not None:
        stage2_yaml = prepare_dataset_from_drive_source(
            drive_dir=drive_root,
            work_dir=work_dir,
            source_name=stage2_zip,
            sam_weights=sam_weights,
            sam_model=sam_model,
        )

    if skip_train:
        return stage2_yaml or stage1_yaml

    from ultralytics import YOLO

    gpu_cfg = _select_gpu_train_params()
    runs_project = drive_root / "runs_vape_seg"

    stage1_model = YOLO("yolo11s-seg.pt")
    stage1_results = stage1_model.train(
        data=str(stage1_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=gpu_cfg["batch"],
        workers=gpu_cfg["workers"],
        cache=gpu_cfg["cache"],
        device=gpu_cfg["device"],
        seed=0,
        patience=15,
        save_period=5,
        degrees=180.0,
        flipud=0.5,
        fliplr=0.5,
        hsv_h=0.4,
        hsv_s=0.6,
        hsv_v=0.4,
        copy_paste=0.2,
        project=str(runs_project),
        name="stage1_pretrain",
        exist_ok=True,
    )
    best_pt = (
        Path(stage1_model.trainer.best)
        if getattr(stage1_model, "trainer", None) is not None
        and getattr(stage1_model.trainer, "best", None)
        else Path(stage1_results.save_dir) / "weights" / "best.pt"
    )
    shutil.copy2(best_pt, drive_root / "vape_stage1_yolo11s_seg_best.pt")
    print(f"Backed up Stage 1 checkpoint -> {drive_root / 'vape_stage1_yolo11s_seg_best.pt'}")

    if stage2_yaml is not None:
        print(f"Starting Stage 2 fine-tuning on {stage2_yaml} ...")
        stage2_model = YOLO(str(best_pt))
        stage2_results = stage2_model.train(
            data=str(stage2_yaml),
            epochs=stage2_epochs,
            optimizer="AdamW",
            lr0=5e-4,
            lrf=0.05,
            patience=15,
            freeze=10,
            seed=0,
            imgsz=imgsz,
            batch=min(16, int(gpu_cfg["batch"])),
            workers=gpu_cfg["workers"],
            device=gpu_cfg["device"],
            degrees=180.0,
            flipud=0.5,
            fliplr=0.5,
            hsv_h=0.4,
            copy_paste=0.2,
            mosaic=0.5,
            close_mosaic=10,
            project=str(runs_project),
            name="stage2_tabletop_finetune",
            exist_ok=True,
        )
        best_pt = (
            Path(stage2_model.trainer.best)
            if getattr(stage2_model, "trainer", None) is not None
            and getattr(stage2_model.trainer, "best", None)
            else Path(stage2_results.save_dir) / "weights" / "best.pt"
        )
        shutil.copy2(best_pt, drive_root / "vape_stage2_yolo11s_seg_best.pt")
        print(f"Backed up Stage 2 checkpoint -> {drive_root / 'vape_stage2_yolo11s_seg_best.pt'}")

    eval_yaml = stage2_yaml or stage1_yaml
    print(f"Evaluating best checkpoint ({best_pt}) on held-out test split ({eval_yaml}) ...")
    YOLO(str(best_pt)).val(
        data=str(eval_yaml),
        split="test",
        imgsz=imgsz,
        device=gpu_cfg["device"],
        project=str(runs_project),
        name="heldout_test_eval",
        exist_ok=True,
    )

    return best_pt


def build_parser() -> argparse.ArgumentParser:
    """CLI parser for running the Google Drive -> Colab GPU training workflow."""
    parser = argparse.ArgumentParser(
        description="Normalize YOLO datasets (flat, Roboflow, or pre-split), run SAM 2, & train."
    )
    parser.add_argument(
        "--drive-dir",
        type=Path,
        default=Path("/content/drive/MyDrive/HACKLEUK"),
        help="Google Drive directory containing dataset .zip(s) or folder.",
    )
    parser.add_argument(
        "--source",
        "--zip-name",
        dest="zip_name",
        type=str,
        default=None,
        help="Dataset folder or .zip name inside --drive-dir (auto-prefers 'vape_yolo' folder).",
    )
    parser.add_argument(
        "--stage2-zip",
        type=str,
        default=None,
        help="Optional Stage 2 tabletop .zip/folder (e.g. 'puff pile.v1i.yolo26.zip') to fine-tune.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/content/datasets"),
        help="Fast local SSD working directory on Colab (default: /content/datasets).",
    )
    parser.add_argument(
        "--sam-weights",
        type=str,
        default="sam2.1_t.pt",
        help="SAM 2 checkpoint for bbox->polygon upgrade (sam2.1_t.pt or sam2.1_b.pt).",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Stage 1 training epochs.")
    parser.add_argument("--stage2-epochs", type=int, default=25, help="Stage 2 fine-tune epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO training image size.")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only normalize splits, run SAM 2, validate, and cache to Drive without training.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for `python -m leukerbad.dataset.colab_train`."""
    args = build_parser().parse_args(argv)
    prepare_and_train_from_drive(
        drive_dir=args.drive_dir,
        work_dir=args.work_dir,
        zip_name=args.zip_name,
        stage2_zip=args.stage2_zip,
        sam_weights=args.sam_weights,
        epochs=args.epochs,
        stage2_epochs=args.stage2_epochs,
        imgsz=args.imgsz,
        skip_train=args.prepare_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
