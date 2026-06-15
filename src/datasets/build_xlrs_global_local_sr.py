"""Build input-budget-aware global/local SR images for XLRS evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageStat

from src.datasets.jsonl import read_jsonl, write_jsonl

_RESAMPLE = Image.Resampling.LANCZOS
_IMAGE_SUFFIX = ".jpg"
_SAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class BuildConfig:
    input_path: Path = Path("data/processed/xlrs_eval.jsonl")
    output_dir: Path = Path("data/processed")
    path_base: Path = Path(".")
    sr_image_dir: Path = Path("data/raw/xlrs_bench/images_sr_lanczos_2x_capped")
    paired_image_dir: Path = Path(
        "data/raw/xlrs_bench/images_paired_global_local_lanczos_2x"
    )
    manifest_dir: Path = Path("data/raw/xlrs_bench/tile_manifests_lanczos_2x")
    sr_scale: int = 2
    max_side: int = 2048
    global_panel_size: int = 1024
    tile_panel_size: int = 512
    grid_size: int = 4
    top_k_tiles: int = 4
    jpeg_quality: int = 90
    limit: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "sr_scale",
            "max_side",
            "global_panel_size",
            "tile_panel_size",
            "grid_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.tile_panel_size > self.global_panel_size * 2:
            raise ValueError("tile_panel_size must fit within the contact sheet width")
        if self.top_k_tiles < 0:
            raise ValueError("top_k_tiles must be non-negative")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        if self.limit is not None and self.limit < 0:
            raise ValueError("limit must be non-negative")


def build_xlrs_global_local_sr(config: BuildConfig) -> None:
    """Generate original, SR, and paired XLRS JSONL files plus derived images."""

    original_rows: list[dict[str, Any]] = []
    sr_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    records = list(read_jsonl(config.input_path))
    if config.limit is not None:
        records = records[: config.limit]

    _ensure_output_dirs(config)
    for record in records:
        built = build_xlrs_global_local_sr_record(record, config)
        original_rows.append(built["original"])
        sr_rows.append(built["sr"])
        paired_rows.append(built["paired"])

    write_jsonl(config.output_dir / "xlrs_eval_original.jsonl", original_rows)
    write_jsonl(config.output_dir / "xlrs_eval_sr.jsonl", sr_rows)
    write_jsonl(config.output_dir / "xlrs_eval_paired.jsonl", paired_rows)


def build_xlrs_global_local_sr_record(
    record: dict[str, Any], config: BuildConfig
) -> dict[str, dict[str, Any]]:
    """Generate derived images and rows for one XLRS EvalSample record."""

    sample_id = str(record["sample_id"])
    global_path = _resolve_path(record["image_path"], config.path_base)
    original_path = _resolve_path(
        record.get("metadata", {}).get("original_image_path", record["image_path"]),
        config.path_base,
    )
    with Image.open(original_path) as original_image:
        original_rgb = original_image.convert("RGB")
    with Image.open(global_path) as global_image:
        global_rgb = global_image.convert("RGB")

    output_stem = _safe_output_stem(sample_id)
    output_name = f"{output_stem}{_IMAGE_SUFFIX}"
    sr_path = config.sr_image_dir / output_name
    paired_path = config.paired_image_dir / output_name
    manifest_path = config.manifest_dir / f"{output_stem}.json"

    sr_full = _make_budget_capped_sr(original_rgb, config)
    _save_jpeg(sr_full, sr_path, config.jpeg_quality)

    tiles = _select_detail_tiles(original_rgb, config.grid_size, config.top_k_tiles)
    tile_images = [_make_tile_panel(original_rgb, tile, config) for tile in tiles]
    paired = _make_contact_sheet(global_rgb, sr_full, tile_images, tiles, config)
    _save_jpeg(paired, paired_path, config.jpeg_quality)
    _write_manifest(
        manifest_path, record, original_path, sr_path, paired_path, tiles, config
    )

    original_row = _variant_row(
        record,
        variant="original",
        image_path=global_path,
        original_path=original_path,
        sr_path=None,
        paired_path=None,
        manifest_path=None,
        config=config,
    )
    sr_row = _variant_row(
        record,
        variant="sr",
        image_path=sr_path,
        original_path=original_path,
        sr_path=sr_path,
        paired_path=None,
        manifest_path=manifest_path,
        config=config,
    )
    paired_row = _variant_row(
        record,
        variant="paired",
        image_path=paired_path,
        original_path=original_path,
        sr_path=sr_path,
        paired_path=paired_path,
        manifest_path=manifest_path,
        config=config,
    )
    return {"original": original_row, "sr": sr_row, "paired": paired_row}


def _make_budget_capped_sr(image: Image.Image, config: BuildConfig) -> Image.Image:
    sr_size = (image.width * config.sr_scale, image.height * config.sr_scale)
    sr = image.resize(sr_size, _RESAMPLE)
    return _cap_max_side(sr, config.max_side)


def _select_detail_tiles(
    image: Image.Image, grid_size: int, top_k: int
) -> list[dict[str, Any]]:
    tiles: list[dict[str, Any]] = []
    cell_width = image.width / grid_size
    cell_height = image.height / grid_size
    for row in range(grid_size):
        for col in range(grid_size):
            x1 = int(round(col * cell_width))
            y1 = int(round(row * cell_height))
            x2 = int(round((col + 1) * cell_width))
            y2 = int(round((row + 1) * cell_height))
            crop = image.crop((x1, y1, x2, y2))
            score = _detail_score(crop)
            tiles.append(
                {
                    "tile_id": f"tile_r{row}c{col}",
                    "row": row,
                    "col": col,
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "detail_score": round(score, 4),
                    "source": "original_image_lanczos_sr_crop",
                }
            )
    return sorted(tiles, key=lambda tile: tile["detail_score"], reverse=True)[:top_k]


def _detail_score(image: Image.Image) -> float:
    gray = ImageOps.grayscale(image)
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_stat = ImageStat.Stat(edges)
    gray_stat = ImageStat.Stat(gray)
    return float(edge_stat.mean[0] + gray_stat.stddev[0])


def _make_tile_panel(
    image: Image.Image, tile: dict[str, Any], config: BuildConfig
) -> Image.Image:
    crop = image.crop(tuple(tile["bbox_xyxy"]))
    sr_size = (crop.width * config.sr_scale, crop.height * config.sr_scale)
    sr_crop = crop.resize(sr_size, _RESAMPLE)
    return ImageOps.contain(
        sr_crop, (config.tile_panel_size, config.tile_panel_size), method=_RESAMPLE
    )


def _make_contact_sheet(
    global_image: Image.Image,
    sr_image: Image.Image,
    tile_images: list[Image.Image],
    tiles: list[dict[str, Any]],
    config: BuildConfig,
) -> Image.Image:
    width = config.global_panel_size * 2
    tiles_per_row = max(1, width // config.tile_panel_size)
    tile_rows = (len(tile_images) + tiles_per_row - 1) // tiles_per_row
    height = config.global_panel_size + tile_rows * config.tile_panel_size
    canvas = Image.new("RGB", (width, height), "white")
    global_panel = ImageOps.contain(
        global_image,
        (config.global_panel_size, config.global_panel_size),
        method=_RESAMPLE,
    )
    sr_panel = ImageOps.contain(
        sr_image, (config.global_panel_size, config.global_panel_size), method=_RESAMPLE
    )
    _paste_centered(
        canvas, global_panel, (0, 0, config.global_panel_size, config.global_panel_size)
    )
    _paste_centered(
        canvas,
        sr_panel,
        (config.global_panel_size, 0, width, config.global_panel_size),
    )
    _draw_label(canvas, (0, 0), "GLOBAL ORIGINAL")
    _draw_label(canvas, (config.global_panel_size, 0), "GLOBAL SR-LANCZOS")

    for index, tile_image in enumerate(tile_images):
        row, col = divmod(index, tiles_per_row)
        x = col * config.tile_panel_size
        y = config.global_panel_size + row * config.tile_panel_size
        panel = (x, y, x + config.tile_panel_size, y + config.tile_panel_size)
        _paste_centered(canvas, tile_image, panel)
        _draw_label(canvas, (x, y), tiles[index]["tile_id"].upper())
    return canvas


def _paste_centered(
    canvas: Image.Image, image: Image.Image, box: tuple[int, int, int, int]
) -> None:
    x1, y1, x2, y2 = box
    x = x1 + ((x2 - x1) - image.width) // 2
    y = y1 + ((y2 - y1) - image.height) // 2
    canvas.paste(image, (x, y))


def _draw_label(canvas: Image.Image, xy: tuple[int, int], text: str) -> None:
    draw = ImageDraw.Draw(canvas)
    x, y = xy
    bbox = draw.textbbox((x + 8, y + 8), text)
    draw.rectangle((bbox[0] - 4, bbox[1] - 3, bbox[2] + 4, bbox[3] + 3), fill="black")
    draw.text((x + 8, y + 8), text, fill="white")


def _variant_row(
    record: dict[str, Any],
    *,
    variant: str,
    image_path: Path,
    original_path: Path,
    sr_path: Path | None,
    paired_path: Path | None,
    manifest_path: Path | None,
    config: BuildConfig,
) -> dict[str, Any]:
    row = dict(record)
    row["sample_id"] = f"{record['sample_id']}_{variant}"
    row["image_path"] = _display_path(image_path)
    metadata = dict(record.get("metadata", {}))
    metadata.update(
        {
            "xlrs_sr_variant": variant,
            "evidence_protocol": "input_budget_aware_global_local_sr",
            "original_image_path": _display_path(original_path),
            "sr_method": "lanczos",
            "sr_scale": config.sr_scale,
            "input_budget_capped": True,
            "derived_image_max_side": config.max_side,
            "global_panel_size": config.global_panel_size,
            "tile_panel_size": config.tile_panel_size,
            "top_k_tiles": config.top_k_tiles,
            "tile_grid_size": config.grid_size,
        }
    )
    if sr_path is not None:
        metadata["sr_image_path"] = _display_path(sr_path)
    if paired_path is not None:
        metadata["paired_image_path"] = _display_path(paired_path)
    if manifest_path is not None:
        metadata["tile_manifest_path"] = _display_path(manifest_path)
    row["metadata"] = metadata
    return row


def _write_manifest(
    path: Path,
    record: dict[str, Any],
    original_path: Path,
    sr_path: Path,
    paired_path: Path,
    tiles: list[dict[str, Any]],
    config: BuildConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "sample_id": record["sample_id"],
        "source_image_path": _display_path(original_path),
        "sr_image_path": _display_path(sr_path),
        "paired_image_path": _display_path(paired_path),
        "sr_method": "lanczos",
        "sr_scale": config.sr_scale,
        "selection_method": "top_k_grid_detail_score",
        "grid_size": config.grid_size,
        "top_k_tiles": config.top_k_tiles,
        "tiles": tiles,
    }
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _cap_max_side(image: Image.Image, max_side: int) -> Image.Image:
    if max(image.size) <= max_side:
        return image
    return ImageOps.contain(image, (max_side, max_side), method=_RESAMPLE)


def _save_jpeg(image: Image.Image, path: Path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=quality, optimize=True)


def _ensure_output_dirs(config: BuildConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.sr_image_dir.mkdir(parents=True, exist_ok=True)
    config.paired_image_dir.mkdir(parents=True, exist_ok=True)
    config.manifest_dir.mkdir(parents=True, exist_ok=True)


def _safe_output_stem(value: str) -> str:
    stem = _SAFE_FILENAME_PATTERN.sub("_", value).strip("._")
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    if not stem:
        stem = "sample"
    return f"{stem[:80]}_{digest}"


def _resolve_path(value: Any, path_base: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return path_base / path


def _display_path(path: Path) -> str:
    return path.as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build XLRS input-budget-aware global/local SR images and JSONL files."
    )
    parser.add_argument("--input-path", default="data/processed/xlrs_eval.jsonl")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--path-base", default=".")
    parser.add_argument(
        "--sr-image-dir", default="data/raw/xlrs_bench/images_sr_lanczos_2x_capped"
    )
    parser.add_argument(
        "--paired-image-dir",
        default="data/raw/xlrs_bench/images_paired_global_local_lanczos_2x",
    )
    parser.add_argument(
        "--manifest-dir", default="data/raw/xlrs_bench/tile_manifests_lanczos_2x"
    )
    parser.add_argument("--sr-scale", type=int, default=2)
    parser.add_argument("--max-side", type=int, default=2048)
    parser.add_argument("--global-panel-size", type=int, default=1024)
    parser.add_argument("--tile-panel-size", type=int, default=512)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--top-k-tiles", type=int, default=4)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    build_xlrs_global_local_sr(
        BuildConfig(
            input_path=Path(args.input_path),
            output_dir=Path(args.output_dir),
            path_base=Path(args.path_base),
            sr_image_dir=Path(args.sr_image_dir),
            paired_image_dir=Path(args.paired_image_dir),
            manifest_dir=Path(args.manifest_dir),
            sr_scale=args.sr_scale,
            max_side=args.max_side,
            global_panel_size=args.global_panel_size,
            tile_panel_size=args.tile_panel_size,
            grid_size=args.grid_size,
            top_k_tiles=args.top_k_tiles,
            jpeg_quality=args.jpeg_quality,
            limit=args.limit,
        )
    )


if __name__ == "__main__":
    main()
