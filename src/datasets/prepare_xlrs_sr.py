"""Prepare XLRS original/SR/paired evaluation files without changing the eval schema."""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal

from src.datasets.convert_xlrs import convert_xlrs_record
from src.datasets.jsonl import read_jsonl, write_jsonl
from src.datasets.schema import EvalSample

Variant = Literal["original", "sr", "paired"]

_VARIANTS: tuple[Variant, ...] = ("original", "sr", "paired")
_GENERIC_IMAGE_KEYS = ("image", "image_path", "img", "file_name")
_ORIGINAL_IMAGE_KEYS = (
    "original_image_path",
    "original_image",
    "lr_image_path",
    "lr_image",
)
_SR_IMAGE_KEYS = (
    "sr_image_path",
    "super_res_image_path",
    "sr_image",
    "hr_image_path",
    "hr_image",
)
_PAIRED_IMAGE_KEYS = (
    "paired_image_path",
    "contact_sheet_path",
    "comparison_image_path",
)
_SR_METADATA_KEYS = (
    "original_image_path",
    "sr_image_path",
    "sr_scale",
    "original_resolution",
    "sr_resolution",
    "quality_tier",
    "pair_id",
    "tile_manifest_path",
    "sr_method",
    "xlrs_sr_variant",
)


def prepare_xlrs_sr_files(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    image_root: str | Path = "",
) -> None:
    """Write original-only, SR-only, and paired XLRS eval JSONL files."""

    records = list(read_jsonl(input_path))
    output_root = Path(output_dir)
    for variant in _VARIANTS:
        rows = (
            prepare_xlrs_sr_record(
                record, variant=variant, image_root=image_root
            ).to_dict()
            for record in records
        )
        write_jsonl(output_root / f"xlrs_eval_{variant}.jsonl", rows)


def prepare_xlrs_sr_record(
    record: dict[str, Any],
    *,
    variant: Variant,
    image_root: str | Path = "",
) -> EvalSample:
    """Map one XLRS-like record to a variant-specific EvalSample."""

    enriched = dict(record)
    original_image = _first_present(record, _ORIGINAL_IMAGE_KEYS, None)
    sr_image = _first_present(record, _SR_IMAGE_KEYS, None)
    paired_image = _first_present(record, _PAIRED_IMAGE_KEYS, None)

    resolved_original = _resolve_optional_image(original_image, image_root)
    resolved_sr = _resolve_optional_image(sr_image, image_root)
    resolved_paired = _resolve_optional_image(paired_image, image_root)

    if resolved_original is not None:
        enriched["original_image_path"] = resolved_original
    if resolved_sr is not None:
        enriched["sr_image_path"] = resolved_sr
    if resolved_paired is not None:
        enriched["paired_image_path"] = resolved_paired

    image_for_variant = _select_variant_image(
        variant,
        original_image=resolved_original,
        sr_image=resolved_sr,
        paired_image=resolved_paired,
        fallback=_first_present(record, _GENERIC_IMAGE_KEYS, ""),
        image_root=image_root,
    )
    for key in _GENERIC_IMAGE_KEYS:
        enriched[key] = image_for_variant
    enriched["xlrs_sr_variant"] = variant

    sample = convert_xlrs_record(enriched)
    metadata = dict(sample.metadata)
    metadata.update(_sr_metadata(enriched, variant))
    return sample.__class__(
        sample_id=f"{sample.sample_id}_{variant}",
        dataset=sample.dataset,
        task_type=sample.task_type,
        image_path=sample.image_path,
        question=sample.question,
        reference_answer=sample.reference_answer,
        choices=sample.choices,
        metadata=metadata,
        taxonomy_hint=sample.taxonomy_hint,
    )


def _select_variant_image(
    variant: Variant,
    *,
    original_image: str | None,
    sr_image: str | None,
    paired_image: str | None,
    fallback: Any,
    image_root: str | Path,
) -> str:
    fallback_path = _resolve_optional_image(fallback, image_root) or str(fallback)
    if variant == "original":
        return original_image or fallback_path
    if variant == "sr":
        if sr_image is None:
            raise ValueError("SR variant requires sr_image_path or equivalent field")
        return sr_image
    if paired_image is None:
        raise ValueError(
            "Paired variant requires paired_image_path or contact sheet field"
        )
    return paired_image


def _sr_metadata(record: dict[str, Any], variant: Variant) -> dict[str, Any]:
    metadata = {key: record[key] for key in _SR_METADATA_KEYS if key in record}
    metadata["xlrs_sr_variant"] = variant
    metadata["evidence_protocol"] = "sr_aware_multiscale"
    if "paired_image_path" in record:
        metadata["paired_image_path"] = record["paired_image_path"]
    return metadata


def _resolve_optional_image(value: Any, image_root: str | Path) -> str | None:
    if value in (None, ""):
        return None
    text = str(value)
    path = Path(text)
    if _is_absolute_path(text) or not image_root:
        return str(path)
    return str(Path(image_root) / path)


def _is_absolute_path(value: str) -> bool:
    return (
        Path(value).is_absolute()
        or PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
    )


def _first_present(record: dict[str, Any], keys: tuple[str, ...], default: Any) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return default


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare XLRS original/SR/paired EvalSample JSONL files."
    )
    parser.add_argument("input_path")
    parser.add_argument("output_dir")
    parser.add_argument("--image-root", default="")
    args = parser.parse_args()
    prepare_xlrs_sr_files(args.input_path, args.output_dir, image_root=args.image_root)


if __name__ == "__main__":
    main()
