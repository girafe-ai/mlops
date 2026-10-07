"""Build tabular metadata from Cityscapes fine polygon annotations."""

import csv
import json
from pathlib import Path


def build_metadata(
    annotations_root: Path,
    images_root: Path,
    output_dir: Path,
) -> dict[str, int]:
    """Extract image and object records from ``*_gtFine_polygons.json`` files.

    Args:
        annotations_root: Root of the extracted ``gtFine`` directory.
        images_root: Expected root of extracted ``leftImg8bit`` images.
        output_dir: Directory for the generated ``images.csv`` and ``objects.csv``.

    Returns:
        Counts of written image and object records.

    Raises:
        FileNotFoundError: If the annotation directory is absent.
    """
    if not annotations_root.is_dir():
        raise FileNotFoundError(
            f"Annotation directory does not exist: {annotations_root}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    images_path = output_dir / "images.csv"
    objects_path = output_dir / "objects.csv"
    image_count = 0
    object_count = 0

    with (
        images_path.open("w", encoding="utf-8", newline="") as images_file,
        objects_path.open("w", encoding="utf-8", newline="") as objects_file,
    ):
        image_writer = csv.DictWriter(
            images_file,
            fieldnames=["image_id", "split", "city", "image_path", "annotation_path"],
        )
        object_writer = csv.DictWriter(
            objects_file,
            fieldnames=["image_id", "object_index", "label_name", "polygon"],
        )
        image_writer.writeheader()
        object_writer.writeheader()

        for annotation_path in sorted(annotations_root.rglob("*_gtFine_polygons.json")):
            relative_path = annotation_path.relative_to(annotations_root)
            split, city = relative_path.parts[:2]
            stem = annotation_path.name.removesuffix("_gtFine_polygons.json")
            image_id = f"{split}/{city}/{stem}"
            image_path = images_root / split / city / f"{stem}_leftImg8bit.png"
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))

            image_writer.writerow(
                {
                    "image_id": image_id,
                    "split": split,
                    "city": city,
                    "image_path": image_path,
                    "annotation_path": annotation_path,
                }
            )
            image_count += 1

            for object_index, object_data in enumerate(annotation.get("objects", [])):
                object_writer.writerow(
                    {
                        "image_id": image_id,
                        "object_index": object_index,
                        "label_name": object_data["label"],
                        "polygon": json.dumps(object_data.get("polygon", [])),
                    }
                )
                object_count += 1

    return {"images": image_count, "objects": object_count}
