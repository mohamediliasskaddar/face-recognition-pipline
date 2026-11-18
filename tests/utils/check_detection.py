from pathlib import Path
import math

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.detection import FaceDetector
from src.io_utils import list_images, read_image


gallery_paths = list_images(Path("data/gallery"))
query_path = Path("data/input/query.jpg")
samples = [(path, "galerie") for path in gallery_paths]
samples.append((query_path, "requête"))

detector = FaceDetector()
annotated_images = []
aligned_faces = []
gallery_with_faces = 0

for image_path, category in samples:
    image_bgr = read_image(image_path)
    faces = detector.detect(image_bgr)

    if category == "galerie" and faces:
        gallery_with_faces += 1

    annotated_bgr = image_bgr.copy()
    for face_index, face in enumerate(faces):
        x1, y1, x2, y2 = (int(round(value)) for value in face.bbox)
        cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for x, y in face.kps:
            cv2.circle(
                annotated_bgr,
                (int(round(x)), int(round(y))),
                3,
                (0, 0, 255),
                -1,
            )

        crop_bgr = detector.align(image_bgr, face.kps)
        aligned_faces.append(
            (f"{image_path.name} — visage {face_index + 1}", crop_bgr)
        )

    annotated_images.append(
        (
            f"{category}: {image_path.name} — {len(faces)} visage(s)",
            annotated_bgr,
        )
    )

gallery_count = len(gallery_paths)
percentage = 100 * gallery_with_faces / gallery_count if gallery_count else 0
print(
    f"Galerie : {gallery_with_faces}/{gallery_count} images avec au moins "
    f"une détection ({percentage:.1f} %)."
)

output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)


def save_grid(items, destination, columns=5):
    rows = math.ceil(len(items) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * 3, rows * 3),
        squeeze=False,
    )

    for axis, (title, image_bgr) in zip(axes.flat, items):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        axis.imshow(image_rgb)
        axis.set_title(title, fontsize=8)
        axis.axis("off")

    for axis in axes.flat[len(items):]:
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)


save_grid(
    annotated_images,
    output_dir / "detection_grid.png",
)
save_grid(
    aligned_faces,
    output_dir / "aligned_faces_grid.png",
)

print("Grilles enregistrées dans outputs/.")