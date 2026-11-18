from matplotlib import pyplot as plt

from src.config import load_config
from src.detection import FaceDetector, select_main_face
from src.embedding import InsightFaceEmbedder
from src.gallery import build_gallery_index
from src.io_utils import read_image
from src.matching import match_images

config = load_config()
embedder = InsightFaceEmbedder(config)
detector = FaceDetector(config, analysis=embedder.analysis)

query_path = "data/input/query.jpg"
query_image = read_image(query_path)
query_faces = detector.detect(query_image)

if not query_faces:
    raise RuntimeError(f"Aucun visage détecté dans {query_path}")

query_face = select_main_face(query_faces)
query_crop = detector.align(query_image, query_face.kps)
query_embedding = embedder.embed_detected_face(query_face, query_crop)

records = build_gallery_index("data/gallery", detector, embedder)
matches = match_images(query_embedding, records, config.matching.threshold)[:10]

print(f"Seuil : {config.matching.threshold:.2f}")
print(f"Visages indexés dans la galerie : {len(records)}")
print(f"Correspondances affichées : {len(matches)}")
for rank, match in enumerate(matches, start=1):
    print(
        f"{rank:2}. score={match.best_score:.3f} "
        f"visage={match.matched_face_index}  {match.image_path}"
    )

columns = 3
rows = (len(matches) + 1 + columns - 1) // columns
figure, axes = plt.subplots(rows, columns, figsize=(15, 5 * rows), squeeze=False)
flat_axes = axes.ravel()

flat_axes[0].imshow(query_image[:, :, ::-1])
flat_axes[0].set_title("Requête")

for index, match in enumerate(matches, start=1):
    image = read_image(match.image_path)
    flat_axes[index].imshow(image[:, :, ::-1])
    flat_axes[index].set_title(
        f"{match.best_score:.3f} — visage {match.matched_face_index}\n"
        f"{match.image_path}"
    )

for axis in flat_axes[len(matches) + 1:]:
    axis.axis("off")

for axis in flat_axes[:len(matches) + 1]:
    axis.axis("off")

figure.tight_layout()
# plt.show()
output_path = "outputs/top10.png"
figure.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Grille enregistrée : {output_path}")




