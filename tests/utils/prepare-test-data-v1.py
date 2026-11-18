import csv
import shutil
import random
from pathlib import Path

# ================= CONFIGURATION =================
SOURCE_DIR = Path.home() / "Downloads/archive2/lfw-funneled/lfw_funneled"
OUTPUT_DIR = Path.home() / "Downloads/test_lfw_sample"

MIN_IMAGES_PER_PERSON = 5    # Minimum de photos pour qu'un groupe ait du sens
MAX_IMAGES_PER_PERSON = 10   # Évite qu'une seule célébrité prenne 500 photos
MAX_PEOPLE = 25              # Nombre de personnes différentes dans l'échantillon

MAX_FILE_SIZE_MB = 20.0      # Limite par image (20 Mio)
MAX_FOLDER_SIZE_MB = 100.0   # Marge de sécurité sous ta limite de 250 Mio
USE_P_IDS = True             # True -> P01, P02... | False -> Nom_Prenom
# =================================================

MAX_FILE_BYTES = int(MAX_FILE_SIZE_MB * 1024 * 1024)
MAX_FOLDER_BYTES = int(MAX_FOLDER_SIZE_MB * 1024 * 1024)

all_images_dir = OUTPUT_DIR / "all_images"
ground_truth_dir = OUTPUT_DIR / "ground_truth"
csv_path = OUTPUT_DIR / "labels.csv"

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

all_images_dir.mkdir(parents=True)
ground_truth_dir.mkdir(parents=True)

random.seed(42)  # Reproductibilité

# 1. Filtrer les personnes valides et vérifier la taille unitaire des images
valid_people = []
for person_dir in sorted(SOURCE_DIR.iterdir()):
    if person_dir.is_dir():
        valid_imgs = [
            img for img in sorted(person_dir.glob("*.jpg"))
            if img.stat().st_size <= MAX_FILE_BYTES
        ]
        if len(valid_imgs) >= MIN_IMAGES_PER_PERSON:
            # Limiter le nombre d'images par personne pour garder un test équilibré
            if len(valid_imgs) > MAX_IMAGES_PER_PERSON:
                valid_imgs = random.sample(valid_imgs, MAX_IMAGES_PER_PERSON)
            valid_people.append((person_dir.name, valid_imgs))

# 2. Sélectionner l'échantillon de personnes
selected_people = random.sample(valid_people, min(MAX_PEOPLE, len(valid_people)))

# 3. Copier en surveillant strictement le poids total du dossier all_images
csv_rows = []
current_folder_bytes = 0
total_images = 0
people_included = 0

pad_width = max(2, len(str(len(selected_people))))

for idx, (person_name, images) in enumerate(sorted(selected_people), start=1):
    # Calculer le poids du lot pour cette personne
    batch_bytes = sum(img.stat().st_size for img in images)
    
    # Arrêter proprement avant de dépasser la limite globale du dossier
    if current_folder_bytes + batch_bytes > MAX_FOLDER_BYTES:
        print(f"Arrêt anticipé : limite de {MAX_FOLDER_SIZE_MB} Mio atteinte.")
        break

    person_id = f"P{idx:0{pad_width}d}" if USE_P_IDS else person_name
    person_gt_dir = ground_truth_dir / person_id
    person_gt_dir.mkdir(exist_ok=True)

    for img_path in images:
        shutil.copy2(img_path, all_images_dir / img_path.name)
        shutil.copy2(img_path, person_gt_dir / img_path.name)
        csv_rows.append([img_path.name, person_id])
        total_images += 1

    current_folder_bytes += batch_bytes
    people_included += 1

# Mélanger l'ordre dans le CSV
random.shuffle(csv_rows)

with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "person_id"])
    writer.writerows(csv_rows)

size_mb = current_folder_bytes / (1024 * 1024)
print("-" * 45)
print(f"Échantillon prêt dans : {OUTPUT_DIR}")
print(f"- Personnes incluses  : {people_included}")
print(f"- Images totales      : {total_images}")
print(f"- Poids total dossier : {size_mb:.2f} Mio / {MAX_FOLDER_SIZE_MB} Mio max")