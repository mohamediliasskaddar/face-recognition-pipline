import os
import csv
import shutil
import random
from pathlib import Path

# ================= CONFIGURATION =================
SOURCE_DIR = Path.home() / "Downloads/archive2/lfw-funneled/lfw_funneled"
OUTPUT_DIR = Path.home() / "Downloads/test_lfw_experiment"

MIN_IMAGES_PER_PERSON = 5   # Garder uniquement les personnes avec au moins X photos
MAX_PEOPLE = 30             # Nombre max de personnes pour ce test (mettre None pour tout prendre)
USE_P_IDS = True            # True -> P01, P02... | False -> garde "Aaron_Eckhart"
# =================================================

all_images_dir = OUTPUT_DIR / "all_images"
ground_truth_dir = OUTPUT_DIR / "ground_truth"
csv_path = OUTPUT_DIR / "labels.csv"

# Nettoyage si le dossier de test existe déjà
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

all_images_dir.mkdir(parents=True)
ground_truth_dir.mkdir(parents=True)

# 1. Filtrer les dossiers selon le nombre minimum d'images
valid_people = []
for person_dir in sorted(SOURCE_DIR.iterdir()):
    if person_dir.is_dir():
        images = list(person_dir.glob("*.jpg"))
        if len(images) >= MIN_IMAGES_PER_PERSON:
            valid_people.append((person_dir.name, images))

print(f"Personnes trouvées avec au moins {MIN_IMAGES_PER_PERSON} images : {len(valid_people)}")

# 2. Sélectionner un sous-ensemble aléatoire (si MAX_PEOPLE est défini)
if MAX_PEOPLE and len(valid_people) > MAX_PEOPLE:
    random.seed(42)  # Pour avoir toujours le même échantillon reproductible
    selected_people = random.sample(valid_people, MAX_PEOPLE)
else:
    selected_people = valid_people

# 3. Générer les dossiers et le labels.csv
csv_rows = []
total_images = 0

# Calcul du padding pour les IDs (ex: P01..P30 ou P001..P150)
pad_width = max(2, len(str(len(selected_people))))

for idx, (person_name, images) in enumerate(sorted(selected_people), start=1):
    person_id = f"P{idx:0{pad_width}d}" if USE_P_IDS else person_name
    
    # Dossier vérité terrain pour cette personne
    person_gt_dir = ground_truth_dir / person_id
    person_gt_dir.mkdir(exist_ok=True)
    
    for img_path in images:
        # Copie dans le dossier global (entrée de ton algorithme)
        shutil.copy2(img_path, all_images_dir / img_path.name)
        
        # Copie dans le dossier vérité terrain (pour ta comparaison visuelle)
        shutil.copy2(img_path, person_gt_dir / img_path.name)
        
        # Ajout au CSV
        csv_rows.append([img_path.name, person_id])
        total_images += 1

# Mélanger l'ordre des lignes dans le CSV pour simuler un flux non trié
random.shuffle(csv_rows)

with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "person_id"])
    writer.writerows(csv_rows)

print("-" * 40)
print(f"Préparation terminée dans : {OUTPUT_DIR}")
print(f"- Personnes sélectionnées : {len(selected_people)}")
print(f"- Total images copiées    : {total_images}")
print(f"- Dossier entrée projet   : {all_images_dir}")
print(f"- Dossier vérité terrain  : {ground_truth_dir}")
print(f"- Fichier labels          : {csv_path}")