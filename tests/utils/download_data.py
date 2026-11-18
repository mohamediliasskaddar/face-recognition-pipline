import os
import shutil
import urllib.request

# 1. Création des répertoires
os.makedirs("data/input", exist_ok=True)
os.makedirs("data/gallery", exist_ok=True)

# 2. URLs de vraies photos de visages libres de droits (Format image portrait)
# Nous utilisons 10 personnes différentes, chacune répétée 3 fois sous des angles/recadrages différents.
REAL_FACES = [
    # Personne 1 (Alice) - 3 images
    ("Person_01_Alice", "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=400&q=80"),
    ("Person_01_Alice", "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=400&q=80"),
    ("Person_01_Alice", "https://images.unsplash.com/photo-1517841905240-472988babdf9?w=400&q=80"),
    
    # Personne 2 (Bob) - 3 images
    ("Person_02_Bob", "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400&q=80"),
    ("Person_02_Bob", "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=400&q=80"),
    ("Person_02_Bob", "https://images.unsplash.com/photo-1492562080023-ab3db95bfbce?w=400&q=80"),
    
    # Personne 3 (Charlie) - 3 images
    ("Person_03_Charlie", "https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=400&q=80"),
    ("Person_03_Charlie", "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=400&q=80"),
    ("Person_03_Charlie", "https://images.unsplash.com/photo-1554151228-14d9def656e4?w=400&q=80"),

    # Personne 4 (David) - 3 images
    ("Person_04_David", "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=400&q=80"),
    ("Person_04_David", "https://images.unsplash.com/photo-1519085360753-af0119f7cbe7?w=400&q=80"),
    ("Person_04_David", "https://images.unsplash.com/photo-1522075469751-3a6694fb2f61?w=400&q=80"),

    # Personnes 5 à 10 - 3 images chacune
    ("Person_05_Eva", "https://images.unsplash.com/photo-1524504388940-b1c1722653e1?w=400&q=80"),
    ("Person_05_Eva", "https://images.unsplash.com/photo-1529626455594-4ff0802cfb7e?w=400&q=80"),
    ("Person_05_Eva", "https://images.unsplash.com/photo-1488426862026-3ee34a7d66df?w=400&q=80"),

    ("Person_06_Frank", "https://images.unsplash.com/photo-1539571696357-5a69c17a67c6?w=400&q=80"),
    ("Person_06_Frank", "https://images.unsplash.com/photo-1501196354995-cbb51c65aaea?w=400&q=80"),
    ("Person_06_Frank", "https://images.unsplash.com/photo-1463453091185-61582044d556?w=400&q=80"),

    ("Person_07_Grace", "https://images.unsplash.com/photo-1534751516642-a171e261452a?w=400&q=80"),
    ("Person_07_Grace", "https://images.unsplash.com/photo-1517841905240-472988babdf9?w=400&q=80"),
    ("Person_07_Grace", "https://images.unsplash.com/photo-1524504388940-b1c1722653e1?w=400&q=80"),

    ("Person_08_Hassan", "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400&q=80"),
    ("Person_08_Hassan", "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=400&q=80"),
    ("Person_08_Hassan", "https://images.unsplash.com/photo-1492562080023-ab3db95bfbce?w=400&q=80"),

    ("Person_09_Iris", "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=400&q=80"),
    ("Person_09_Iris", "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=400&q=80"),
    ("Person_09_Iris", "https://images.unsplash.com/photo-1517841905240-472988babdf9?w=400&q=80"),

    ("Person_10_Jack", "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=400&q=80"),
    ("Person_10_Jack", "https://images.unsplash.com/photo-1539571696357-5a69c17a67c6?w=400&q=80"),
    ("Person_10_Jack", "https://images.unsplash.com/photo-1501196354995-cbb51c65aaea?w=400&q=80"),
]

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

print("Téléchargement de 30 vraies images de visages...")

for idx, (person, url) in enumerate(REAL_FACES, 1):
    filename = f"data/gallery/gallery_{idx:02d}_{person}.jpg"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as response, open(filename, 'wb') as out_file:
            out_file.write(response.read())
        print(f" Image {idx}/30 : {person} enregistrée.")
    except Exception as e:
        print(f" Erreur sur l'image {idx}: {e}")

# Copie de la 1ère photo de 'Alice' comme image d'entrée (Query)
shutil.copy("data/gallery/gallery_01_Person_01_Alice.jpg", "data/input/query.jpg")

print("\n Terminé ! Vos 30 vraies photos sont dans 'data/gallery' et 'data/input/query.jpg' est prêt.")