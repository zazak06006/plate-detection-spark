# Microservice 1 — Prétraitement PySpark

## Description
Lit les images et labels YOLO du dataset de détection de plaques d'immatriculation,
construit un **DataFrame PySpark** enrichi avec du feature engineering, et exporte un **CSV** prêt pour l'entraînement.

## Structure de sortie
```
output/
└── dataset_ready.csv    ← CSV final avec toutes les features
```

## Colonnes du CSV produit
| Colonne | Description |
|---|---|
| `split` | train / valid / test |
| `image_name` | Nom du fichier image |
| `image_path` | Chemin absolu de l'image |
| `img_width` / `img_height` | Dimensions de l'image |
| `class_id` | ID de la classe YOLO (0 = plaque) |
| `cx_norm`, `cy_norm` | Centre de la bbox (normalisé 0–1) |
| `w_norm`, `h_norm` | Largeur/Hauteur bbox (normalisé) |
| `x_min`, `y_min`, `x_max`, `y_max` | Coordonnées absolues bbox |
| `bbox_area` | Aire absolue de la bbox |
| `bbox_area_norm` | Aire normalisée |
| `aspect_ratio` | Ratio largeur/hauteur |
| `bbox_size_category` | small / medium / large |
| `center_x_quadrant` | left / right |
| `center_y_quadrant` | top / bottom |
| `num_boxes` | Nombre de plaques dans l'image |

## Installation
```bash
pip install -r requirements.txt
```

## Lancement
```bash
python preprocessing.py
```
