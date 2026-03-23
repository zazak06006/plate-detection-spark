# 🚗 License Plate Detection — Projet Spark Core

Architecture **3 microservices** pour la détection de plaques d'immatriculation.

---

## Architecture

```
Projet Spark Core/
│
├── license-plate-detection-dataset-10125-images/   ← Dataset YOLO (train/valid/test)
│
├── 1-preprocessing-pyspark/     ← Microservice 1 : Lecture & Prétraitement PySpark
│   ├── preprocessing.py         ← Script principal PySpark
│   ├── requirements.txt
│   ├── README.md
│   └── output/
│       └── dataset_ready.csv    ← CSV généré (features pour entraînement)
│
├── 2-model-training/            ← Microservice 2 : Entraînement YOLOv8
│   ├── training.ipynb           ← Notebook Jupyter complet
│   ├── requirements.txt
│   └── models/                  ← Modèles entraînés & graphiques
│       ├── data.yaml
│       ├── best.pt              ← Meilleur modèle après entraînement
│       └── license_plate_yolov8n/
│
└── 3-web-interface/             ← Microservice 3 : Interface Web Flask
    ├── app.py                   ← API Flask (EDA + Prediction)
    ├── requirements.txt
    ├── model/
    │   └── best.pt              ← Modèle copié depuis MS2
    ├── templates/
    │   ├── index.html           ← Dashboard EDA
    │   └── predict.html         ← Page prédiction
    ├── static/
    │   ├── css/style.css
    │   └── js/main.js & predict.js
    └── uploads/                 ← Images uploadées temporairement
```

---

## 🚀 Ordre d'exécution

### Étape 1 — Prétraitement PySpark
```bash
cd 1-preprocessing-pyspark
pip install -r requirements.txt
python preprocessing.py
# → Génère output/dataset_ready.csv
```

### Étape 2 — Entraînement du modèle
```bash
cd 2-model-training
pip install -r requirements.txt
jupyter notebook training.ipynb
# → Entraîne YOLOv8, exporte best.pt vers 3-web-interface/model/
```

### Étape 3 — Interface Web
```bash
cd 3-web-interface
pip install -r requirements.txt
python app.py
# → http://localhost:5000
```

---

## 🔗 Pages de l'interface web

| Route | Description |
|---|---|
| `http://localhost:5000/` | Dashboard EDA avec graphiques |
| `http://localhost:5000/predict` | Prédiction sur image uploadée |
| `http://localhost:5000/api/stats` | API JSON des statistiques |

---

## 📊 Technologies

| Couche | Technologies |
|---|---|
| Preprocessing | PySpark 3.x, Pillow |
| Entraînement | YOLOv8 (Ultralytics), Pandas, Matplotlib |
| Interface | Flask, HTML/CSS/JS (Vanilla) |
