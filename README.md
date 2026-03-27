# License Plate Detection — Projet Spark Core

Détection de plaques d'immatriculation via une **architecture en 3 microservices** : prétraitement PySpark, entraînement YOLOv8, et interface web.

---

## Structure du projet

```
Projet Spark Core/
├── license-plate-detection-dataset-10125-images/   Jeu de données YOLO
│   ├── train/
│   ├── valid/
│   └── test/
│
├── 1-preprocessing-pyspark/                        Microservice 1 : Prétraitement
│   ├── preprocessing.py
│   ├── data_loading.py
│   ├── data_processing.py
│   ├── spark_setup.py
│   ├── requirements.txt
│   └── data/processed/
│
├── 2-model-training/                               Microservice 2 : Entraînement
│   ├── training.ipynb
│   ├── train.py
│   ├── model.py
│   ├── dataloader.py
│   ├── loss.py
│   ├── requirements.txt
│   ├── checkpoints/
│   │   └── best_model.pt
│   └── models_ssd/
│
└── 3-web-interface/                                Microservice 3 : Interface web
    ├── app.py
    ├── api.py
    ├── inference.py
    ├── functions_stremlit.py
    ├── requirements.txt
    ├── run.sh
    └── images/
```

---

## Démarrage rapide

### 1️⃣ Prétraitement des données (PySpark)

#### Prérequis

Téléchargez le dataset sur [Kaggle](https://www.kaggle.com/) : **License Plate Detection Dataset (10125 images)**

Décompressez l'archive à la racine du projet sous le dossier `license-plate-detection-dataset-10125-images/`

#### Lancement

```bash
cd 1-preprocessing-pyspark
pip install -r requirements.txt
python preprocessing.py
```

**Résultat** : `data/processed/dataset_ready.csv`

---

### 2️⃣ Entraînement du modèle (SimpleSDD)

#### Option A : Utiliser le modèle pré-entraîné

Téléchargez le modèle via ce lien :
https://www.transfernow.net/d/start?utm_source=20260326nNyk0c9m&utm_term=r63cpI

Placez-le dans : `2-model-training/checkpoints/best_model.pt`

#### Option B : Réentraîner

```bash
cd 2-model-training
pip install -r requirements.txt
jupyter notebook training.ipynb
```

**Artefacts générés** :
- `checkpoints/best_model.pt` — Meilleur modèle
- Graphiques et métriques d'entraînement
- Fichiers de configuration YOLO

---

### 3️⃣ Interface web (FastAPI + Streamlit)

```bash
cd 3-web-interface
pip install -r requirements.txt
bash run.sh
```

**Accès** :
- 🎨 Dashboard Streamlit : http://localhost:8501
- 📚 API Swagger : http://localhost:8000/docs

---

## Technologies

| Composant    | Stack                           |
|--------------|--------------------------------|
| Prétraitement | PySpark 3.x, Pillow            |
| Entraînement | YOLOv8 (Ultralytics), PyTorch |
| Interface    | FastAPI, Streamlit             |

---

## Flux de données

```
Dataset brut
    ↓
[Microservice 1] PySpark : Nettoyage & extraction features
    ↓
dataset_ready.csv
    ↓
[Microservice 2] Entraînement YOLOv8 + SSD
    ↓
best_model.pt
    ↓
[Microservice 3] Inférence en temps réel
    ↓
Web Interface (Streamlit)
```

---

## Fichiers importants

- [1-preprocessing-pyspark/README.md](1-preprocessing-pyspark/README.md) — Documentation détaillée du prétraitement
- [2-model-training/training.ipynb](2-model-training/training.ipynb) — Notebook d'entraînement complet
- [3-web-interface/run.sh](3-web-interface/run.sh) — Script de lancement

---

## Notes

- ⚠️ Les modèles pré-entraînés sont volumineux et stockés en ligne
- 📦 Chaque microservice a ses propres dépendances (`requirements.txt`)
- 🔄 Les services peuvent être exécutés indépendamment une fois le modèle disponible
