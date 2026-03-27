import io
import sys
import csv
import uuid
import base64
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from functools import lru_cache
from PIL import Image, ImageDraw, ImageFont


IMAGES_DIR = Path(__file__).parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)

HISTORY_CSV_PATH = Path(__file__).parent / "history.csv"

# ============================================================================
# IMAGE FILESYSTEM MANAGEMENT
# ============================================================================
def save_images_to_filesystem(
    original_image: Optional[Image.Image],
    annotated_image: Optional[Image.Image],
    original_filename: str,
    run_id: Optional[str] = None
) -> Tuple[str, str]:
    """
    Sauvegarde les images originale et annotée sur le système de fichiers.
    Groupées par 'run_id' si fourni.

    Args:
        original_image: Image PIL originale (peut être None)
        annotated_image: Image PIL annotée (peut être None)
        original_filename: Nom du fichier uploadé
        run_id: ID de la session/batch pour groupement (ex: 20260327_0200_xxxx)

    Returns:
        (chemin_relatif_original, chemin_relatif_annoté)
    """
    if original_image is None and annotated_image is None:
        return "", ""

    try:
        # 1. Déterminer le répertoire de destination
        if run_id:
            dest_dir = IMAGES_DIR / "runs" / run_id
        else:
            # Fallback structure si pas de run_id (ex: test unitaire)
            now = datetime.now()
            date_str = now.strftime("%Y/%m/%d")
            time_uuid = now.strftime("%Y%m%d_%H%M%S") + f"_{uuid.uuid4().hex[:8]}"
            dest_dir = IMAGES_DIR / date_str / time_uuid

        dest_dir.mkdir(parents=True, exist_ok=True)

        # 2. Préparer les noms de fichiers
        # On utilise le nom de fichier original comme base pour éviter les collisions dans le run_id folder
        base_name = Path(original_filename).stem
        orig_ext = Path(original_filename).suffix.lower()
        if not orig_ext:
            orig_ext = ".jpg"

        # 3. Sauvegarder l'image originale
        original_path_rel = ""
        if original_image is not None:
            fname_orig = f"{base_name}_orig{orig_ext}"
            original_path_abs = dest_dir / fname_orig
            original_image.save(original_path_abs)
            original_path_rel = str(original_path_abs.relative_to(IMAGES_DIR.parent)).replace("\\", "/")

        # 4. Sauvegarder l'image annotée
        annotated_path_rel = ""
        if annotated_image is not None:
            fname_ann = f"{base_name}_ann.jpg"
            annotated_path_abs = dest_dir / fname_ann
            annotated_image.save(annotated_path_abs, format='JPEG', quality=90)
            annotated_path_rel = str(annotated_path_abs.relative_to(IMAGES_DIR.parent)).replace("\\", "/")

        return original_path_rel, annotated_path_rel

    except Exception as e:
        print(f"❌ Erreur sauvegarde images: {e}")
        return "", ""


def load_image_from_path(relative_path: str) -> Optional[Image.Image]:
    """
    Charge une image depuis un chemin relatif.

    Args:
        relative_path: Chemin relatif à partir de 3-web-interface/

    Returns:
        Image PIL ou None si manquante/erreur
    """
    if not relative_path:
        return None

    try:
        base_dir = IMAGES_DIR.parent
        abs_path = (base_dir / relative_path).resolve()

        # Sécurité: vérifier que le chemin est dans le répertoire images/
        if not str(abs_path).startswith(str(base_dir.resolve())):
            print(f"⚠️ Tentative de lecture hors du répertoire autorisé: {relative_path}")
            return None

        if not abs_path.exists():
            print(f"⚠️ Image manquante: {relative_path}")
            return None

        img = Image.open(abs_path).convert('RGB')
        return img

    except Exception as e:
        print(f"⚠️ Erreur lors du chargement de l'image ({relative_path}): {e}")
        return None


def migrate_old_history_csv() -> bool:
    """
    Migre l'ancien historique (format base64) vers le nouveau (format fichiers).

    Returns:
        True si migration effectuée, False sinon
    """
    history_path = HISTORY_CSV_PATH

    if not history_path.exists():
        return False

    try:
        # Augmenter temporairement la limite pour lire l'ancien CSV avec base64
        csv.field_size_limit(int(1e8))  # 100MB limit pour lecture old format

        # Vérifié le format du CSV
        with open(history_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []

            # Si colonne 'annotated_image_path' existe déjà, déjà migré
            if 'annotated_image_path' in fieldnames:
                return False

            # Si pas de colonne 'annotated_image', format inconnu
            if 'annotated_image' not in fieldnames:
                return False

            # Lire toutes les entrées
            entries = []
            for row in reader:
                entries.append(row)

        print(f"🔄 Migration de {len(entries)} entrées d'historique...")

        # Traiter chaque entrée
        migrated_rows = []
        for row in entries:
            base64_data = row.get('annotated_image', '')
            annotated_path = ""

            # Extraire et sauvegarder le base64 si présent
            if base64_data:
                try:
                    image_bytes = base64.b64decode(base64_data)
                    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

                    # Sauvegarder avec timestamp du CSV si disponible
                    try:
                        timestamp = datetime.fromisoformat(row['timestamp'])
                    except:
                        timestamp = None

                    ts_str = timestamp.strftime("%Y%m%d_%H%M%S") if timestamp else "migrated"
                    _, annotated_path = save_images_to_filesystem(None, image, "migrated.jpg", run_id=f"migrated_{ts_str}")
                except Exception as e:
                    print(f"⚠️ Impossible d'extraire image base64: {e}")

            # Créer la nouvelle ligne
            new_row = {
                'timestamp': row['timestamp'],
                'image_name': row['image_name'],
                'nb_plates': row['nb_plates'],
                'detections': row['detections'],
                'status': row['status'],
                'original_image_path': '',  # On n'a pas l'originale
                'annotated_image_path': annotated_path
            }
            migrated_rows.append(new_row)

        # Sauvegarder le CSV migré
        backup_path = history_path.with_suffix('.csv.backup')
        import shutil
        shutil.copy2(history_path, backup_path)
        print(f"✅ Backup créé: {backup_path}")

        # Réinitialiser la limite à normale
        csv.field_size_limit(131072)  # Reset to default

        # Écrire le nouveau CSV
        with open(history_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['timestamp', 'image_name', 'nb_plates', 'detections', 'status', 'original_image_path', 'annotated_image_path']
            )
            writer.writeheader()
            writer.writerows(migrated_rows)

        print(f"✅ Migration terminée: {len(migrated_rows)} entrées traitées")
        return True

    except Exception as e:
        print(f"❌ Erreur lors de la migration: {e}")
        csv.field_size_limit(131072)  # Reset to default
        return False


# ============================================================================
# HISTORY MANAGEMENT (CSV)
# ============================================================================
def save_to_history(
    image_name: str,
    nb_plates: int,
    detections: List[Dict],
    status: str = "success",
    original_image_path: Optional[str] = None,
    annotated_image_path: Optional[str] = None,
    annotated_image_base64: Optional[str] = None
):
    """
    Sauvegarde une prédiction dans l'historique CSV avec le nouveau format (chemins).

    Args:
        image_name: Nom du fichier image
        nb_plates: Nombre de plaques détectées
        detections: Liste des détections
        status: Statut de la prédiction
        original_image_path: Chemin relatif de l'image originale (nouveau format)
        annotated_image_path: Chemin relatif de l'image annotée (nouveau format)
        annotated_image_base64: Image annotée en base64 (ancien format, déprécié)
    """
    history_path = HISTORY_CSV_PATH

    # Créer le fichier avec header si inexistant
    if not history_path.exists():
        with open(history_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'image_name', 'nb_plates', 'detections', 'status',
                'original_image_path', 'annotated_image_path'
            ])

    # Ajouter l'entrée
    with open(history_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            image_name,
            nb_plates,
            str(detections),
            status,
            original_image_path or "",
            annotated_image_path or ""
        ])


def load_history(limit: int = 30) -> List[Dict]:
    """
    Charge l'historique depuis le CSV (nouveau format avec chemins).
    Effectue une migration automatique du format ancien si nécessaire.

    Args:
        limit: Nombre max d'entrées à charger

    Returns:
        Liste des entrées (les plus récentes en premier)
    """
    history_path = HISTORY_CSV_PATH

    if not history_path.exists():
        return []

    # Effectuer la migration si nécessaire
    migrate_old_history_csv()

    entries = []
    try:
        with open(history_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                entries.append({
                    "timestamp": row['timestamp'],
                    "image_name": row['image_name'],
                    "nb_plates": int(row['nb_plates']),
                    "detections": row['detections'],
                    "status": row['status'],
                    "original_image_path": row.get('original_image_path', ''),
                    "annotated_image_path": row.get('annotated_image_path', '')
                })
    except Exception as e:
        print(f"❌ Erreur lors du chargement de l'historique: {e}")
        return []

    # Retourner les plus récentes en premier
    return list(reversed(entries[-limit:]))


def clear_history():
    """Efface l'historique"""
    if HISTORY_CSV_PATH.exists():
        HISTORY_CSV_PATH.unlink()