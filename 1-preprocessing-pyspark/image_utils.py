"""Image processing utilities"""
import io
from PIL import Image
from pyspark.sql import functions as F
from pyspark.sql.types import BinaryType, StructType, StructField, FloatType
from config import IMG_SIZE


# Schema pour le retour de letterbox_resize_image
LETTERBOX_SCHEMA = StructType([
    StructField("image", BinaryType(), True),
    StructField("scale", FloatType(), True),
    StructField("pad_x", FloatType(), True),
    StructField("pad_y", FloatType(), True),
])


@F.udf(BinaryType())
def resize_image(content):
    """
    DEPRECATED: Utilisez letterbox_resize_image pour un resize sans déformation.
    Resize image to IMG_SIZE x IMG_SIZE (avec stretching)
    """
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        out = io.BytesIO()
        img.save(out, format="JPEG")
        return out.getvalue()
    except:
        return None


@F.udf(LETTERBOX_SCHEMA)
def letterbox_resize_image(content):
    """
    Letterbox resize: redimensionne en gardant le ratio d'aspect,
    puis ajoute du padding noir pour atteindre IMG_SIZE x IMG_SIZE.

    Returns:
        struct avec:
        - image: bytes JPEG de l'image letterboxée
        - scale: facteur d'échelle appliqué
        - pad_x: padding horizontal (gauche) en pixels, normalisé [0,1]
        - pad_y: padding vertical (haut) en pixels, normalisé [0,1]
    """
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
        orig_w, orig_h = img.size

        # Calculer le scale pour garder le ratio
        scale = min(IMG_SIZE / orig_w, IMG_SIZE / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        # Redimensionner avec le ratio préservé
        img_resized = img.resize((new_w, new_h), Image.BILINEAR)

        # Créer une nouvelle image noire et coller l'image centrée
        img_letterbox = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        pad_x = (IMG_SIZE - new_w) // 2
        pad_y = (IMG_SIZE - new_h) // 2
        img_letterbox.paste(img_resized, (pad_x, pad_y))

        # Sauvegarder en JPEG
        out = io.BytesIO()
        img_letterbox.save(out, format="JPEG", quality=95)

        # Retourner les infos normalisées pour transformation des labels
        return {
            "image": out.getvalue(),
            "scale": float(scale),
            "pad_x": float(pad_x) / IMG_SIZE,  # Normalisé [0, 1]
            "pad_y": float(pad_y) / IMG_SIZE,  # Normalisé [0, 1]
        }
    except Exception:
        return None
