"""Image processing utilities"""
import io
from PIL import Image
from pyspark.sql import functions as F
from pyspark.sql.types import BinaryType
from config import IMG_SIZE

@F.udf(BinaryType())
def resize_image(content):
    """Resize image to IMG_SIZE x IMG_SIZE"""
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        out = io.BytesIO()
        img.save(out, format="JPEG")
        return out.getvalue()
    except:
        return None
