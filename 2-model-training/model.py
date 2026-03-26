"""
SSD Simple pour détection de plaques d'immatriculation
Backbone CNN léger + Heads classification/régression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional
import torchvision.ops as ops


# ============================================================================
# CONFIGURATION
# ============================================================================
IMG_SIZE = 256
NUM_CLASSES = 2  # 0 = background, 1 = plate
CLASS_NAMES = ["background", "plate"]
MAX_OBJECTS = 20


# ============================================================================
# BOX CODER - Encode/Decode offsets relatifs aux anchors
# ============================================================================
class BoxCoder:
    """
    Encode les GT boxes en offsets (deltas) par rapport aux anchors.
    Decode les prédictions (deltas) en boxes absolues.

    Formules d'encodage (GT -> deltas):
        delta_cx = (gt_cx - anchor_cx) / anchor_w
        delta_cy = (gt_cy - anchor_cy) / anchor_h
        delta_w = log(gt_w / anchor_w)
        delta_h = log(gt_h / anchor_h)

    Formules de décodage (deltas -> boxes):
        pred_cx = delta_cx * anchor_w + anchor_cx
        pred_cy = delta_cy * anchor_h + anchor_cy
        pred_w = exp(delta_w) * anchor_w
        pred_h = exp(delta_h) * anchor_h
    """

    def __init__(self, weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)):
        """
        Args:
            weights: Poids pour normaliser les deltas (cx, cy, w, h).
                     Des valeurs plus grandes réduisent la variance des deltas.
        """
        self.weights = weights

    def encode(self, gt_boxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """
        Encode les GT boxes en deltas par rapport aux anchors.

        Args:
            gt_boxes: [N, 4] format (cx, cy, w, h) normalisé [0,1]
            anchors: [N, 4] format (cx, cy, w, h) normalisé [0,1]

        Returns:
            deltas: [N, 4] offsets encodés
        """
        wx, wy, ww, wh = self.weights

        # Extraire les composantes
        gt_cx, gt_cy, gt_w, gt_h = gt_boxes.unbind(-1)
        anchor_cx, anchor_cy, anchor_w, anchor_h = anchors.unbind(-1)

        # Éviter division par zéro
        anchor_w = anchor_w.clamp(min=1e-6)
        anchor_h = anchor_h.clamp(min=1e-6)
        gt_w = gt_w.clamp(min=1e-6)
        gt_h = gt_h.clamp(min=1e-6)

        # Calculer les deltas
        delta_cx = wx * (gt_cx - anchor_cx) / anchor_w
        delta_cy = wy * (gt_cy - anchor_cy) / anchor_h
        delta_w = ww * torch.log(gt_w / anchor_w)
        delta_h = wh * torch.log(gt_h / anchor_h)

        return torch.stack([delta_cx, delta_cy, delta_w, delta_h], dim=-1)

    def decode(self, deltas: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """
        Decode les deltas en boxes absolues.

        Args:
            deltas: [N, 4] offsets prédits
            anchors: [N, 4] format (cx, cy, w, h)

        Returns:
            boxes: [N, 4] format (cx, cy, w, h) normalisé [0,1]
        """
        wx, wy, ww, wh = self.weights

        # Extraire les composantes
        delta_cx, delta_cy, delta_w, delta_h = deltas.unbind(-1)
        anchor_cx, anchor_cy, anchor_w, anchor_h = anchors.unbind(-1)

        # Clamp delta_w et delta_h pour éviter exp() trop grand
        delta_w = delta_w.clamp(max=4.0)
        delta_h = delta_h.clamp(max=4.0)

        # Décoder
        pred_cx = (delta_cx / wx) * anchor_w + anchor_cx
        pred_cy = (delta_cy / wy) * anchor_h + anchor_cy
        pred_w = torch.exp(delta_w / ww) * anchor_w
        pred_h = torch.exp(delta_h / wh) * anchor_h

        # Clamp pour rester dans [0, 1]
        pred_cx = pred_cx.clamp(0, 1)
        pred_cy = pred_cy.clamp(0, 1)
        pred_w = pred_w.clamp(min=0.001, max=1.0)
        pred_h = pred_h.clamp(min=0.001, max=1.0)

        return torch.stack([pred_cx, pred_cy, pred_w, pred_h], dim=-1)


# Instance globale du BoxCoder
BOX_CODER = BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))  # Poids standard SSD


# ============================================================================
# BACKBONE CNN SIMPLE
# ============================================================================
class SimpleCNNBackbone(nn.Module):
    """
    Backbone CNN léger et simple.
    Input: [B, 3, 256, 256]
    Output: feature maps à différentes échelles
    """

    def __init__(self):
        super().__init__()

        # Block 1: 256 -> 128
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 256 -> 128
        )

        # Block 2: 128 -> 64
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 128 -> 64
        )

        # Block 3: 64 -> 32
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 64 -> 32
        )

        # Block 4: 32 -> 16
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 32 -> 16
        )

        # Block 5: 16 -> 8
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 16 -> 8
        )

        # Nombre de canaux en sortie de chaque bloc
        self.out_channels = [128, 256, 512]  # conv3, conv4, conv5

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: [B, 3, 256, 256]
        Returns:
            List de feature maps: [32x32, 16x16, 8x8]
        """
        x = self.conv1(x)  # [B, 32, 128, 128]
        x = self.conv2(x)  # [B, 64, 64, 64]

        feat1 = self.conv3(x)   # [B, 128, 32, 32]
        feat2 = self.conv4(feat1)  # [B, 256, 16, 16]
        feat3 = self.conv5(feat2)  # [B, 512, 8, 8]

        return [feat1, feat2, feat3]


# ============================================================================
# DETECTION HEAD
# ============================================================================
class DetectionHead(nn.Module):
    """
    Head pour classification et régression sur une feature map.
    """

    def __init__(self, in_channels: int, num_anchors: int = 1, num_classes: int = 2):
        super().__init__()

        self.num_anchors = num_anchors
        self.num_classes = num_classes

        # Classification head
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=1)
        )

        # Regression head (cx, cy, w, h)
        self.reg_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            cls_preds: [B, H*W*num_anchors, num_classes]
            reg_preds: [B, H*W*num_anchors, 4]
        """
        B = x.size(0)

        # Classification
        cls = self.cls_head(x)  # [B, num_anchors*num_classes, H, W]
        cls = cls.permute(0, 2, 3, 1).contiguous()  # [B, H, W, num_anchors*num_classes]
        cls = cls.view(B, -1, self.num_classes)  # [B, H*W*num_anchors, num_classes]

        # Regression
        reg = self.reg_head(x)  # [B, num_anchors*4, H, W]
        reg = reg.permute(0, 2, 3, 1).contiguous()  # [B, H, W, num_anchors*4]
        reg = reg.view(B, -1, 4)  # [B, H*W*num_anchors, 4]

        return cls, reg


# ============================================================================
# SSD MODEL
# ============================================================================
class SimpleSSD(nn.Module):
    """
    SSD simplifié pour détection de plaques.

    Architecture:
    - Backbone CNN léger
    - 3 feature maps (32x32, 16x16, 8x8)
    - 1 anchor par position
    - Heads classification + régression

    Total anchors: 32*32 + 16*16 + 8*8 = 1024 + 256 + 64 = 1344
    """

    def __init__(self, num_classes: int = 2, num_anchors: int = 1):
        super().__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Backbone
        self.backbone = SimpleCNNBackbone()
        self.register_buffer('anchors', self._make_anchors())

        # Detection heads (un par feature map)
        self.heads = nn.ModuleList([
            DetectionHead(128, num_anchors, num_classes),  # 32x32
            DetectionHead(256, num_anchors, num_classes),  # 16x16
            DetectionHead(512, num_anchors, num_classes),  # 8x8
        ])

        # Calcul du nombre total d'anchors
        self.num_anchors_total = 32*32 + 16*16 + 8*8  # = 1344

        # Initialisation des poids
        self._init_weights()
        
    def _make_anchors(self) -> torch.Tensor:
        """Génère les 1344 anchors (cx, cy, w, h) normalisées [0, 1]"""
        resolutions = [32, 16, 8]
        # On définit une taille d'anchor par défaut pour chaque niveau (ex: 10%, 20%, 40% de l'image)
        anchor_sizes = [0.1, 0.2, 0.4] 
        
        all_anchors = []
        for res, size in zip(resolutions, anchor_sizes):
            # Créer une grille de centres (cx, cy)
            grid = torch.linspace(0.5 / res, 1 - 0.5 / res, res)
            cy, cx = torch.meshgrid(grid, grid, indexing='ij')
            
            # Formater en [H*W, 4] -> (cx, cy, w, h)
            # Ici on utilise une seule anchor carrée par position pour simplifier
            anchors_level = torch.stack([
                cx.flatten(), 
                cy.flatten(), 
                torch.full((res*res,), size), # largeur w
                torch.full((res*res,), size)  # hauteur h
            ], dim=1)
            
            all_anchors.append(anchors_level)
            
        return torch.cat(all_anchors, dim=0) # [1344, 4]

    def _init_weights(self):
        """Initialisation Xavier/Kaiming"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, 3, 256, 256]
        Returns:
            cls_preds: [B, 1344, num_classes] - logits
            reg_preds: [B, 1344, 4] - deltas (offsets par rapport aux anchors)
        """
        # Extraire les feature maps
        features = self.backbone(x)

        # Prédictions de chaque head
        all_cls = []
        all_reg = []

        for feat, head in zip(features, self.heads):
            cls, reg = head(feat)
            all_cls.append(cls)
            all_reg.append(reg)

        # Concaténer toutes les prédictions
        cls_preds = torch.cat(all_cls, dim=1)  # [B, 1344, num_classes]
        reg_preds = torch.cat(all_reg, dim=1)  # [B, 1344, 4] - deltas bruts

        # NOTE: On ne fait plus sigmoid ici !
        # Les reg_preds sont maintenant des deltas qui seront décodés
        # avec le BoxCoder lors de l'inférence

        return cls_preds, reg_preds


# ============================================================================
# FACTORY
# ============================================================================
def create_model(num_classes: int = NUM_CLASSES) -> SimpleSSD:
    """Crée et retourne le modèle SSD"""
    model = SimpleSSD(num_classes=num_classes)

    # Compter les paramètres
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"📦 Model created: SimpleSSD")
    print(f"   Parameters: {num_params:,} ({num_trainable:,} trainable)")
    print(f"   Num classes: {num_classes}")
    print(f"   Total anchors: {model.num_anchors_total}")

    return model


# ============================================================================
# INFERENCE UTILITIES
# ============================================================================
def decode_predictions(
    cls_preds: torch.Tensor,
    reg_preds: torch.Tensor,
    anchors: torch.Tensor,
    score_threshold: float = 0.5,
    nms_threshold: float = 0.4,
    max_detections: int = 100
) -> List[Dict]:
    """
    Décode les prédictions (deltas) en boxes finales avec NMS.

    Args:
        cls_preds: [B, num_anchors, num_classes] - logits
        reg_preds: [B, num_anchors, 4] - deltas (offsets par rapport aux anchors)
        anchors: [num_anchors, 4] - anchors (cx, cy, w, h)
        score_threshold: Seuil de confiance
        nms_threshold: Seuil IoU pour NMS
        max_detections: Max boxes par image

    Returns:
        Liste de dicts (un par image du batch):
        {
            'boxes': Tensor [N, 4] en format (x1, y1, x2, y2) normalisé,
            'scores': Tensor [N],
            'classes': Tensor [N]
        }
    """
    B = cls_preds.size(0)
    device = cls_preds.device

    # Appliquer softmax pour avoir des probabilités
    cls_probs = F.softmax(cls_preds, dim=-1)

    results = []

    for b in range(B):
        # Prendre la classe 1 (plate) uniquement
        scores = cls_probs[b, :, 1]  # [num_anchors]
        deltas = reg_preds[b]         # [num_anchors, 4]

        # Filtrer par score
        mask = scores > score_threshold
        scores = scores[mask]
        deltas = deltas[mask]
        anchors_masked = anchors[mask]

        if len(scores) == 0:
            results.append({
                'boxes': torch.empty((0, 4), device=device),
                'scores': torch.empty(0, device=device),
                'classes': torch.empty(0, dtype=torch.long, device=device)
            })
            continue

        # Décoder les deltas en boxes (cx, cy, w, h)
        boxes_cxcywh = BOX_CODER.decode(deltas, anchors_masked)

        # Convertir cx,cy,w,h -> x1,y1,x2,y2
        boxes_xyxy = cxcywh_to_xyxy(boxes_cxcywh)

        # Clamp pour être sûr d'être dans [0, 1]
        boxes_xyxy = boxes_xyxy.clamp(0, 1)

        # NMS
        keep = ops.nms(boxes_xyxy, scores, nms_threshold)
        keep = keep[:max_detections]

        results.append({
            'boxes': boxes_xyxy[keep],
            'scores': scores[keep],
            'classes': torch.ones(len(keep), dtype=torch.long, device=device)  # Classe 1 = plate
        })

    return results


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convertit (cx, cy, w, h) -> (x1, y1, x2, y2)

    Args:
        boxes: [..., 4] avec (cx, cy, w, h)
    Returns:
        boxes: [..., 4] avec (x1, y1, x2, y2)
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convertit (x1, y1, x2, y2) -> (cx, cy, w, h)

    Args:
        boxes: [..., 4] avec (x1, y1, x2, y2)
    Returns:
        boxes: [..., 4] avec (cx, cy, w, h)
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


# ============================================================================
# PREDICT FUNCTION (pour inférence)
# ============================================================================
@torch.no_grad()
def predict(
    model: SimpleSSD,
    images: torch.Tensor,
    score_threshold: float = 0.5,
    nms_threshold: float = 0.4,
    max_detections: int = 100
) -> List[Dict]:
    """
    Fonction de prédiction avec BoxCoder.

    Args:
        model: Modèle SSD entraîné
        images: Tensor [B, 3, 256, 256] ou [3, 256, 256]
        score_threshold: Seuil de confiance
        nms_threshold: Seuil IoU pour NMS
        max_detections: Max boxes par image

    Returns:
        Liste de dicts (un par image):
        {
            'boxes': Tensor [N, 4] format (x1, y1, x2, y2) normalisé [0-1],
            'scores': Tensor [N],
            'classes': Tensor [N]
        }

    Usage:
        results = predict(model, image_tensor)
        for det in results:
            print(det['boxes'], det['scores'])
    """
    model.eval()

    # Ajouter dimension batch si nécessaire
    if images.dim() == 3:
        images = images.unsqueeze(0)

    # Forward pass
    cls_preds, reg_preds = model(images)

    # Récupérer les anchors du modèle
    anchors = model.anchors.to(images.device)

    # Décoder avec NMS (utilise BoxCoder en interne)
    results = decode_predictions(
        cls_preds, reg_preds, anchors,
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
        max_detections=max_detections
    )

    return results


def predict_single(
    model: SimpleSSD,
    image: torch.Tensor,
    score_threshold: float = 0.5,
    nms_threshold: float = 0.4
) -> Dict:
    """
    Prédit sur une seule image (raccourci).

    Args:
        model: Modèle SSD
        image: Tensor [3, 256, 256]

    Returns:
        Dict avec boxes, scores, classes
    """
    results = predict(model, image, score_threshold, nms_threshold)
    return results[0]


# ============================================================================
# TEST
# ============================================================================
if __name__ == "__main__":
    print("🧪 Testing SimpleSSD model...")

    # Créer le modèle
    model = create_model()

    # Test forward pass
    x = torch.randn(2, 3, 256, 256)
    cls_preds, reg_preds = model(x)

    print(f"\n🔍 Forward pass:")
    print(f"   Input: {x.shape}")
    print(f"   Cls output: {cls_preds.shape}")
    print(f"   Reg output: {reg_preds.shape}")

    # Test predict
    print(f"\n🔍 Testing predict()...")
    results = predict(model, x, score_threshold=0.3)

    for i, det in enumerate(results):
        print(f"   Image {i}: {len(det['boxes'])} detections")

    print("\n✅ Model OK!")
