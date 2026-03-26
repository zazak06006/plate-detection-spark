"""
Loss Functions pour SSD avec BoxCoder
- Focal Loss pour classification (gère le déséquilibre de classes)
- Smooth L1 Loss pour régression bbox (sur les deltas)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# Import du BoxCoder depuis model.py
from model import BOX_CODER


# ============================================================================
# FOCAL LOSS
# ============================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss pour classification.
    Réduit la contribution des exemples faciles pour se concentrer sur les difficiles.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha: Poids pour la classe positive (plates). Default 0.25.
            gamma: Facteur de focus. Default 2.0.
            reduction: 'mean', 'sum' ou 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [N, num_classes] - logits (avant softmax)
            targets: [N] - indices des classes (0 ou 1)

        Returns:
            loss: scalar ou [N] selon reduction
        """
        # Normaliser les targets pour éviter les valeurs négatives
        targets = targets.long().clamp(0, inputs.size(-1) - 1)

        # Cross entropy loss (sans réduction)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Probabilités
        p = F.softmax(inputs, dim=-1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weight
        alpha_t = torch.where(
            targets == 1,
            torch.tensor(self.alpha, device=inputs.device),
            torch.tensor(1 - self.alpha, device=inputs.device)
        )

        # Focal loss
        loss = alpha_t * focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ============================================================================
# SMOOTH L1 LOSS
# ============================================================================
class SmoothL1Loss(nn.Module):
    """
    Smooth L1 Loss pour régression bbox.
    Plus robuste aux outliers que MSE.
    """

    def __init__(self, beta: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [N, 4] - prédictions (cx, cy, w, h)
            targets: [N, 4] - ground truth

        Returns:
            loss: scalar
        """
        return F.smooth_l1_loss(inputs, targets, beta=self.beta, reduction=self.reduction)


# ============================================================================
# SSD COMBINED LOSS
# ============================================================================
class SSDLoss(nn.Module):
    """
    Loss combinée pour SSD:
    - Focal Loss pour classification
    - Smooth L1 pour régression

    Total Loss = cls_loss + reg_weight * reg_loss
    """

    def __init__(
        self,
        num_classes: int = 2,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reg_weight: float = 1.0,
        neg_pos_ratio: float = 3.0
    ):
        """
        Args:
            num_classes: Nombre de classes (2 = background + plate)
            alpha: Alpha pour Focal Loss
            gamma: Gamma pour Focal Loss
            reg_weight: Poids de la loss de régression
            neg_pos_ratio: Ratio négatifs/positifs pour hard negative mining
        """
        super().__init__()

        self.num_classes = num_classes
        self.reg_weight = reg_weight
        self.neg_pos_ratio = neg_pos_ratio

        self.cls_loss_fn = FocalLoss(alpha=alpha, gamma=gamma, reduction='none')
        self.reg_loss_fn = SmoothL1Loss(reduction='none')

    def forward(
        self,
        cls_preds: torch.Tensor,
        reg_preds: torch.Tensor,
        cls_targets: torch.Tensor,
        reg_targets: torch.Tensor,
        pos_mask: torch.Tensor,
        anchors
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            cls_preds: [B, num_anchors, num_classes] - logits
            reg_preds: [B, num_anchors, 4] - (cx, cy, w, h) normalisés
            cls_targets: [B, MAX_OBJECTS] - classe de chaque GT box
            reg_targets: [B, MAX_OBJECTS, 4] - coordonnées GT
            pos_mask: [B, MAX_OBJECTS] - 1 si objet valide, 0 sinon

        Returns:
            total_loss, cls_loss, reg_loss
        """
        B = cls_preds.size(0)
        num_anchors = cls_preds.size(1)
        device = cls_preds.device

        # Calculer le nombre de positifs par image
        num_pos = pos_mask.sum(dim=1).clamp(min=1)  # [B]

        # ============================================================
        # STRATÉGIE SIMPLIFIÉE: Matching direct
        # ============================================================
        # On va matcher chaque GT box au meilleur anchor basé sur l'IoU
        # Approche simplifiée: on utilise les prédictions directement

        total_cls_loss = torch.tensor(0.0, device=device)
        total_reg_loss = torch.tensor(0.0, device=device)

        for b in range(B):
            # Targets pour cette image
            gt_classes = cls_targets[b]  # [MAX_OBJECTS]
            gt_boxes = reg_targets[b]     # [MAX_OBJECTS, 4]
            mask = pos_mask[b]             # [MAX_OBJECTS]

            # Nombre de GT valides
            n_gt = int(mask.sum().item())

            if n_gt == 0:
                # Pas d'objets: tous les anchors sont négatifs
                anchor_targets = torch.zeros(num_anchors, dtype=torch.long, device=device)
                cls_loss_b = self.cls_loss_fn(cls_preds[b], anchor_targets)
                total_cls_loss = total_cls_loss + cls_loss_b.mean()
                continue

            # GT valides seulement
            gt_boxes_valid = gt_boxes[mask.bool()]  # [n_gt, 4]
            gt_classes_valid = gt_classes[mask.bool()]  # [n_gt]

            # Pour chaque anchor, trouver la GT box la plus proche
            # Calcul IoU entre prédictions et GT
            pred_boxes = reg_preds[b]  # [num_anchors, 4]

            # Calculer IoU (format cx,cy,w,h)
            #ious = box_iou_cxcywh(pred_boxes, gt_boxes_valid) 
            ious = box_iou_cxcywh(anchors, gt_boxes_valid) # [num_anchors, n_gt]

            # Pour chaque anchor, trouver le meilleur GT
            best_iou, best_gt_idx = ious.max(dim=1)  # [num_anchors]

            # Seuil IoU pour positif
            pos_threshold = 0.5
            neg_threshold = 0.3

            # Masque positif/négatif
            pos_anchors = best_iou > pos_threshold
            neg_anchors = best_iou < neg_threshold

            # Targets pour classification
            anchor_targets = torch.zeros(num_anchors, dtype=torch.long, device=device)
            anchor_targets[pos_anchors] = 1  # Classe plate

            # ============================================================
            # CLASSIFICATION LOSS avec Hard Negative Mining
            # ============================================================
            cls_loss_all = self.cls_loss_fn(cls_preds[b], anchor_targets)  # [num_anchors]

            # Positifs
            pos_cls_loss = cls_loss_all[pos_anchors]

            # Hard Negative Mining: garder les négatifs les plus difficiles
            n_pos = pos_anchors.sum().item()
            n_neg = int(min(neg_anchors.sum().item(), self.neg_pos_ratio * max(n_pos, 1)))

            if n_neg > 0 and neg_anchors.sum() > 0:
                neg_cls_loss = cls_loss_all[neg_anchors]
                neg_cls_loss_sorted, _ = neg_cls_loss.sort(descending=True)
                neg_cls_loss = neg_cls_loss_sorted[:n_neg]
            else:
                neg_cls_loss = torch.tensor(0.0, device=device)

            # Combiner
            if n_pos > 0:
                cls_loss_b = (pos_cls_loss.sum() + neg_cls_loss.sum()) / max(n_pos, 1)
            else:
                cls_loss_b = neg_cls_loss.mean() if isinstance(neg_cls_loss, torch.Tensor) and neg_cls_loss.numel() > 0 else torch.tensor(0.0, device=device)

            total_cls_loss = total_cls_loss + cls_loss_b

            # ============================================================
            # REGRESSION LOSS (seulement sur positifs) - avec BoxCoder
            # ============================================================
            if pos_anchors.sum() > 0:
                # Deltas prédits pour les positifs
                pred_deltas_pos = pred_boxes[pos_anchors]  # [n_pos, 4] - ce sont des deltas maintenant

                # Anchors correspondants aux positifs
                anchors_pos = anchors[pos_anchors]  # [n_pos, 4]

                # GT correspondantes
                gt_idx_pos = best_gt_idx[pos_anchors]
                gt_boxes_pos = gt_boxes_valid[gt_idx_pos]  # [n_pos, 4]

                # Encoder les GT en deltas par rapport aux anchors
                gt_deltas_pos = BOX_CODER.encode(gt_boxes_pos, anchors_pos)  # [n_pos, 4]

                # Loss de régression sur les deltas
                reg_loss_b = self.reg_loss_fn(pred_deltas_pos, gt_deltas_pos)
                reg_loss_b = reg_loss_b.sum() / max(n_pos, 1)
            else:
                reg_loss_b = torch.tensor(0.0, device=device)

            total_reg_loss = total_reg_loss + reg_loss_b

        # Moyenne sur le batch
        total_cls_loss = total_cls_loss / B
        total_reg_loss = total_reg_loss / B

        # Loss totale
        total_loss = total_cls_loss + self.reg_weight * total_reg_loss

        return total_loss, total_cls_loss, total_reg_loss


# ============================================================================
# UTILITY: IoU pour format cx,cy,w,h
# ============================================================================
def box_iou_cxcywh(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Calcule l'IoU entre deux ensembles de boxes en format (cx, cy, w, h).

    Args:
        boxes1: [N, 4]
        boxes2: [M, 4]

    Returns:
        iou: [N, M]
    """
    # Convertir en x1,y1,x2,y2
    boxes1_xyxy = cxcywh_to_xyxy(boxes1)
    boxes2_xyxy = cxcywh_to_xyxy(boxes2)

    # Aires
    area1 = (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]) * (boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1])
    area2 = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]) * (boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1])

    # Intersection
    lt = torch.max(boxes1_xyxy[:, None, :2], boxes2_xyxy[None, :, :2])  # [N, M, 2]
    rb = torch.min(boxes1_xyxy[:, None, 2:], boxes2_xyxy[None, :, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    # Union
    union = area1[:, None] + area2[None, :] - inter

    # IoU
    iou = inter / union.clamp(min=1e-6)

    return iou


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convertit (cx, cy, w, h) -> (x1, y1, x2, y2)"""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


# ============================================================================
# FACTORY
# ============================================================================
def create_ssd_loss(
    num_classes: int = 2,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reg_weight: float = 1.0
) -> SSDLoss:
    """
    Crée la fonction de loss SSD.

    Args:
        num_classes: Nombre de classes
        alpha: Focal Loss alpha
        gamma: Focal Loss gamma
        reg_weight: Poids de la loss de régression

    Returns:
        SSDLoss instance
    """
    return SSDLoss(
        num_classes=num_classes,
        alpha=alpha,
        gamma=gamma,
        reg_weight=reg_weight
    )


# ============================================================================
# TEST
# ============================================================================
if __name__ == "__main__":
    print("🧪 Testing SSD Loss with BoxCoder...")

    # Créer la loss
    criterion = create_ssd_loss()

    # Données factices (avec requires_grad pour tester backward)
    B = 2
    num_anchors = 1344
    num_classes = 2
    max_objects = 20

    # Anchors valides (format cx, cy, w, h)
    # Simuler des anchors sur une grille
    anchors = torch.zeros(num_anchors, 4)
    anchors[:, 0] = torch.linspace(0.1, 0.9, num_anchors)  # cx
    anchors[:, 1] = torch.linspace(0.1, 0.9, num_anchors)  # cy
    anchors[:, 2] = 0.1  # w (10% de l'image)
    anchors[:, 3] = 0.1  # h

    cls_preds = torch.randn(B, num_anchors, num_classes, requires_grad=True)
    # reg_preds sont maintenant des deltas bruts (pas de sigmoid)
    reg_preds = torch.randn(B, num_anchors, 4, requires_grad=True)

    cls_targets = torch.zeros(B, max_objects)
    cls_targets[0, 0] = 1  # Une plaque dans image 0
    cls_targets[1, :2] = 1  # Deux plaques dans image 1

    reg_targets = torch.rand(B, max_objects, 4) * 0.5 + 0.25  # Boxes au centre

    pos_mask = torch.zeros(B, max_objects)
    pos_mask[0, 0] = 1
    pos_mask[1, :2] = 1

    # Calculer la loss
    total_loss, cls_loss, reg_loss = criterion(
        cls_preds, reg_preds, cls_targets, reg_targets, pos_mask, anchors
    )

    print(f"\n📊 Loss values:")
    print(f"   Total: {total_loss.item():.4f}")
    print(f"   Classification: {cls_loss.item():.4f}")
    print(f"   Regression: {reg_loss.item():.4f}")

    # Vérifier que les gradients passent
    total_loss.backward()
    print(f"\n✅ Gradients OK!")

    print("\n✅ Loss OK!")
