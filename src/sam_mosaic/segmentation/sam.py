"""SAM model wrapper and mask generation."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import torch


@dataclass
class Mask:
    """A single segmentation mask."""
    mask: np.ndarray          # Binary mask (H, W)
    score: float              # Predicted IoU
    stability: float          # Stability score
    area: int                 # Mask area in pixels
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    point: Optional[Tuple[int, int]] = None  # Prompt point if any


class SAMPredictor:
    """Wrapper for SAM model."""

    # SAM2 config file mapping
    SAM2_CONFIGS = {
        "large": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    }

    def __init__(
        self,
        checkpoint: str,
        model_type: str = "large",
        device: str = "cuda"
    ):
        """Initialize SAM predictor.

        Args:
            checkpoint: Path to SAM checkpoint file.
            model_type: Model type. For SAM2: large, base_plus, small, tiny.
                        For SAM1: vit_h, vit_l, vit_b.
            device: Device to run on (cuda, cpu).
        """
        self.device = device
        self.model_type = model_type

        # Import SAM here to allow installation check
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            self._sam_version = 2
        except ImportError:
            from segment_anything import sam_model_registry, SamPredictor as _SamPredictor
            from segment_anything import SamAutomaticMaskGenerator as _SamAMG
            self._sam_version = 1

        if self._sam_version == 2:
            # SAM 2 initialization - needs config yaml path
            config_path = self.SAM2_CONFIGS.get(model_type, model_type)
            self.model = build_sam2(config_path, checkpoint, device=device)
            self.predictor = SAM2ImagePredictor(self.model)
            self._amg_class = SAM2AutomaticMaskGenerator
        else:
            # SAM 1 initialization
            self.model = sam_model_registry[model_type](checkpoint=checkpoint)
            self.model.to(device)
            self.predictor = _SamPredictor(self.model)
            self._amg_class = _SamAMG

        self._current_image = None

    def set_image(self, image: np.ndarray) -> None:
        """Set image for prediction.

        Args:
            image: RGB image array (H, W, 3) with values 0-255.
        """
        self.predictor.set_image(image)
        self._current_image = image

    def predict_point(
        self,
        point: Tuple[int, int],
        iou_thresh: float = 0.88,
        stability_thresh: float = 0.92
    ) -> Optional[Mask]:
        """Predict mask from a single point prompt.

        Args:
            point: (x, y) coordinate.
            iou_thresh: Minimum predicted IoU.
            stability_thresh: Minimum stability score.

        Returns:
            Best mask meeting thresholds, or None.
        """
        point_coords = np.array([[point[0], point[1]]])
        point_labels = np.array([1])  # foreground

        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
        )

        # Find best mask meeting thresholds
        best_mask = None
        best_score = -1

        for i, (m, s) in enumerate(zip(masks, scores)):
            # Calculate stability
            stability = self._calc_stability(logits[i])

            if s >= iou_thresh and stability >= stability_thresh:
                if s > best_score:
                    best_score = s
                    best_mask = Mask(
                        mask=m.astype(np.uint8),
                        score=float(s),
                        stability=stability,
                        area=int(m.sum()),
                        bbox=self._mask_to_bbox(m),
                        point=point
                    )

        return best_mask

    def predict_points(
        self,
        points: np.ndarray,
        iou_thresh: float = 0.88,
        stability_thresh: float = 0.92
    ) -> List[Mask]:
        """Predict masks from multiple points (per-point, slow).

        Args:
            points: Array of shape (N, 2) with (x, y) coordinates.
            iou_thresh: Minimum predicted IoU.
            stability_thresh: Minimum stability score.

        Returns:
            List of masks meeting thresholds.
        """
        masks = []
        for point in points:
            mask = self.predict_point(
                tuple(point),
                iou_thresh=iou_thresh,
                stability_thresh=stability_thresh
            )
            if mask is not None:
                masks.append(mask)
        return masks

    def predict_points_batched(
        self,
        image: np.ndarray,
        points: np.ndarray,
        iou_thresh: float = 0.88,
        stability_thresh: float = 0.92,
        min_mask_area: int = 100,
        box_nms_thresh: float = 0.7
    ) -> List[Mask]:
        """Predict masks from points using batched AutomaticMaskGenerator.

        This is MUCH faster than predict_points() because it processes
        all points in a single batch instead of one-by-one.

        Args:
            image: RGB image array (H, W, 3).
            points: Array of shape (N, 2) with (x, y) pixel coordinates.
            iou_thresh: Minimum predicted IoU.
            stability_thresh: Minimum stability score.
            min_mask_area: Minimum mask area in pixels.
            box_nms_thresh: Box NMS threshold.

        Returns:
            List of masks meeting thresholds.
        """
        h, w = image.shape[:2]

        # Convert pixel coords to normalized coords (0-1)
        points_norm = points.astype(np.float32).copy()
        points_norm[:, 0] /= w  # x
        points_norm[:, 1] /= h  # y

        # Create generator with custom point grid
        generator = self._amg_class(
            model=self.model,
            points_per_side=None,  # Disable automatic grid
            point_grids=[points_norm],  # Use custom points
            pred_iou_thresh=iou_thresh,
            stability_score_thresh=stability_thresh,
            min_mask_region_area=min_mask_area,
            box_nms_thresh=box_nms_thresh,
        )

        # Generate masks in one batched call
        mask_outputs = generator.generate(image)

        # Convert to our Mask format
        masks = []
        for mask_data in mask_outputs:
            m = mask_data['segmentation']
            masks.append(Mask(
                mask=m.astype(np.uint8),
                score=float(mask_data.get('predicted_iou', 0.0)),
                stability=float(mask_data.get('stability_score', 0.0)),
                area=int(mask_data.get('area', m.sum())),
                bbox=tuple(mask_data.get('bbox', self._mask_to_bbox(m))),
                point=None  # Batched - no single point association
            ))

        return masks

    def _calc_stability(self, logits: np.ndarray) -> float:
        """Calculate mask stability score."""
        high = (logits > 0.5).sum()
        low = (logits > -0.5).sum()
        if low == 0:
            return 0.0
        return float(high / low)

    def _mask_to_bbox(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Get bounding box from mask."""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not rows.any():
            return (0, 0, 0, 0)

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        return (int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1))


def generate_masks(
    predictor: SAMPredictor,
    image: np.ndarray,
    points: np.ndarray,
    iou_thresh: float = 0.88,
    stability_thresh: float = 0.92
) -> List[Mask]:
    """Generate masks from point grid.

    Args:
        predictor: SAM predictor instance.
        image: RGB image array (H, W, 3).
        points: Point coordinates (N, 2).
        iou_thresh: Minimum predicted IoU.
        stability_thresh: Minimum stability score.

    Returns:
        List of valid masks.
    """
    predictor.set_image(image)
    return predictor.predict_points(
        points,
        iou_thresh=iou_thresh,
        stability_thresh=stability_thresh
    )


def apply_black_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply black masking to already-segmented regions.

    Args:
        image: RGB image array (H, W, 3).
        mask: Binary mask (H, W) where non-zero = already segmented.

    Returns:
        Image with masked regions set to black.
    """
    result = image.copy()
    result[mask > 0] = 0
    return result
