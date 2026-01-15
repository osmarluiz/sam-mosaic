"""SAM2 predictor wrapper for segmentation."""

from pathlib import Path
from typing import Optional, List
import numpy as np
import torch

from sam_mosaic.sam.masks import Mask


class SAMPredictor:
    """Wrapper for SAM2 model predictions.

    This class handles model loading, image encoding, and batched
    point-based predictions for segmentation.

    Attributes:
        checkpoint_path: Path to SAM2 checkpoint.
        device: Torch device (cuda/cpu).
        model: Loaded SAM2 model.
    """

    # SAM2 config file mapping based on checkpoint name
    SAM2_CONFIGS = {
        "large": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    }

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        model_type: str = "large"
    ):
        """Initialize SAM2 predictor.

        Args:
            checkpoint_path: Path to SAM2 checkpoint file.
            device: Device to use ('cuda', 'cpu', or None for auto).
            model_type: SAM2 model type (large, base_plus, small, tiny).
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.model_cfg = self.SAM2_CONFIGS.get(model_type, model_type)

        self._model = None
        self._predictor = None
        self._current_image = None

    def load_model(self) -> None:
        """Load SAM2 model into memory.

        Raises:
            ImportError: If sam2 package is not installed.
            FileNotFoundError: If checkpoint file doesn't exist.
        """
        if self._model is not None:
            return

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            self._model = build_sam2(
                self.model_cfg,
                str(self.checkpoint_path),
                device=self.device
            )
            self._predictor = SAM2ImagePredictor(self._model)

        except ImportError as e:
            raise ImportError(
                "SAM2 package not found. Please install it from: "
                "https://github.com/facebookresearch/sam2"
            ) from e

    def set_image(self, image: np.ndarray) -> None:
        """Set the image for prediction.

        Args:
            image: RGB image array of shape (H, W, 3) with dtype uint8.

        Raises:
            ValueError: If image format is invalid.
        """
        if self._predictor is None:
            self.load_model()

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB image (H, W, 3), got shape {image.shape}")

        if image.dtype != np.uint8:
            raise ValueError(f"Expected uint8 image, got {image.dtype}")

        self._predictor.set_image(image)
        self._current_image = image

    def predict_points_batched(
        self,
        image: np.ndarray,
        points: np.ndarray,
        iou_threshold: float = 0.88,
        stability_threshold: float = 0.92,
        min_mask_area: int = 100,
        box_nms_thresh: float = 0.7
    ) -> List[Mask]:
        """Predict masks using batched AutomaticMaskGenerator.

        This is MUCH faster than point-by-point prediction because it
        processes all points in a single batch using SAM2AutomaticMaskGenerator.

        Args:
            image: RGB image array (H, W, 3).
            points: Array of shape (N, 2) with (x, y) pixel coordinates.
            iou_threshold: Minimum predicted IoU to accept mask.
            stability_threshold: Minimum stability score to accept mask.
            min_mask_area: Minimum mask area in pixels.
            box_nms_thresh: Box NMS threshold for overlapping masks.

        Returns:
            List of Mask objects that passed thresholds.
        """
        if self._model is None:
            self.load_model()

        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        h, w = image.shape[:2]

        # Convert pixel coords to normalized coords (0-1)
        points_norm = points.astype(np.float32).copy()
        points_norm[:, 0] /= w  # x
        points_norm[:, 1] /= h  # y

        # Create generator with custom point grid
        generator = SAM2AutomaticMaskGenerator(
            model=self._model,
            points_per_side=None,  # Disable automatic grid
            point_grids=[points_norm],  # Use custom points
            pred_iou_thresh=iou_threshold,
            stability_score_thresh=stability_threshold,
            min_mask_region_area=min_mask_area,
            box_nms_thresh=box_nms_thresh,
        )

        # Generate masks in one batched call
        mask_outputs = generator.generate(image)

        # Free generator memory immediately (prevents memory leak across tiles/passes)
        del generator

        # Convert to our Mask format
        masks = []
        for mask_data in mask_outputs:
            m = mask_data['segmentation']
            masks.append(Mask(
                data=m.astype(np.uint8),
                score=float(mask_data.get('predicted_iou', 0.0)),
                stability=float(mask_data.get('stability_score', 0.0)),
                point=None  # Batched - no single point association
            ))

        return masks

    def reset_image(self) -> None:
        """Clear the current image from memory."""
        self._current_image = None
        if self._predictor is not None:
            self._predictor.reset_predictor()

    def unload_model(self) -> None:
        """Unload model from memory."""
        self._model = None
        self._predictor = None
        self._current_image = None
        torch.cuda.empty_cache()

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    @property
    def has_image(self) -> bool:
        """Check if an image is currently set."""
        return self._current_image is not None
