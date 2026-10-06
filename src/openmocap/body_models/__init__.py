"""Asset-independent body models and numerical SMPL-family linear blend skinning."""

from .model import BodyModel, ModelAssetError, load_body_model

__all__ = ["BodyModel", "ModelAssetError", "load_body_model"]
