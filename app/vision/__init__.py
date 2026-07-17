"""
app/vision/__init__.py

Vision package — NVIDIA NIM multimodal image extraction.
"""

from app.vision.nvidia_vision import (
    NVIDIAVisionClient,
    VisionExtractionResult,
    get_vision_client,
)

__all__ = ["NVIDIAVisionClient", "VisionExtractionResult", "get_vision_client"]
