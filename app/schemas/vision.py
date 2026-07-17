"""
app/schemas/vision.py

Request / response schemas for the vision-capable multipart endpoint.

POST /api/v2/analyze/multipart
  - Accepts: multipart/form-data
      · image file  (JPEG / PNG / WEBP / GIF / BMP / TIFF)
      · document    (PDF / DOCX / DOC)
      · plain text  string
  - Returns: VisionAnalysisResponse  (extraction result + full FinalAnalysisReport)
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.analysis import FinalAnalysisReport
from app.vision.document_extractor import DocumentExtractionResult
from app.vision.nvidia_vision import VisionExtractionResult


# ---------------------------------------------------------------------------
# Intermediate / diagnostic
# ---------------------------------------------------------------------------

class VisionRoutingDecision(BaseModel):
    """Records how the endpoint routed the incoming request."""
    input_mode:     Literal["image", "document", "text"] = Field(
        ...,
        description="How the request was routed: image → NVIDIAVision | document → text extractor | text → direct",
    )
    filename:       str | None = Field(default=None)
    mime_type:      str | None = Field(default=None)
    file_size_kb:   float | None = Field(default=None)
    extracted_query: str = Field(
        default="",
        description="The effective query sent to the orchestrator",
    )


# ---------------------------------------------------------------------------
# Full vision analysis response
# ---------------------------------------------------------------------------

class VisionAnalysisResponse(BaseModel):
    """
    Full response from POST /api/v2/analyze/multipart.

    Contains:
      - routing_decision:   how the input was interpreted
      - vision_extraction:  Gemini/NVIDIA Vision result (image path only)
      - document_extraction: PDF/DOCX text extraction result (document path only)
      - report:             the complete multi-agent analysis report
    """
    routing_decision:     VisionRoutingDecision
    vision_extraction:    VisionExtractionResult | None = Field(
        default=None,
        description="Molecule extraction result from NVIDIA Vision (image input only)",
    )
    document_extraction:  DocumentExtractionResult | None = Field(
        default=None,
        description="Text extraction result from PDF/DOCX (document input only)",
    )
    report:               FinalAnalysisReport
