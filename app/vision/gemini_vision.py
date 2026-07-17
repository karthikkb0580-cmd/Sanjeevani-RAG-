"""
app/vision/gemini_vision.py

Gemini Vision Client for Sanjeevani AI.

Accepts raw image bytes (JPEG / PNG / WEBP / GIF) or a base64-encoded string
and returns a structured VisionExtractionResult describing the molecule(s)
depicted in the image.

Supported image types:
  - Molecular structure diagrams (2-D skeletal / 3-D ball-and-stick)
  - Chemical reaction schemes
  - Handwritten structural formulas or notes
  - Photographs of physical models
  - Scanned pages from papers / textbooks

Uses the google-generativeai SDK which is already in requirements.txt for
Gemini embeddings, so no new dependency is needed.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
from pathlib import Path
from typing import Literal

import google.generativeai as genai
from pydantic import BaseModel, Field

from app.config.settings import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class VisionExtractionResult(BaseModel):
    """Structured molecule information extracted from an image."""

    # Core identification
    molecule_name:      str | None = Field(default=None, description="IUPAC or common name")
    smiles:             str | None = Field(default=None, description="SMILES string")
    molecular_formula:  str | None = Field(default=None, description="Molecular formula, e.g. C6H6")
    cas_number:         str | None = Field(default=None, description="CAS registry number if visible")
    inchi:              str | None = Field(default=None, description="InChI string if derivable")

    # Context
    image_type:         str        = Field(default="unknown",
                                           description="Type of image detected")
    raw_description:    str        = Field(default="",
                                           description="Full text description returned by Gemini")
    research_query:     str        = Field(default="",
                                           description="Suggested free-text query for the RAG pipeline")
    confidence:         Literal["high", "medium", "low", "none"] = Field(default="none")
    warnings:           list[str]  = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Gemini Vision Client
# ---------------------------------------------------------------------------

# Detailed system prompt for chemical image extraction
_EXTRACTION_PROMPT = """\
You are an expert chemical image analyst and cheminformatics assistant integrated into the Sanjeevani pharmaceutical research platform.

Analyse the provided image and extract all chemical / molecular information. Return ONLY a JSON object (no markdown fences, no extra text) with these exact keys:

{
  "molecule_name": "<IUPAC or most common name, or null>",
  "smiles": "<canonical SMILES string derived from the structure, or null>",
  "molecular_formula": "<e.g. C9H8O4, or null>",
  "cas_number": "<CAS number if shown in the image, or null>",
  "inchi": "<InChI string if derivable, or null>",
  "image_type": "<one of: skeletal_structure | reaction_scheme | 3d_model | handwritten | scanned_paper | photograph | unknown>",
  "raw_description": "<1-3 sentence plain-English description of what the image shows>",
  "research_query": "<a rich, precise free-text query suitable for searching a pharmaceutical literature database about this molecule — include name, functional groups, class, biological target if visible>",
  "confidence": "<high | medium | low | none>",
  "warnings": ["<any ambiguities, multiple molecules, poor image quality, etc.>"]
}

Rules:
- If multiple molecules are visible (e.g. a reaction scheme), focus on the PRODUCT or the most prominent molecule.
- If the image is a scanned paper or handwritten note, extract any SMILES or structure you can identify.
- If you cannot identify any molecule, set all chemical fields to null, set confidence to "none", and explain in raw_description.
- Derive SMILES from the structure — do NOT guess if you are unsure; set to null instead.
- The research_query must be useful to a RAG retrieval system for pharmaceutical research.
"""


class GeminiVisionClient:
    """
    Thin wrapper around the Gemini Vision API for chemical image extraction.

    The client is cheap to instantiate and safe to reuse across requests.
    """

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Vision extraction requires the Gemini API."
            )
        genai.configure(api_key=settings.gemini_api_key)
        self._model_name = settings.gemini_vision_model
        self._model = genai.GenerativeModel(self._model_name)
        logger.info("GeminiVisionClient initialised with model=%s", self._model_name)

    async def extract_from_bytes(
        self,
        image_bytes: bytes,
        mime_type: str = "image/jpeg",
    ) -> VisionExtractionResult:
        """
        Send raw image bytes to Gemini Vision and parse the response.

        Args:
            image_bytes: Raw bytes of the image file.
            mime_type:   MIME type string, e.g. "image/png".

        Returns:
            VisionExtractionResult with all extracted fields.
        """
        import asyncio
        import json

        logger.info(
            "Sending image to Gemini Vision (%s, %d bytes) …",
            mime_type, len(image_bytes),
        )

        # Build the multimodal request parts
        image_part = {
            "mime_type": mime_type,
            "data": image_bytes,
        }

        try:
            # Gemini SDK is synchronous — run in executor to avoid blocking the loop
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._model.generate_content(
                    [_EXTRACTION_PROMPT, image_part],
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=1024,
                    ),
                ),
            )
        except Exception as exc:
            logger.exception("Gemini Vision API call failed: %s", exc)
            return VisionExtractionResult(
                raw_description=f"Gemini Vision API error: {exc}",
                confidence="none",
                warnings=[str(exc)],
            )

        raw_text = response.text.strip()
        logger.debug("Gemini Vision raw response: %s", raw_text[:500])

        return self._parse_response(raw_text)

    async def extract_from_base64(
        self,
        b64_string: str,
        mime_type: str = "image/jpeg",
    ) -> VisionExtractionResult:
        """Convenience wrapper accepting a base64-encoded image string."""
        try:
            image_bytes = base64.b64decode(b64_string)
        except Exception as exc:
            return VisionExtractionResult(
                raw_description=f"Invalid base64 image: {exc}",
                confidence="none",
                warnings=[f"base64 decode error: {exc}"],
            )
        return await self.extract_from_bytes(image_bytes, mime_type)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(raw_text: str) -> VisionExtractionResult:
        """Parse the JSON returned by Gemini into a VisionExtractionResult."""
        import json, re

        # Strip possible markdown fences Gemini sometimes adds
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.MULTILINE).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning("Could not parse Gemini response as JSON: %s", exc)
            return VisionExtractionResult(
                raw_description=raw_text,
                confidence="none",
                warnings=[f"JSON parse error: {exc}", "Raw response stored in raw_description"],
            )

        # Validate/coerce with Pydantic
        try:
            return VisionExtractionResult.model_validate(data)
        except Exception as exc:
            logger.warning("VisionExtractionResult validation failed: %s", exc)
            return VisionExtractionResult(
                raw_description=data.get("raw_description", raw_text),
                research_query=data.get("research_query", ""),
                confidence="low",
                warnings=[f"Schema validation error: {exc}"],
            )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_vision_client: GeminiVisionClient | None = None


def get_vision_client() -> GeminiVisionClient:
    """Return the module-level GeminiVisionClient singleton."""
    global _vision_client
    if _vision_client is None:
        _vision_client = GeminiVisionClient()
    return _vision_client
