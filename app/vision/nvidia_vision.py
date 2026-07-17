"""
app/vision/nvidia_vision.py

NVIDIA NIM Vision Client for Sanjeevani AI.

Sends images (molecular structures, chemical diagrams, handwritten notes,
scanned papers, photographs) to a multimodal NVIDIA NIM model via the
OpenAI-compatible chat completions endpoint and returns a structured
VisionExtractionResult with molecule identity + a ready-to-use RAG query.

Supported NVIDIA vision models (set NVIDIA_VISION_MODEL in .env):
  - nvidia/llama-3.2-90b-vision-instruct   ← default, best accuracy
  - nvidia/llama-3.2-11b-vision-instruct   ← faster / cheaper
  - microsoft/phi-3.5-vision-instruct
  - meta/llama-4-maverick-17b-128e-instruct

The NIM endpoint accepts images as base64 data-URLs inside the standard
OpenAI vision content block:
  {"type": "image_url", "image_url": {"url": "data:<mime>;base64,<data>"}}
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
from typing import Literal

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from app.config.settings import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output schema (shared with api/vision.py and schemas/vision.py)
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
    image_type:         str = Field(
        default="unknown",
        description="Type of image detected",
    )
    raw_description:    str = Field(
        default="",
        description="Full text description returned by the model",
    )
    research_query:     str = Field(
        default="",
        description="Suggested free-text query for the RAG pipeline",
    )
    confidence:         Literal["high", "medium", "low", "none"] = Field(default="none")
    warnings:           list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

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
- If multiple molecules are visible (e.g. a reaction scheme), focus on the PRODUCT or most prominent molecule.
- If the image is a scanned paper or handwritten note, extract any SMILES or structure you can identify.
- If you cannot identify any molecule, set all chemical fields to null, set confidence to "none", and explain in raw_description.
- Derive SMILES from the structure — do NOT guess if unsure; set to null instead.
- The research_query must be maximally useful for a pharmaceutical RAG retrieval system.
"""


# ---------------------------------------------------------------------------
# NVIDIA NIM Vision Client
# ---------------------------------------------------------------------------

_NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"


class NVIDIAVisionClient:
    """
    Vision client that sends images to NVIDIA NIM multimodal models.

    Uses the OpenAI-compatible `/chat/completions` endpoint with a
    base64 image_url content block — no extra SDK required beyond `openai`.
    """

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.nvidia_api_key:
            raise RuntimeError(
                "NVIDIA_API_KEY is not set. Vision extraction requires the NVIDIA NIM API."
            )
        self._client = AsyncOpenAI(
            api_key=settings.nvidia_api_key,
            base_url=_NVIDIA_NIM_BASE_URL,
        )
        self._model = settings.nvidia_vision_model
        self._max_tokens = settings.nvidia_vision_max_tokens
        self._temperature = settings.nvidia_vision_temperature
        logger.info("NVIDIAVisionClient initialised — model=%s", self._model)

    async def extract_from_bytes(
        self,
        image_bytes: bytes,
        mime_type: str = "image/jpeg",
    ) -> VisionExtractionResult:
        """
        Send raw image bytes to the NVIDIA NIM vision model and parse the response.

        Args:
            image_bytes: Raw bytes of the image file.
            mime_type:   MIME type string, e.g. "image/png".

        Returns:
            VisionExtractionResult with all extracted molecule fields.
        """
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{mime_type};base64,{b64}"

        logger.info(
            "Sending image to NVIDIA Vision (%s, %d bytes, model=%s) …",
            mime_type, len(image_bytes), self._model,
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _EXTRACTION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url,
                            # detail level: "high" for accurate SMILES derivation
                            "detail": "high",
                        },
                    },
                ],
            }
        ]

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
        except Exception as exc:
            logger.exception("NVIDIA Vision API call failed: %s", exc)
            return VisionExtractionResult(
                raw_description=f"NVIDIA Vision API error: {exc}",
                confidence="none",
                warnings=[str(exc)],
            )

        raw_text = (response.choices[0].message.content or "").strip()
        logger.debug("NVIDIA Vision raw response: %s", raw_text[:500])

        return self._parse_response(raw_text)

    async def extract_from_base64(
        self,
        b64_string: str,
        mime_type: str = "image/jpeg",
    ) -> VisionExtractionResult:
        """Convenience wrapper that accepts a base64-encoded image string."""
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
        """Parse JSON returned by the NIM model into a VisionExtractionResult."""
        # Strip markdown code fences if the model added them
        cleaned = re.sub(
            r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.MULTILINE
        ).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning("Could not parse NIM vision response as JSON: %s", exc)
            return VisionExtractionResult(
                raw_description=raw_text,
                confidence="none",
                warnings=[
                    f"JSON parse error: {exc}",
                    "Raw response stored in raw_description",
                ],
            )

        try:
            return VisionExtractionResult.model_validate(data)
        except Exception as exc:
            logger.warning("VisionExtractionResult validation error: %s", exc)
            return VisionExtractionResult(
                raw_description=data.get("raw_description", raw_text),
                research_query=data.get("research_query", ""),
                confidence="low",
                warnings=[f"Schema validation error: {exc}"],
            )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_vision_client: NVIDIAVisionClient | None = None


def get_vision_client() -> NVIDIAVisionClient:
    """Return the module-level NVIDIAVisionClient singleton."""
    global _vision_client
    if _vision_client is None:
        _vision_client = NVIDIAVisionClient()
    return _vision_client
