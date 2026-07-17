"""
app/api/vision.py

Vision-Capable Multipart Analysis Endpoint
POST /api/v2/analyze/multipart

Accepts EITHER:
  A) An image file (photo of molecule structure, chemical diagram,
     handwritten note, or paper page) via multipart/form-data  field `image`
  B) A plain-text string via form field `text`

Plus optional override fields (all mirroring AnalysisRequest):
  - top_k, similarity_threshold, use_mmr, mmr_lambda
  - agents_enabled (comma-separated agent names)
  - stream (bool)

Routing logic:
  1. If `image` is present  → GeminiVisionClient extracts molecule info
                              → builds AnalysisRequest from extraction
  2. If `text` is present   → builds AnalysisRequest directly
  3. If both are present    → image takes precedence; text is appended to query
  4. If neither             → HTTP 422

The resulting AnalysisRequest is handed to the shared AnalysisOrchestrator.
"""

from __future__ import annotations

import logging
import mimetypes
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse, StreamingResponse

from app.orchestrator.orchestrator import AnalysisOrchestrator, get_orchestrator
from app.schemas.analysis import AnalysisRequest, MoleculeInput
from app.schemas.vision import VisionAnalysisResponse, VisionRoutingDecision
from app.vision.nvidia_vision import VisionExtractionResult, get_vision_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2", tags=["Vision · Multipart Analysis"])

# Accepted image MIME types
_ALLOWED_IMAGE_MIMES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/bmp",
    "image/tiff",
}

# Max image upload size: 20 MB
_MAX_IMAGE_BYTES = 20 * 1024 * 1024


# ---------------------------------------------------------------------------
# POST /api/v2/analyze/multipart
# ---------------------------------------------------------------------------

@router.post(
    "/analyze/multipart",
    response_model=VisionAnalysisResponse,
    summary="Vision-Capable Multipart Analysis",
    description=(
        "Accepts an **image** (molecular structure, chemical diagram, handwritten note, "
        "or paper scan) **or** a plain **text** string, then automatically routes to the "
        "multi-agent orchestrator.\n\n"
        "- **Image path**: Gemini Vision extracts SMILES, name, formula → orchestrator.\n"
        "- **Text path**: query is passed directly → orchestrator.\n"
        "- Both fields: image takes precedence; text is appended to the derived query.\n\n"
        "Set `stream=true` to receive a Server-Sent Events stream instead of JSON."
    ),
    status_code=status.HTTP_200_OK,
)
async def analyze_multipart(
    # ── Image upload (optional) ─────────────────────────────────────────────
    image: Annotated[
        UploadFile | None,
        File(description="Image of a molecule structure, diagram, or handwritten note"),
    ] = None,

    # ── Text input (optional) ───────────────────────────────────────────────
    text: Annotated[
        str | None,
        Form(description="Free-text query, molecule name, SMILES, or research question"),
    ] = None,

    # ── Retrieval overrides ─────────────────────────────────────────────────
    top_k: Annotated[
        int,
        Form(description="Number of chunks to retrieve (1-50)", ge=1, le=50),
    ] = 10,

    similarity_threshold: Annotated[
        float,
        Form(description="Minimum cosine similarity (0-1)", ge=0.0, le=1.0),
    ] = 0.60,

    use_mmr: Annotated[
        bool,
        Form(description="Enable Maximal Marginal Relevance reranking"),
    ] = True,

    mmr_lambda: Annotated[
        float,
        Form(description="MMR diversity weight (0=diverse, 1=relevant)", ge=0.0, le=1.0),
    ] = 0.5,

    # ── Agent / streaming control ───────────────────────────────────────────
    agents_enabled: Annotated[
        str | None,
        Form(description="Comma-separated agent names to run (all agents if omitted)"),
    ] = None,

    stream: Annotated[
        bool,
        Form(description="Return a Server-Sent Events stream instead of JSON"),
    ] = False,

    # ── Injected dependencies ───────────────────────────────────────────────
    orchestrator: AnalysisOrchestrator = Depends(get_orchestrator),
) -> VisionAnalysisResponse | StreamingResponse:
    """
    Vision-capable multipart analysis endpoint.

    Routes the incoming payload to Gemini Vision (image) or directly to the
    orchestrator (text), then returns a VisionAnalysisResponse containing both
    the extraction result and the full multi-agent report.
    """

    # ── 1. Validate: at least one of image / text must be present ────────────
    if image is None and (text is None or not text.strip()):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Provide either an 'image' file or a non-empty 'text' field.",
        )

    # ── 2. Parse agents_enabled ──────────────────────────────────────────────
    enabled_agents: list[str] | None = None
    if agents_enabled:
        enabled_agents = [a.strip() for a in agents_enabled.split(",") if a.strip()]

    # ── 3. Route: IMAGE path ─────────────────────────────────────────────────
    vision_extraction: VisionExtractionResult | None = None
    routing: VisionRoutingDecision

    if image is not None:
        # Validate MIME type
        content_type = image.content_type or "application/octet-stream"
        # Fallback: infer from filename
        if content_type == "application/octet-stream" and image.filename:
            guessed, _ = mimetypes.guess_type(image.filename)
            if guessed:
                content_type = guessed

        if content_type not in _ALLOWED_IMAGE_MIMES:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=(
                    f"Unsupported image type '{content_type}'. "
                    f"Accepted: {sorted(_ALLOWED_IMAGE_MIMES)}"
                ),
            )

        # Read and size-check
        image_bytes = await image.read()
        if len(image_bytes) > _MAX_IMAGE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Image exceeds the 20 MB limit ({len(image_bytes) / 1e6:.1f} MB).",
            )

        logger.info(
            "Vision multipart: image='%s' mime=%s size=%.1f KB",
            image.filename, content_type, len(image_bytes) / 1024,
        )

        # Call Gemini Vision
        try:
            vision_client = get_vision_client()
            vision_extraction = await vision_client.extract_from_bytes(image_bytes, content_type)
        except RuntimeError as exc:
            # NVIDIA_API_KEY not configured → degrade gracefully if text is also present
            logger.error("Vision client unavailable: %s", exc)
            if not text:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Vision service unavailable and no text fallback provided: {exc}",
                )
            # Fall through to text path below
            vision_extraction = None

        # Build query from extraction
        if vision_extraction is not None:
            query_parts: list[str] = []

            if vision_extraction.research_query:
                query_parts.append(vision_extraction.research_query)
            elif vision_extraction.raw_description:
                query_parts.append(vision_extraction.raw_description)

            # Append explicit text if also provided
            if text and text.strip():
                query_parts.append(text.strip())

            effective_query = " | ".join(query_parts) if query_parts else (text or "")

            # Build MoleculeInput from extraction
            molecule: MoleculeInput | None = None
            if any([
                vision_extraction.molecule_name,
                vision_extraction.smiles,
                vision_extraction.cas_number,
            ]):
                molecule = MoleculeInput(
                    name=vision_extraction.molecule_name,
                    smiles=vision_extraction.smiles,
                    cas=vision_extraction.cas_number,
                    inchi=vision_extraction.inchi,
                )

            analysis_request = AnalysisRequest(
                query=effective_query or None,
                molecule=molecule,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                use_mmr=use_mmr,
                mmr_lambda=mmr_lambda,
                agents_enabled=enabled_agents,
                stream=stream,
            )

            routing = VisionRoutingDecision(
                input_mode="image",
                image_filename=image.filename,
                image_mime=content_type,
                image_size_kb=round(len(image_bytes) / 1024, 2),
                extracted_query=effective_query,
            )

        else:
            # Vision client failed, fall through to text
            routing = VisionRoutingDecision(
                input_mode="text",
                extracted_query=text or "",
            )
            analysis_request = AnalysisRequest(
                query=text,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                use_mmr=use_mmr,
                mmr_lambda=mmr_lambda,
                agents_enabled=enabled_agents,
                stream=stream,
            )

    # ── 4. Route: TEXT path ──────────────────────────────────────────────────
    else:
        logger.info("Vision multipart: text-only path, query='%s…'", (text or "")[:80])
        routing = VisionRoutingDecision(
            input_mode="text",
            extracted_query=text or "",
        )
        analysis_request = AnalysisRequest(
            query=text,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            use_mmr=use_mmr,
            mmr_lambda=mmr_lambda,
            agents_enabled=enabled_agents,
            stream=stream,
        )

    # ── 5. Validate orchestrator input ───────────────────────────────────────
    try:
        _ = analysis_request.effective_query()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    # ── 6. Streaming path ────────────────────────────────────────────────────
    if stream:
        logger.info("Vision multipart: streaming SSE response")
        return StreamingResponse(
            orchestrator.stream_analyze(analysis_request),
            media_type="text/event-stream",
            headers={
                "Cache-Control":     "no-cache",
                "Connection":        "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Input-Mode":      routing.input_mode,
            },
        )

    # ── 7. Standard JSON path ────────────────────────────────────────────────
    try:
        report = await orchestrator.analyze(analysis_request)
    except Exception as exc:
        logger.exception("Orchestrator failed in vision multipart endpoint: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis pipeline error: {exc}",
        )

    logger.info(
        "Vision multipart complete: mode=%s agents_succeeded=%d failed=%d %.0f ms",
        routing.input_mode,
        report.agents_succeeded,
        report.agents_failed,
        report.total_processing_time_ms,
    )

    return VisionAnalysisResponse(
        routing_decision=routing,
        vision_extraction=vision_extraction,
        report=report,
    )


# ---------------------------------------------------------------------------
# GET /api/v2/analyze/multipart/info
# ---------------------------------------------------------------------------

@router.get(
    "/analyze/multipart/info",
    summary="Vision Multipart Endpoint Info",
    description="Returns capabilities and accepted formats for the vision multipart endpoint.",
)
async def multipart_info() -> dict:
    """Describe the vision endpoint capabilities."""
    return {
        "endpoint":         "POST /api/v2/analyze/multipart",
        "description":      "Vision-capable multipart endpoint that routes image or text to the multi-agent orchestrator",
        "accepted_inputs":  {
            "image":        {
                "field":    "image (multipart/form-data file)",
                "formats":  sorted(_ALLOWED_IMAGE_MIMES),
                "max_size": "20 MB",
                "use_cases": [
                    "2-D skeletal / bond-line structure diagrams",
                    "3-D ball-and-stick model photographs",
                    "Chemical reaction schemes",
                    "Handwritten structural formulas or notes",
                    "Scanned pages from papers or textbooks",
                ],
            },
            "text": {
                "field":    "text (multipart/form-data string)",
                "examples": [
                    "Aspirin",
                    "CC(=O)Oc1ccccc1C(=O)O",
                    "What are the synthesis routes for ibuprofen?",
                ],
            },
        },
        "optional_overrides": [
            "top_k", "similarity_threshold", "use_mmr",
            "mmr_lambda", "agents_enabled", "stream",
        ],
        "vision_backend":   "Google Gemini Vision (free tier)",
        "routing_logic":    "image → GeminiVision → orchestrator | text → orchestrator",
    }
