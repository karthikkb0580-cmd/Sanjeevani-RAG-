"""
Module 12b: API – Documents
app/api/documents.py

Endpoints:
  POST /documents/index          – Index a single document
  POST /documents/batch-index    – Index multiple documents
  DELETE /documents/{id}         – Delete a document and all its chunks
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.config.settings import Settings, get_settings
from app.schemas.document import (
    BatchIndexResponse,
    DeleteDocumentResponse,
    IndexDocumentResponse,
)
from app.services.indexing_service import IndexingService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])


def _get_indexing_service() -> IndexingService:
    return IndexingService()


async def _save_upload(file: UploadFile, upload_dir: Path) -> Path:
    """Persist an uploaded file to disk and return its path."""
    safe_name = f"{uuid.uuid4()}_{file.filename}"
    dest = upload_dir / safe_name
    content = await file.read()
    dest.write_bytes(content)
    return dest


def _validate_file(file: UploadFile, settings: Settings) -> None:
    """Raise HTTPException if file type or size is invalid."""
    suffix = Path(file.filename or "").suffix.lower().lstrip(".")
    if suffix not in settings.allowed_extension_set:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type '.{suffix}'. "
                f"Allowed: {', '.join(settings.allowed_extension_set)}"
            ),
        )


# ---------------------------------------------------------------------------
# POST /documents/index
# ---------------------------------------------------------------------------

@router.post(
    "/index",
    response_model=IndexDocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Index a single document",
    description=(
        "Upload a research paper (PDF, TXT, or DOCX) to be processed and "
        "indexed into the Qdrant vector database. Returns the assigned "
        "document_id and indexing statistics."
    ),
)
async def index_document(
    file: UploadFile = File(..., description="Research paper file (PDF/TXT/DOCX)"),
    title: str | None = Form(default=None, description="Override auto-detected title"),
    chunk_size: int | None = Form(default=None, ge=100, le=4000, description="Override chunk token size"),
    chunk_overlap: int | None = Form(default=None, ge=0, le=500, description="Override chunk overlap"),
    settings: Settings = Depends(get_settings),
    service: IndexingService = Depends(_get_indexing_service),
) -> IndexDocumentResponse:
    """
    Index a single document.

    Example request (curl):
    ```bash
    curl -X POST http://localhost:8000/documents/index \\
      -F "file=@paper.pdf" \\
      -F "title=My Research Paper"
    ```

    Example response:
    ```json
    {
        "document_id": "550e8400-e29b-41d4-a716-446655440000",
        "title": "My Research Paper",
        "total_chunks": 42,
        "pages": 12,
        "processing_time_ms": 3241.5,
        "status": "indexed",
        "message": "Successfully indexed 42 chunks from 12 pages"
    }
    ```
    """
    _validate_file(file, settings)
    upload_dir = settings.ensure_upload_dir()

    saved_path: Path | None = None
    try:
        saved_path = await _save_upload(file, upload_dir)
        result = await service.index_document(
            file_path=saved_path,
            title_override=title,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return result
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("Failed to index document '%s'", file.filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing failed: {exc}",
        )
    finally:
        if saved_path and saved_path.exists():
            saved_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# POST /documents/batch-index
# ---------------------------------------------------------------------------

@router.post(
    "/batch-index",
    response_model=BatchIndexResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch index multiple documents",
    description=(
        "Upload and index multiple research papers in a single request. "
        "Returns per-file indexing results and a summary of successes/failures."
    ),
)
async def batch_index_documents(
    files: list[UploadFile] = File(..., description="List of research paper files"),
    settings: Settings = Depends(get_settings),
    service: IndexingService = Depends(_get_indexing_service),
) -> BatchIndexResponse:
    """
    Batch index multiple documents.

    Example request (curl):
    ```bash
    curl -X POST http://localhost:8000/documents/batch-index \\
      -F "files=@paper1.pdf" \\
      -F "files=@paper2.docx"
    ```
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one file must be provided",
        )

    upload_dir = settings.ensure_upload_dir()
    results: list[IndexDocumentResponse] = []
    errors: list[dict] = []

    for file in files:
        saved_path: Path | None = None
        try:
            _validate_file(file, settings)
            saved_path = await _save_upload(file, upload_dir)
            result = await service.index_document(file_path=saved_path)
            results.append(result)
        except HTTPException as exc:
            errors.append({"filename": file.filename, "error": exc.detail})
        except Exception as exc:
            logger.exception("Batch indexing failed for '%s'", file.filename)
            errors.append({"filename": file.filename, "error": str(exc)})
        finally:
            if saved_path and saved_path.exists():
                saved_path.unlink(missing_ok=True)

    return BatchIndexResponse(
        total_files=len(files),
        successful=len(results),
        failed=len(errors),
        results=results,
        errors=errors,
    )


# ---------------------------------------------------------------------------
# DELETE /documents/{document_id}
# ---------------------------------------------------------------------------

@router.delete(
    "/{document_id}",
    response_model=DeleteDocumentResponse,
    summary="Delete a document",
    description=(
        "Permanently delete a document and all its associated vector chunks "
        "from the Qdrant collection."
    ),
)
async def delete_document(
    document_id: str,
    service: IndexingService = Depends(_get_indexing_service),
) -> DeleteDocumentResponse:
    """
    Delete a document by its document_id.

    Example request:
    ```bash
    curl -X DELETE http://localhost:8000/documents/550e8400-e29b-41d4-a716-446655440000
    ```

    Example response:
    ```json
    {
        "document_id": "550e8400-e29b-41d4-a716-446655440000",
        "deleted_chunks": 42,
        "status": "deleted",
        "message": "Document and all associated chunks deleted"
    }
    ```
    """
    try:
        deleted_chunks = await service.delete_document(document_id)
        return DeleteDocumentResponse(
            document_id=document_id,
            deleted_chunks=deleted_chunks,
        )
    except Exception as exc:
        logger.exception("Failed to delete document '%s'", document_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Deletion failed: {exc}",
        )
