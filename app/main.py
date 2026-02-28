"""
FastAPI application — RagBluCare Medical Assistant v2.

Phase 2: Full diagnostic reasoning engine with:
  - Dual-layer symptom extraction (rule + LLM)
  - Dynamic follow-up conversation intelligence
  - Hybrid severity classification (rule + answer-analysis + LLM)
  - Safe OTC medication policy
  - Home-remedy integration
  - Structured diagnostic response streaming

Endpoints:
  GET  /health          → service health check
  POST /start-session   → create a new user session
  POST /chat            → streaming diagnostic response (SSE)
  POST /admin/ingest    → manually trigger data ingestion
"""

from __future__ import annotations

import io
import json
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from pathlib import Path

import httpx

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.models import (
    StartSessionRequest,
    StartSessionResponse,
    ChatRequest,
    RegisterRequest,
    LoginRequest,
    AuthResponse,
)
from app.llm_client import LLMClient
from app.embeddings import EmbeddingManager
from app.ingestion import IngestionPipeline
from app.retriever import HybridRetriever
from app.firebase_manager import FirebaseManager
from app.diagnosis_engine import DiagnosisEngine
from app.utils import get_settings, setup_logger

logger = setup_logger(__name__)
settings = get_settings()

# ═══════════════════════════════════════════════════════════
# Global singletons (initialised in lifespan)
# ═══════════════════════════════════════════════════════════
llm_client: LLMClient | None = None
embedding_mgr: EmbeddingManager | None = None
retriever: HybridRetriever | None = None
firebase_mgr: FirebaseManager | None = None
diagnosis_engine: DiagnosisEngine | None = None


# ═══════════════════════════════════════════════════════════
# Lifespan (startup / shutdown)
# ═══════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator:
    global llm_client, embedding_mgr, retriever, firebase_mgr, diagnosis_engine

    logger.info("=" * 60)
    logger.info("  RagBluCare v2 — starting up")
    logger.info("=" * 60)

    # 1. LLM client
    llm_client = LLMClient()
    logger.info("LLM client ready")

    # 2. Embedding manager + FAISS
    embedding_mgr = EmbeddingManager()
    if not embedding_mgr.load_index():
        logger.info("No cached index — running ingestion …")
        pipeline = IngestionPipeline(llm_client=llm_client)
        nodes = pipeline.run(use_llm=False)
        if nodes:
            embedding_mgr.build_index(nodes)
        else:
            logger.warning(
                "No data found. Place .txt / .json files in %s",
                settings.DATA_DIR,
            )

    # 3. Retriever
    retriever = HybridRetriever(embedding_mgr)
    logger.info("Hybrid retriever ready")

    # 4. Firebase / in-memory sessions
    firebase_mgr = FirebaseManager()
    firebase_mgr.initialize()

    # 5. Diagnosis engine (Phase 2 orchestrator)
    diagnosis_engine = DiagnosisEngine(
        llm_client=llm_client,
        retriever=retriever,
        firebase_mgr=firebase_mgr,
    )
    logger.info("Diagnosis engine ready")

    logger.info("=" * 60)
    logger.info("  RagBluCare v2 — ready to serve")
    logger.info("=" * 60)

    yield  # ── application is running ──

    # Shutdown
    if llm_client:
        await llm_client.close()
    logger.info("RagBluCare shut down.")


# ═══════════════════════════════════════════════════════════
# App factory
# ═══════════════════════════════════════════════════════════

app = FastAPI(
    title="RagBluCare Medical Assistant",
    description=(
        "Semi-clinical AI diagnostic reasoning engine with hybrid RAG, "
        "dynamic severity classification, and intelligent conversation flow."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════
# GET /health
# ═══════════════════════════════════════════════════════════

@app.get("/health")
async def health_check():
    idx_size = (
        embedding_mgr.index.ntotal
        if embedding_mgr and embedding_mgr.index
        else 0
    )
    return {
        "status": "healthy",
        "version": "2.0.0",
        "index_size": idx_size,
        "firebase_active": firebase_mgr.is_firebase_active if firebase_mgr else False,
        "diagnosis_engine": diagnosis_engine is not None,
    }


# ═══════════════════════════════════════════════════════════
# POST /register  — Create a new user account
# ═══════════════════════════════════════════════════════════

@app.post("/register", response_model=AuthResponse)
async def register(req: RegisterRequest):
    if not req.name or not req.password:
        raise HTTPException(400, "Name and password are required")
    if len(req.password) < 4:
        raise HTTPException(400, "Password must be at least 4 characters")
    if not firebase_mgr:
        raise HTTPException(503, "Service unavailable")

    result = firebase_mgr.register_user(req.name, req.password)
    return AuthResponse(**result)


# ═══════════════════════════════════════════════════════════
# POST /login  — Authenticate an existing user
# ═══════════════════════════════════════════════════════════

@app.post("/login", response_model=AuthResponse)
async def login(req: LoginRequest):
    if not req.name or not req.password:
        raise HTTPException(400, "Name and password are required")
    if not firebase_mgr:
        raise HTTPException(503, "Service unavailable")

    result = firebase_mgr.authenticate_user(req.name, req.password)
    return AuthResponse(**result)


# ═══════════════════════════════════════════════════════════
# POST /admin/clear-records  — Delete all old Firebase data
# ═══════════════════════════════════════════════════════════

@app.post("/admin/clear-records")
async def clear_records():
    if not firebase_mgr:
        raise HTTPException(503, "Service unavailable")
    result = firebase_mgr.clear_all_records()
    return {"status": "cleared", **result}


# ═══════════════════════════════════════════════════════════
# POST /start-session
# ═══════════════════════════════════════════════════════════

@app.post("/start-session", response_model=StartSessionResponse)
async def start_session(req: StartSessionRequest):
    if not req.user_id:
        raise HTTPException(400, "user_id is required")
    if not firebase_mgr:
        raise HTTPException(503, "Session service unavailable")

    sid = firebase_mgr.create_session(req.user_id)

    # Store optional context fields if provided at session start
    update_fields = {}
    if req.name:
        update_fields["patient_name"] = req.name
    if req.age:
        update_fields["patient_age"] = req.age
    if req.gender:
        update_fields["patient_gender"] = req.gender
    if req.country:
        update_fields["patient_country"] = req.country
    if req.temporary:
        update_fields["is_temporary"] = True
    if update_fields:
        firebase_mgr.update_session(req.user_id, sid, update_fields)

    return StartSessionResponse(session_id=sid)


# ═══════════════════════════════════════════════════════════
# POST /chat  (Server-Sent Events streaming via DiagnosisEngine)
# ═══════════════════════════════════════════════════════════

@app.post("/chat")
async def chat(req: ChatRequest):
    if not all([req.user_id, req.session_id, req.message]):
        raise HTTPException(400, "user_id, session_id, and message are required")
    if not diagnosis_engine or not firebase_mgr:
        raise HTTPException(503, "Service not ready")

    # Verify session exists
    session = firebase_mgr.get_session(req.user_id, req.session_id)
    if not session:
        raise HTTPException(404, "Session not found. Call /start-session first.")

    # ── Delegate to DiagnosisEngine (Phase 2 orchestrator) ──
    return StreamingResponse(
        _stream_diagnosis(req.user_id, req.session_id, req.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _stream_diagnosis(
    user_id: str, session_id: str, message: str
) -> AsyncGenerator[str, None]:
    """
    Yield SSE events with Gemini-like typing cadence.

    DiagnosisEngine.process_turn() already yields pre-formatted SSE strings
    (via its _sse() helper), so we pass them through directly.
    """
    try:
        async for sse_chunk in diagnosis_engine.process_turn(
            user_id=user_id,
            session_id=session_id,
            message=message,
        ):
            yield sse_chunk
            await asyncio.sleep(0.015)  # Gemini-like typing cadence
    except Exception as exc:
        logger.error("Stream error: %s", exc, exc_info=True)
        yield f"data: {json.dumps({'error': str(exc)})}\n\n"


# ═══════════════════════════════════════════════════════════
# POST /admin/ingest
# ═══════════════════════════════════════════════════════════

@app.post("/admin/ingest")
async def trigger_ingestion(use_llm: bool = False, force: bool = False):
    """Manually (re)trigger the data-ingestion + index-build pipeline."""
    if not llm_client or not embedding_mgr:
        raise HTTPException(503, "Service not ready")

    pipeline = IngestionPipeline(llm_client=llm_client)
    nodes = pipeline.run(use_llm=use_llm, force_refresh=force)
    embedding_mgr.build_index(nodes)

    return {
        "status": "success",
        "nodes_created": len(nodes),
        "index_size": embedding_mgr.index.ntotal if embedding_mgr.index else 0,
    }


# ═══════════════════════════════════════════════════════════
# POST /generate-pdf  -- Download consultation report as PDF
# ═══════════════════════════════════════════════════════════

@app.post("/generate-pdf")
async def generate_pdf(request: Request):
    """Generate a professional PDF report of the diagnostic conversation."""
    try:
        from app.pdf_generator import generate_report_pdf
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="PDF generation unavailable. Install reportlab: pip install reportlab",
        )

    try:
        body = await request.json()
        user_id = body.get("user_id", "")
        session_id_req = body.get("session_id", "")

        # If session info provided, pull rich data from the session
        if user_id and session_id_req and firebase_mgr:
            session = firebase_mgr.get_session(user_id, session_id_req)
            if session:
                messages = session.previous_messages
                diag = session.diagnosis_summary
                detail = session.detail_data or {}
                meds = detail.get("medications", [])
                rems = detail.get("remedies", [])
                tests = detail.get("tests", "")
                syms = list(session.extracted_symptoms)
                ts = session.created_at
            else:
                messages = body.get("messages", [])
                diag = body.get("diagnosis", "")
                meds, rems, tests, syms = [], [], "", []
                ts = body.get("timestamp", "")
        else:
            messages = body.get("messages", [])
            diag = body.get("diagnosis", "")
            meds = body.get("medications", [])
            rems = body.get("remedies", [])
            tests = body.get("tests", "")
            syms = body.get("symptoms", [])
            ts = body.get("timestamp", "")

        buffer = generate_report_pdf(
            messages=messages,
            diagnosis=diag,
            symptoms=syms,
            medications=meds,
            remedies=rems,
            tests=tests,
            session_id=session_id_req,
            timestamp=ts,
        )

        date_part = ts.split("T")[0] if "T" in str(ts) else "report"
        filename = f"RagBluCare_Report_{date_part}.pdf"

        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as exc:
        logger.error("PDF generation error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ═══════════════════════════════════════════════════════════
# GET /api/sessions  -- List all sessions for a user
# ═══════════════════════════════════════════════════════════

@app.get("/api/sessions")
async def list_sessions(user_id: str = Query(..., description="User identifier")):
    """Return summary list of all sessions for a user."""
    if not firebase_mgr:
        raise HTTPException(503, "Session service unavailable")
    sessions = firebase_mgr.list_sessions(user_id)
    return {"sessions": sessions}


# ═══════════════════════════════════════════════════════════
# GET /api/sessions/{session_id}  -- Full session detail
# ═══════════════════════════════════════════════════════════

@app.get("/api/sessions/{session_id}")
async def get_session_detail(
    session_id: str,
    user_id: str = Query(..., description="User identifier"),
):
    """Return full session data for read-only viewing."""
    if not firebase_mgr:
        raise HTTPException(503, "Session service unavailable")
    session = firebase_mgr.get_session(user_id, session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return session.model_dump()


# ═══════════════════════════════════════════════════════════
# DELETE /api/sessions/{session_id}  -- Delete a session
# ═══════════════════════════════════════════════════════════

@app.delete("/api/sessions/{session_id}")
async def delete_session(
    session_id: str,
    user_id: str = Query(..., description="User identifier"),
):
    """Delete a session (used for temporary sessions cleanup)."""
    if not firebase_mgr:
        raise HTTPException(503, "Session service unavailable")
    deleted = firebase_mgr.delete_session(user_id, session_id)
    if not deleted:
        raise HTTPException(404, "Session not found")
    return {"status": "deleted", "session_id": session_id}


# ═══════════════════════════════════════════════════════════
# Static files & Chat UI
# ═══════════════════════════════════════════════════════════

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def chat_ui():
    """Serve the chat interface."""
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return HTMLResponse("<h1>BluCare API is running. No UI found.</h1>")


@app.get("/my-sessions", response_class=HTMLResponse)
async def sessions_page():
    """Serve the session history page."""
    sessions_file = STATIC_DIR / "sessions.html"
    if sessions_file.exists():
        return FileResponse(str(sessions_file))
    return HTMLResponse("<h1>Sessions page not found.</h1>")


# ═══════════════════════════════════════════════════════════
# Hospital service proxy  (avoids browser CORS restriction)
# ═══════════════════════════════════════════════════════════

HOSPITAL_SERVICE_URL = "http://127.0.0.1:8001/hospitals"
AMBULANCE_SERVICE_URL = "http://127.0.0.1:8001/ambulance"


@app.get("/proxy/hospitals")
async def proxy_hospitals(
    disease: str = Query(..., description="Diagnosed disease name"),
    lat: float = Query(..., description="Patient latitude"),
    lon: float = Query(..., description="Patient longitude"),
):
    """Forward hospital lookup to the internal service, bypassing browser CORS."""
    target = f"{HOSPITAL_SERVICE_URL}?disease={disease}&lat={lat}&lon={lon}"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                HOSPITAL_SERVICE_URL,
                params={"disease": disease, "lat": lat, "lon": lon},
            )
            resp.raise_for_status()
            return JSONResponse(content=resp.json())
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail=f"Cannot reach hospital service at {HOSPITAL_SERVICE_URL}")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Hospital service timed out after 30 s")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)


# ═══════════════════════════════════════════════════════════
# Ambulance service proxy  (avoids browser CORS restriction)
# ═══════════════════════════════════════════════════════════

@app.get("/proxy/ambulance")
async def proxy_ambulance(
    lat: float = Query(..., description="Patient latitude"),
    lon: float = Query(..., description="Patient longitude"),
):
    """Forward ambulance lookup to the internal service, bypassing browser CORS."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                AMBULANCE_SERVICE_URL,
                params={"lat": lat, "lon": lon},
            )
            resp.raise_for_status()
            return JSONResponse(content=resp.json())
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail=f"Cannot reach ambulance service at {AMBULANCE_SERVICE_URL}")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Ambulance service timed out after 30 s")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)


@app.get("/nearby-ambulance", response_class=HTMLResponse)
async def ambulance_page():
    """Serve the nearby ambulance page."""
    amb_file = STATIC_DIR / "ambulance.html"
    if amb_file.exists():
        return FileResponse(str(amb_file))
    return HTMLResponse("<h1>Ambulance page not found.</h1>")


# ═══════════════════════════════════════════════════════════
# Direct execution
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )
