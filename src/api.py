"""
CliniGuard – REST API Service
==============================
Production-grade FastAPI backend enabling RESTful integration with
mobile applications, web apps, and external microservices.

Endpoints:
- GET  /health              Service healthcheck & telemetry
- POST /api/v1/query        Evaluate safety query through LangGraph agent
- GET  /api/v1/sops         Retrieve active SOP catalog
- POST /api/v1/sops/reload  Hot-reload SOP policy definitions
"""

import uuid
import json
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Security, Depends, status
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.graph import run_agent_turn, stream_agent_turn
from src.policy.loader import PolicyLoader
from src.retrieval.retriever import SOPRetriever
from src.retrieval.embeddings import EmbeddingService
from src.auth import (
    authenticate_session,
    validate_api_key,
    is_auth_enabled,
    check_rate_limit,
    record_query,
)
from src.config import LLM_PROVIDER
from src.observability import get_logger, setup_logging

# Explicit central logging initialization for API service
setup_logging()
logger = get_logger("api")


# ------------------------------------------------------------------------------
# App Initialization & Middleware
# ------------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Confirms central logging setup and service initialization on FastAPI startup."""
    setup_logging()
    logger.info("CliniGuard Environmental Safety Advisory API service initialized.")
    yield


app = FastAPI(
    title="CliniGuard Environmental Safety Advisory API",
    description="Agentic AI system providing grounded outdoor safety advisories based on live meteorological telemetry.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# ------------------------------------------------------------------------------
# Security Dependency
# ------------------------------------------------------------------------------

def verify_access_key(key: Optional[str] = Depends(api_key_header)) -> bool:
    """Validates X-API-Key header against server ACCESS_KEY statelessly."""
    if not is_auth_enabled():
        return True
    if not key or not validate_api_key(key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key in X-API-Key header",
        )
    return True



# ------------------------------------------------------------------------------
# Request & Response Schemas
# ------------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language safety inquiry", min_length=2)
    thread_id: Optional[str] = Field(None, description="Session/thread identifier for multi-turn continuity")
    location_index: int = Field(0, description="Selected index for disambiguating locations with multiple matches")


class QueryResponse(BaseModel):
    thread_id: str
    advisory: str
    is_safe: bool
    risk_level: Optional[str] = None
    location: Optional[str] = None
    location_candidates: List[Dict[str, Any]] = []
    weather: Optional[Dict[str, Any]] = None
    matched_sops: List[str] = []
    retrieved_sops: List[Dict[str, Any]] = Field(default_factory=list)
    retrieval_error: Optional[str] = None
    requires_clarification: bool = False
    clarification_question: Optional[str] = None
    execution_time_ms: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    embedding_provider: str
    sop_count: int
    auth_enabled: bool


# ------------------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Returns system status, active providers, and loaded SOP metrics."""
    loader = PolicyLoader()
    sops = loader.load_policies()
    embed_service = EmbeddingService()

    return HealthResponse(
        status="healthy",
        llm_provider=LLM_PROVIDER,
        embedding_provider=embed_service.provider,
        sop_count=len(sops),
        auth_enabled=is_auth_enabled(),
    )


@app.post("/api/v1/query", response_model=QueryResponse)
def evaluate_query(
    req: QueryRequest,
    authenticated: bool = Depends(verify_access_key),
) -> QueryResponse:
    """
    Executes the full CliniGuard state machine for a user query.
    Grounded with live weather telemetry and deterministic SOP compliance.
    """
    thread_id = req.thread_id or str(uuid.uuid4())

    # Rate limiting check (atomic check-and-record to prevent TOCTOU race conditions)
    allowed, wait = check_rate_limit(thread_id, record=True)
    if not allowed:
        logger.warning("Rate limit exceeded for session %s (wait %ss)", thread_id, wait)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Please retry after {wait} seconds.",
        )

    # Execute conversational agent turn
    try:
        logger.info("Evaluating safety query for session %s", thread_id)
        final_state = run_agent_turn(user_query=req.query, session_id=thread_id)
    except Exception as e:
        logger.error("Agent workflow execution error for session %s: %s", thread_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent workflow execution error: {str(e)}",
        )

    user_context = final_state.get("user_context") or {}
    risk_assessment = final_state.get("risk_assessment") or {}
    loc_candidates = final_state.get("location_candidates") or []

    raw_matched = final_state.get("matched_sops") or []
    matched_ids = [
        s.get("id", str(s)) if isinstance(s, dict) else str(s)
        for s in raw_matched
    ]

    # Format response
    return QueryResponse(
        thread_id=thread_id,
        advisory=final_state.get("final_response") or "No advisory produced.",
        is_safe=risk_assessment.get("is_safe", True) if isinstance(risk_assessment, dict) else getattr(risk_assessment, "is_safe", True),
        risk_level=risk_assessment.get("risk_level") if isinstance(risk_assessment, dict) else getattr(risk_assessment, "risk_level", None),
        location=user_context.get("location") if isinstance(user_context, dict) else getattr(user_context, "location", None) or final_state.get("location"),
        location_candidates=loc_candidates,
        weather=final_state.get("weather_data"),
        matched_sops=matched_ids,
        retrieved_sops=final_state.get("retrieved_sops") or [],
        retrieval_error=final_state.get("retrieval_error"),
        requires_clarification=final_state.get("pending_clarification", False),
        clarification_question=final_state.get("final_response") if final_state.get("pending_clarification") else None,
    )


@app.post("/api/v1/query/stream")
def stream_query(
    req: QueryRequest,
    authenticated: bool = Depends(verify_access_key),
):
    """
    Streams LangGraph agent execution events via Server-Sent Events (SSE).
    Emits structured events:
      - 'status': Real-time processing milestones (guardrails, weather fetch, policy eval).
      - 'provisional': Intermediate advisory text (marked clearly with is_verified: false).
      - 'result': Authoritative, validated advisory with safety level and telemetry.
      - 'error': Safe error information if pipeline fails.
      - 'done': Stream completion marker.
    """
    thread_id = req.thread_id or str(uuid.uuid4())

    # Rate limiting check (atomic check-and-record to prevent TOCTOU race conditions)
    allowed, wait = check_rate_limit(thread_id, record=True)
    if not allowed:
        logger.warning("Rate limit exceeded for streaming query in session %s (wait %ss)", thread_id, wait)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Please retry after {wait} seconds.",
        )

    def sse_event_stream():
        try:
            for item in stream_agent_turn(user_query=req.query, session_id=thread_id):
                event_type = item.get("event", "message")
                data_payload = json.dumps(item.get("data", {}))
                yield f"event: {event_type}\ndata: {data_payload}\n\n"
        except Exception as e:
            logger.error("Stream processing failed for session %s: %s", thread_id, e)
            err_payload = json.dumps({"error": f"Stream processing failed: {str(e)}"})
            yield f"event: error\ndata: {err_payload}\n\n"

    return StreamingResponse(
        sse_event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )



@app.get("/api/v1/sops")
def list_sops(authenticated: bool = Depends(verify_access_key)) -> List[Dict[str, Any]]:
    """Lists all Standard Operating Procedures currently loaded in CliniGuard."""
    loader = PolicyLoader()
    return loader.load_policies()


@app.post("/api/v1/sops/reload")
def reload_sops(authenticated: bool = Depends(verify_access_key)) -> Dict[str, Any]:
    """Dynamically reloads SOP policy catalog from disk without downtime."""
    loader = PolicyLoader()
    sops = loader.load_policies(force_reload=True)
    retriever = SOPRetriever()
    retriever.reload_sops()
    logger.info("Successfully reloaded %d SOP policies from disk.", len(sops))
    return {
        "status": "success",
        "message": f"Successfully reloaded {len(sops)} SOP policies.",
        "sop_count": len(sops),
    }
