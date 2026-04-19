from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db
from app.schemas.report import ReportRequest, ReportResponse
from app.services.llm import generate_report
from app.services.prompt_builder import build_analysis_prompt

router = APIRouter()

@router.post("/analyze", response_model = ReportResponse)
async def analyze(
    request : ReportRequest,
    db : AsyncSession = Depends(get_db)):
    """
    Main endpoint — takes customer data + model choice,
    returns a full AI-generated strategy report.

    Two ways to call this:
    1. Send raw data directly (for testing)
    2. Send a job_id (when ML pipeline is integrated in Phase 3)
    """
    data = await _get_data(request, db)
    prompt = build_analysis_prompt(data, focus = request.focus.value)
    try:
        result = await generate_report(prompt, model = request.model.value)
    except ValueError as e:
        raise HTTPException(status_code = 400, detail = str(e))
    except Exception as e:
        raise Exception(status_code = 503, detail = f"LLM Service Error : {str(e)}")
    
    return ReportResponse(
        job_id = request.job_id,
        model_used = result["model_used"],
        focus = request.focus.value,
        report = result['text'],
        input_tokens = result["input_tokens"],
        output_tokens = result["output_tokens"],
    )

async def _get_data(request : ReportRequest, db : AsyncSession) -> dict:
    """
    Figures out where to get the ML data from.
    Either from a job_id in the database or directly from the request.
    """
    if request.data:
        return request.data
    if request.job_id:
        raise HTTPException(status_code = 501, detail = "Job-based analysis not yet implemented. Please provide data directly for now.")
    raise HTTPException(
        status_code = 400,
        detail = "Either job_id or data must be provided."
    )