from pydantic import BaseModel
from typing import Optional
from enum import Enum

class ModelChoice(str, Enum):
    """
    The Model a user can choose from.
    Using am Enum means only these exact values are accepted.
    If someone sends "gpt5" it gets rejected automaticaly
    """
    GROQ = "groq"
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"

class ReportFocus(str,Enum):
    """
    The Strategic angle for report.
    """
    GENERAL = "general"
    RETENTION = "retention"
    UPSELL = "upsell"
    ACQUISITION = "acquisition"
    SEASONAL = "seasonal"

class ReportRequest(BaseModel):
    """
    What the user sends to request a report.
    In the real app, job_id is enough — we fetch the ML data from DB.
    For now we also accept raw data directly so we can test without
    a full ML pipeline.    
    """ 
    job_id : Optional[str] = None
    model : ModelChoice = ModelChoice.GROQ
    focus : ReportFocus = ReportFocus.GENERAL
    data : Optional[dict] = None

class ReportResponse(BaseModel):
    """
    What CarterX sends Back.
    """
    model_config = {"protected_namespaces": ()}
    job_id : Optional[str]
    model_used : str
    focus : str
    report : str
    input_tokens : Optional[int]
    output_tokens : Optional[int]