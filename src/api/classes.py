from pydantic import BaseModel
from typing import Dict, Optional, Any


class QueryRequest(BaseModel):
    query: str
    user_id: str
    chat_id: str
    config: Optional[Dict[str, Any]] = None
    config_path: Optional[str] = None


class DirectConfigRequest(BaseModel):
    config: Dict


class QueryResponse(BaseModel):
    response: str
    status: str