from pydantic import BaseModel


class ClassificationResult(BaseModel):
    collection_name: str
    confidence: float
