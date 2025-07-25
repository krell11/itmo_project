from typing import TypedDict, List, Dict


class GraphState(TypedDict):
    query: str
    collection_name: str
    context: List[str]
    answer: str
    validation: Dict[str, bool]
    validation_attempts: int
