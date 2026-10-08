from typing import Optional

from pydantic import BaseModel


class RFPAnalysis(BaseModel):
    title: Optional[str] = None
    client: Optional[str] = None
    deadline: Optional[str] = None
    budget: Optional[str] = None
    requirements: list[str] = []
    deliverables: list[str] = []
    evaluation_criteria: list[str] = []
    submission_requirements: list[str] = []