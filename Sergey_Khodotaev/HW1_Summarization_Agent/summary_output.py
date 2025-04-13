from pydantic import BaseModel

class SummaryOutput(BaseModel):
    abstractive_summary: str
    extractive_summary: str