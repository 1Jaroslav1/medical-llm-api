from typing import List
from pydantic import BaseModel


class HistoryItem(BaseModel):
    human_message: str
    ai_message: str


class ChatItem(BaseModel):
    question: str
    history: List[HistoryItem]


class AnswerResponse(BaseModel):
    answer: str
