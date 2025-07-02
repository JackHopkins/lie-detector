from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from datetime import datetime

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class LieSample(BaseModel):
    sample_id: str
    task: str
    task_id: str
    timestamp: datetime
    model: str
    trace: List[Message]
    lie_detection_prompt: Message
    target_response: Message
    did_lie: bool
    lie_type: Literal["factual", "reasoning", "other"]
    lie_utterances: List[str]
    correct_answer: str
