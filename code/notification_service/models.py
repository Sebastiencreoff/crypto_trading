from pydantic import BaseModel
from typing import Optional

class NotificationRequest(BaseModel):
    message: str
    channel_id: Optional[str] = None # If not provided, will use default channel from config

class NotificationResponse(BaseModel):
    status: str # e.g., "success", "error"
    details: Optional[str] = None # E.g., error message or confirmation
    message_id: Optional[str] = None # Optional: Slack's message timestamp (ts) if available
    channel_id: Optional[str] = None # Channel where message was sent
