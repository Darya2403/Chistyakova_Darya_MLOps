from datetime import datetime
from typing import Optional

class RequestLog:
    def __init__(self, method: str, url: str, data: Optional[dict] = None):
        self.method = method
        self.url = url
        self.timestamp = datetime.utcnow()
        self.data = data if data else {}

    def to_dict(self):
        return {
            "method": self.method,
            "url": self.url,
            "timestamp": self.timestamp,
            "data": self.data
        }