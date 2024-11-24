from datetime import datetime
from typing import Optional

class RequestLog:
    def __init__(self, method: str, url: str, ip:str, data: Optional[dict] = None):
        # Инициализация объекта RequestLog
        self.method = method
        self.url = url
        self.ip = ip
        self.timestamp = datetime.utcnow()
        self.data = data if data else {}

    def to_dict(self):
        # Преобразование объекта RequestLog в словарь.
        return {
            "method": self.method,
            "url": self.url,
            "ip": self.ip,
            "timestamp": self.timestamp,
            "data": self.data
        }