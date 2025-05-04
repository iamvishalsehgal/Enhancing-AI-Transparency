import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

class Logger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
    def _get_log_file(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d")
        return self.log_dir / f"rag_{timestamp}.log"
    
    def log(self, event_type: str, data: Dict[str, Any]):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            **data
        }
        
        with open(self._get_log_file(), "a") as f:
            f.write(json.dumps(entry) + "\n")
            
        # Also print to console for debugging
        print(f"[{event_type}] {json.dumps(data, indent=2)}")