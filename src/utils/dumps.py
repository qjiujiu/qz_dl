from dataclasses import is_dataclass, asdict
from pydantic import BaseModel
from pathlib import Path
from typing import Any, Dict

def ensure_dirs(*dirs: str):
    for d in dirs:
        if d is not None:
            Path(d).mkdir(parents=True, exist_ok=True)


def to_jsonable(obj: Any) -> Dict:
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if is_dataclass(obj):
        return to_jsonable(asdict(obj)) 
    elif isinstance(obj, list):
        return [to_jsonable(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    else:
        return obj