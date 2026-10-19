from typing import Optional, Dict, List, Any
from pydantic import BaseModel, model_serializer


class FieldMatchDetailV4(BaseModel):
    match: bool
    ocr: Optional[str] = None
    db: Optional[str] = None

    @model_serializer
    def serialize_field(self) -> Dict[str, Any]:
        if self.match:
            return {"match": True}
        return {
            "match": False,
            "ocr": self.ocr,
            "db": self.db
        }


class KTPValidationResponseV4(BaseModel):
    different: bool
    mismatched_fields: List[str]
    details: Dict[str, FieldMatchDetailV4]
