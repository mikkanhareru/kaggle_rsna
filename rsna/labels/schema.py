# The JSON contract for report -> weak lables. vLLM enforces this with 
# Guided decoding, so every one of the 4487 reports is guaranteed to parse.

from enum import Enum 
from typing import Optional
from pydantic import BaseModel, Field

class Status(str, Enum):
    present = "present" # report asserts the finding 
    absent = "absent"   # report explicitly denies it ("e.g. ACL is intact")
    uncertain = "uncertain" # hedged ("possihble", "cannot exclude")
    not_mentioned = "not_mentioned" # report is silent 
    
class Severity(str, Enum):
    mild = "mild"
    moderate = "moderate"
    severe = "severe"
    unspecified = "unspecified"
    
class Temporality(str, Enum):
    current = "current" # active finding at this exam
    historical = "historical" # healed
    postoperative = "postoperative" # post-surgical change
    differential = "differential" # listed only as a possibility in the impression
    
class Finding(BaseModel):
    status: Status
    severity: Optional[Severity] = None
    temporality: Temporality
    evidence: str = Field(default="", max_length=160) # vervatun span, source language
    
class ReportExtraction(BaseModel):
    language:         str = Field(max_length=8)   # ISO-639-1 as detected by the model
    acl:              Finding
    mcl:              Finding
    medial_meniscus:  Finding
    lateral_meniscus: Finding
    medial_oa:        Finding
    lateral_oa:       Finding
    pf_oa:            Finding
    effusion:         Finding
    synovitis:        Finding
    bakers:           Finding
    contusion:        Finding
    fracture:         Finding

# Maps JSON keys back to the exact train.csv / submission.csv column names.
COLUMN_MAP = {
    "acl": "ACL", "mcl": "MCL",
    "medial_meniscus": "Medial Meniscus", "lateral_meniscus": "Lateral Meniscus",
    "medial_oa": "Medial OA", "lateral_oa": "Lateral OA", "pf_oa": "PF OA",
    "effusion": "Effusion", "synovitis": "Synovitis", "bakers": "Baker's",
    "contusion": "Contusion", "fracture": "Fracture",
}

if __name__ == "__main__":
    import json
    schema = ReportExtraction.model_json_schema()
    print("schema bytes:", len(json.dumps(schema)))
    print("findings:", len(COLUMN_MAP), "| column names match train.csv:", list(COLUMN_MAP.values()))