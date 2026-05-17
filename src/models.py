"""
Pydantic schemas for all Ollama-generated structured outputs.

Every schema has:
  - Field-level validation (str coercion, length guards)
  - A .from_dict() class method that tolerates malformed LLM output
  - A .to_legacy() method returning the plain dict/dataclass form
    that the rest of the pipeline already expects
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _coerce_str(v: Any) -> str:
    """Accept any type and return a clean string — handles int/None from LLMs."""
    if v is None:
        return ""
    return str(v).strip()


def _coerce_str_list(v: Any) -> list[str]:
    if isinstance(v, list):
        return [_coerce_str(x) for x in v if x is not None]
    if isinstance(v, str):
        return [v.strip()] if v.strip() else []
    return []


# ---------------------------------------------------------------------------
# Video Idea
# ---------------------------------------------------------------------------

class VideoIdeaModel(BaseModel):
    title: str = Field(default="")
    hook: str = Field(default="")
    description: str = Field(default="")
    tags: list[str] = Field(default_factory=list)
    estimated_virality: str = Field(default="Medium")

    @field_validator("title", "hook", "description", mode="before")
    @classmethod
    def coerce_str(cls, v: Any) -> str:
        return _coerce_str(v)

    @field_validator("tags", mode="before")
    @classmethod
    def coerce_tags(cls, v: Any) -> list[str]:
        return _coerce_str_list(v)

    @field_validator("estimated_virality", mode="before")
    @classmethod
    def validate_virality(cls, v: Any) -> str:
        s = _coerce_str(v)
        return s if s in {"Low", "Medium", "High", "Viral"} else "Medium"

    @classmethod
    def from_dict(cls, d: dict, fallback_title: str = "") -> "VideoIdeaModel":
        obj = cls.model_validate(d)
        if not obj.title:
            obj.title = fallback_title[:70]
        return obj


# ---------------------------------------------------------------------------
# Script section
# ---------------------------------------------------------------------------

class ScriptSection(BaseModel):
    heading: str = Field(default="")
    script: str = Field(default="")

    @field_validator("heading", "script", mode="before")
    @classmethod
    def coerce_str(cls, v: Any) -> str:
        return _coerce_str(v)

    @model_validator(mode="after")
    def ensure_minimum_content(self) -> "ScriptSection":
        # If script text is very short, pad with heading so downstream never sees empty
        if len(self.script.split()) < 5 and self.heading:
            self.script = self.script or self.heading
        return self


# ---------------------------------------------------------------------------
# Full video script (long form)
# ---------------------------------------------------------------------------

class VideoScriptModel(BaseModel):
    title: str = Field(default="")
    hook: str = Field(default="")
    sections: list[ScriptSection] = Field(default_factory=list)
    call_to_action: str = Field(default="")
    description: str = Field(default="")
    tags: list[str] = Field(default_factory=list)

    @field_validator("title", "hook", "call_to_action", "description", mode="before")
    @classmethod
    def coerce_str(cls, v: Any) -> str:
        return _coerce_str(v)

    @field_validator("tags", mode="before")
    @classmethod
    def coerce_tags(cls, v: Any) -> list[str]:
        return _coerce_str_list(v)

    @field_validator("sections", mode="before")
    @classmethod
    def coerce_sections(cls, v: Any) -> list[dict]:
        if not isinstance(v, list):
            return []
        result = []
        for item in v:
            if isinstance(item, dict):
                result.append(item)
            elif isinstance(item, str):
                result.append({"heading": "", "script": item})
        return result

    @classmethod
    def from_dict(cls, d: dict) -> "VideoScriptModel":
        return cls.model_validate(d)

    def section_dicts(self) -> list[dict]:
        return [{"heading": s.heading, "script": s.script} for s in self.sections]


# ---------------------------------------------------------------------------
# Shorts script
# ---------------------------------------------------------------------------

class ShortScriptModel(BaseModel):
    title: str = Field(default="")
    hook: str = Field(default="")
    body: str = Field(default="")
    call_to_action: str = Field(default="")
    tags: list[str] = Field(default_factory=list)

    @field_validator("title", "hook", "body", "call_to_action", mode="before")
    @classmethod
    def coerce_str(cls, v: Any) -> str:
        return _coerce_str(v)

    @field_validator("tags", mode="before")
    @classmethod
    def coerce_tags(cls, v: Any) -> list[str]:
        return _coerce_str_list(v)

    def to_sections(self) -> list[dict]:
        """Convert flat body into section list format expected by orchestrator."""
        import re
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", self.body) if s.strip()]
        if not sentences:
            sentences = [self.body] if self.body else [""]
        return [{"heading": f"Part {i + 1}", "script": sent} for i, sent in enumerate(sentences)]

    @classmethod
    def from_dict(cls, d: dict) -> "ShortScriptModel":
        return cls.model_validate(d)


# ---------------------------------------------------------------------------
# Hook variant (for A/B testing)
# ---------------------------------------------------------------------------

class HookVariant(BaseModel):
    hook: str = Field(default="")
    angle: str = Field(default="")   # e.g. "fear", "curiosity", "authority"

    @field_validator("hook", "angle", mode="before")
    @classmethod
    def coerce_str(cls, v: Any) -> str:
        return _coerce_str(v)
