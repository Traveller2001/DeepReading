"""LLM service entry point — re-exports the public API.

All implementation has been split into focused modules:
  core.prompts           – prompt templates
  core.citation          – citation normalization and enhancement
  core.report_generator  – report generation (tool-calling loop)
  core.discussion        – reader/writer discussion and polish
"""

from core.report_generator import generate_report_stream
from core.discussion import generate_discussion_stream

__all__ = ["generate_report_stream", "generate_discussion_stream"]
