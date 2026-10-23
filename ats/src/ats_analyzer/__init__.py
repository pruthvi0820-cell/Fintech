"""Explainable ATS-style resume analysis."""

__version__ = "2026.10.23.2"     # bump with each fix so a running app shows which code it has

from .extract import ExtractionError, extract_bytes, extract_file
from .findings import Finding
from .parser import ParsedResume, parse_bytes, parse_file, parse_text

__all__ = ["ExtractionError", "Finding", "ParsedResume", "extract_bytes", "extract_file",
           "parse_bytes", "parse_file", "parse_text"]
