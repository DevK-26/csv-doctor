"""Phase B2 — Ask-Your-Data agent (BLUEPRINT T4).

question -> generated pandas program (from schema + facts, never from memory
of the file) -> sandbox -> independent verification -> verified answer, or an
honest "could not be verified" / failure. See agent.py for the contract and
README.md in this folder for the LLM upgrade path.
"""
from .agent import AskInputError, answer_question
from .context import ContextError
from .generator import (CodeGenerator, GeneratedProgram, GenerationError, GenerationRequest,
                        LLMCodeGenerator, LocalRulesGenerator, Unanswerable)
from .sandbox import Limits, SandboxBusy

__all__ = ["answer_question", "AskInputError", "ContextError", "CodeGenerator",
           "GeneratedProgram", "GenerationError", "GenerationRequest", "LLMCodeGenerator",
           "LocalRulesGenerator", "Unanswerable", "Limits", "SandboxBusy"]
