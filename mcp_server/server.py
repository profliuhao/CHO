"""
server.py

MCP stdio server exposing the six CHO tools defined in cho_tools.py.

This is the published artifact: reviewers can wire it up to any MCP-aware
LLM client (Claude Desktop, Cline, Open WebUI, etc.) and have the model
query CHO live.

The auditor harness in evaluation/mode3/auditor.py does NOT go through
this server — it imports cho_tools directly for speed. Both paths
serve the same tools so behavior is identical.

Usage:
    python -m mcp_server.server
or, with uv (the recommended MCP launcher):
    uv run mcp_server/server.py

The server uses stdio transport. To debug, the MCP CLI's `mcp dev`
command is convenient:
    mcp dev mcp_server/server.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add the repo root to sys.path so we can import cho_tools
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from mcp.server.fastmcp import FastMCP

from mcp_server import cho_tools


mcp = FastMCP(
    name="cho",
    instructions=(
        "The CHO (Clinical Hallucination Ontology) MCP server. Provides "
        "six tools for grounding LLM-generated clinical text against "
        "structured knowledge: drug indications, treats-relation checks, "
        "lab-value plausibility, reasoner-based hallucination "
        "classification, CHO class lookup, and harm-tier resolution. "
        "Use these to verify clinical claims before reporting them to "
        "the user, or to classify hallucinations in generated text."
    ),
)


@mcp.tool()
def lookup_drug_indications(drug: str) -> dict:
    """Look up the FDA-approved indications for a drug from RxNorm.

    Args:
        drug: Generic drug name (e.g. 'metformin').
    """
    return cho_tools.lookup_drug_indications(drug)


@mcp.tool()
def check_treats_relation(drug: str, condition: str) -> dict:
    """Check whether a (drug, treats, condition) claim is permitted by
    clinical knowledge. Returns 'permitted', 'contraindicated',
    'unsupported', or 'unknown', with the supporting evidence.

    Args:
        drug: Generic drug name.
        condition: Disease, syndrome, or other clinical condition.
    """
    return cho_tools.check_treats_relation(drug, condition)


@mcp.tool()
def get_lab_plausibility(lab: str, value: float, unit: str = "") -> dict:
    """Check whether a lab value is within physiologically plausible bounds.

    Args:
        lab: Canonical lab name (e.g. 'potassium', 'creatinine').
        value: Numeric value.
        unit: Optional unit string (e.g. 'mEq/L').
    """
    return cho_tools.get_lab_plausibility(lab, value, unit or None)


@mcp.tool()
def classify_error_via_reasoner(claim_text: str) -> dict:
    """Classify a clinical claim using the formal CHO ontology and the
    HermiT reasoner. The most rigorous classification — the only tool
    whose output cannot be obtained by lookup alone.

    Args:
        claim_text: A short clinical claim or sentence.
    """
    return cho_tools.classify_error_via_reasoner(claim_text)


@mcp.tool()
def lookup_cho_class(class_name: str) -> dict:
    """Look up a CHO ontology class definition.

    Args:
        class_name: A CHO class name (e.g. 'ForbiddenRelation').
    """
    return cho_tools.lookup_cho_class(class_name)


@mcp.tool()
def get_harm_tier(class_name: str) -> dict:
    """Return the harm-tier classes (T1/T2/T3/T4) the TBox restricts
    a given CHO class to.

    Args:
        class_name: A CHO class name.
    """
    return cho_tools.get_harm_tier(class_name)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
