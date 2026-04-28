"""
auditor.py

Mode 3 auditor harness. Given a clinical claim and an arm specification,
queries an LLM (with or without tool access) and returns its
classification of the claim as hallucinated/correct + (optionally) which
CHO class.

Four experimental arms:
  - 'no_tools':         baseline. LLM answers from parametric memory.
  - 'generic_knowledge': LLM has access to ONE tool (UMLS-style concept
                         lookup) — controls for "any structured
                         knowledge helps", isolating CHO's contribution.
  - 'cho_tools':        LLM has access to all 6 CHO tools.
  - 'cho_ablated':      LLM has access to CHO tools EXCEPT
                        classify_error_via_reasoner — tests whether the
                        reasoner contributes beyond lookups.

OpenAI-compatible client surface (works with the local Qwen serving
stack, OpenAI proper, vLLM, etc.). Set OPENAI_BASE_URL and
OPENAI_API_KEY to point at your endpoint.

A MockLLMClient is provided so the harness can be unit-tested in
environments without LLM access.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol

HERE = Path(__file__).parent
EVAL_ROOT = HERE.parent
REPO_ROOT = EVAL_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from mcp_server import cho_tools


# =====================================================
# Tool subsets per arm
# =====================================================

CHO_FULL_TOOLS = [
    "lookup_drug_indications",
    "check_treats_relation",
    "get_lab_plausibility",
    "classify_error_via_reasoner",
    "lookup_cho_class",
    "get_harm_tier",
]

CHO_ABLATED_TOOLS = [
    "lookup_drug_indications",
    "check_treats_relation",
    "get_lab_plausibility",
    # classify_error_via_reasoner removed
    "lookup_cho_class",
    "get_harm_tier",
]

# 'generic_knowledge' arm: a single concept-lookup tool that simulates
# what an LLM would get from a generic UMLS/SNOMED MCP server. It returns
# the concept's name, semantic types, and (if drug) basic info — but
# nothing about treats/contraindications/reasoner classification.
GENERIC_KNOWLEDGE_TOOLS = ["lookup_concept_generic"]


def lookup_concept_generic(concept: str) -> dict:
    """Generic-knowledge baseline: returns canonical name + semantic
    type info for a concept, simulating what a vanilla UMLS lookup
    server would provide. Does NOT expose treats/contraindications/
    reasoner — those are CHO-specific."""
    from extraction.claim_extraction import get_lexicon
    lex = get_lexicon()
    info = lex.lookup(concept)
    if info is None:
        return {"concept": concept, "found": False,
                "evidence": "Concept not recognized in our terminology."}
    canonical, category = info
    return {
        "concept": canonical,
        "found": True,
        "category": category,
        "evidence": f"Recognized as a {category}.",
    }


GENERIC_TOOL_DISPATCH = {
    "lookup_concept_generic": lookup_concept_generic,
}


def get_tools_for_arm(arm: str) -> tuple[list[str], dict[str, callable]]:
    """Return (tool_names, dispatch_table) for the given arm."""
    if arm == "no_tools":
        return [], {}
    if arm == "generic_knowledge":
        return GENERIC_KNOWLEDGE_TOOLS, GENERIC_TOOL_DISPATCH
    if arm == "cho_tools":
        return CHO_FULL_TOOLS, cho_tools.TOOLS
    if arm == "cho_ablated":
        return CHO_ABLATED_TOOLS, {n: cho_tools.TOOLS[n] for n in CHO_ABLATED_TOOLS}
    raise ValueError(f"Unknown arm: {arm}")


def _tool_schemas(tool_names: list[str]) -> list[dict]:
    """Build OpenAI function-calling schemas for the named tools."""
    schemas = []
    # CHO tool schemas
    by_name = {s["name"]: s for s in cho_tools.TOOL_SCHEMAS}
    by_name["lookup_concept_generic"] = {
        "name": "lookup_concept_generic",
        "description": ("Look up a clinical concept. Returns canonical "
                        "name and high-level category (drug, disease, "
                        "lab, etc.). Generic concept-lookup baseline."),
        "parameters": {
            "type": "object",
            "properties": {
                "concept": {"type": "string"}
            },
            "required": ["concept"],
        },
    }
    for name in tool_names:
        s = by_name[name]
        schemas.append({
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s["description"],
                "parameters": s["parameters"],
            },
        })
    return schemas


# =====================================================
# Result dataclass
# =====================================================

@dataclass
class AuditResult:
    arm: str
    example_id: str
    is_hallucination_pred: Optional[bool]   # None if model abstained
    cho_class_pred: Optional[str]
    confidence: Optional[str]               # 'high' | 'medium' | 'low' | None
    rationale: str
    tool_calls: list[dict] = field(default_factory=list)
    raw_response: str = ""
    n_llm_calls: int = 0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "arm": self.arm,
            "example_id": self.example_id,
            "is_hallucination_pred": self.is_hallucination_pred,
            "cho_class_pred": self.cho_class_pred,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "tool_calls": self.tool_calls,
            "raw_response": self.raw_response,
            "n_llm_calls": self.n_llm_calls,
            "error": self.error,
        }


# =====================================================
# Prompt templates
# =====================================================

SYSTEM_PROMPT = """You are a clinical hallucination auditor.

Given a clinical claim, decide whether it contains a hallucination — a \
clinically incorrect or unsupported statement that could mislead a \
clinician or harm a patient.

REQUIRED OUTPUT FORMAT (strict): briefly state your reasoning in 1-2 \
short sentences (under 60 words total), THEN emit the JSON verdict on \
the last line. The JSON verdict is REQUIRED — your response is invalid \
without it. If you are uncertain, emit the JSON anyway with confidence \
"low".

JSON format (exactly):
{"is_hallucination": true|false, "cho_class": "<class_name or null>", \
"confidence": "high"|"medium"|"low", "rationale": "<one sentence>"}

The cho_class field, when is_hallucination is true, should be one of:
  - "FabricatedEntity"     - a clinical entity that does not exist (made-up drug name, fake condition, fake lab, fake procedure)
  - "ForbiddenRelation"    - a drug→treats→condition that is contraindicated or unsupported
  - "InvertedRelation"     - a causal/treatment relation in the wrong direction (e.g. effect→causes→cause)
  - "ThresholdViolation"   - a numeric value outside physiologic bounds
  - "ModifierOmission"     - a negation or qualifier was dropped
Use null for cho_class if is_hallucination is false.

Do NOT spend many tokens reasoning. Two sentences then JSON."""


SYSTEM_PROMPT_TOOLS = SYSTEM_PROMPT + """

You have access to tools that can verify claims against structured \
clinical knowledge. Use them when relevant — they are more reliable \
than your parametric memory for specific facts (drug indications, \
contraindications, lab ranges, ontology classifications). For each \
significant claim in the input, you should consider calling the \
appropriate tool to ground your verdict in evidence.

Reason briefly about which tools to call BEFORE calling them; then call \
them; then give the final JSON verdict on the last line."""


def build_user_message(claim_text: str) -> str:
    return f"Clinical claim to audit:\n\n{claim_text.strip()}"


# =====================================================
# LLM client interface
# =====================================================

class LLMClient(Protocol):
    """Minimal interface the auditor needs from an LLM client."""
    def chat_completion(self, messages: list[dict],
                        tools: list[dict] | None) -> dict:
        ...


class OpenAICompatibleClient:
    """Thin wrapper over the openai SDK pointed at any OpenAI-compatible
    chat-completions endpoint (Qwen/vLLM/OpenAI-proper/Anthropic-proxy)."""

    def __init__(self, model: str,
                 base_url: str | None = None,
                 api_key: str | None = None,
                 max_tokens: int = 4096,
                 temperature: float = 0.0):
        from openai import OpenAI
        self.client = OpenAI(
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
        )
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def chat_completion(self, messages: list[dict],
                        tools: list[dict] | None) -> dict:
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        resp = self.client.chat.completions.create(**kwargs)
        return resp.model_dump()


class MockLLMClient:
    """Test-only client. Returns canned responses based on simple text
    pattern matching. Useful for unit-testing the harness without an
    actual LLM endpoint."""

    def __init__(self, responses: dict[str, dict] | None = None):
        # responses: keyword -> response dict (chat completion shape)
        self.responses = responses or {}
        self.call_log = []

    def chat_completion(self, messages: list[dict],
                        tools: list[dict] | None) -> dict:
        self.call_log.append({"messages": messages, "tools": tools})
        last_msg = messages[-1]["content"] if isinstance(
            messages[-1].get("content"), str) else ""
        # Default: claim something is hallucinated
        verdict_json = ('{"is_hallucination": true, "cho_class": '
                        '"ForbiddenRelation", "confidence": "high", '
                        '"rationale": "Mock verdict."}')
        for keyword, resp in self.responses.items():
            if keyword.lower() in last_msg.lower():
                return resp
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": f"Reasoning omitted in mock.\n\n{verdict_json}",
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }],
            "usage": {"total_tokens": 100},
        }


# =====================================================
# Auditor (main loop)
# =====================================================

def audit_one(client: LLMClient, claim_text: str, arm: str,
              example_id: str = "ex",
              max_tool_iterations: int = 6) -> AuditResult:
    """Run the auditor on a single claim under a single arm."""
    tool_names, dispatch = get_tools_for_arm(arm)
    schemas = _tool_schemas(tool_names) if tool_names else None

    sys_prompt = SYSTEM_PROMPT_TOOLS if tool_names else SYSTEM_PROMPT
    messages: list[dict] = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": build_user_message(claim_text)},
    ]

    result = AuditResult(
        arm=arm, example_id=example_id,
        is_hallucination_pred=None, cho_class_pred=None,
        confidence=None, rationale="",
    )

    final_text = ""
    try:
        for _ in range(max_tool_iterations):
            result.n_llm_calls += 1
            resp = client.chat_completion(messages, schemas)
            choice = resp["choices"][0]
            msg = choice["message"]
            content = msg.get("content") or ""
            tool_calls = msg.get("tool_calls") or []

            # Append assistant message to history (with tool_calls if present)
            assistant_msg = {"role": "assistant", "content": content}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            if not tool_calls:
                final_text = content
                break

            # Execute each tool call and append the tool result message
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                args_str = tc["function"]["arguments"]
                try:
                    args = json.loads(args_str) if args_str else {}
                except json.JSONDecodeError:
                    args = {}
                if fn_name in dispatch:
                    tool_out = dispatch[fn_name](**args)
                else:
                    tool_out = {"error": f"unknown tool {fn_name}"}
                result.tool_calls.append({
                    "name": fn_name, "args": args, "result": tool_out,
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(tool_out)[:4000],
                })
        else:
            # ran out of iterations without a final non-tool response
            result.error = "max_tool_iterations exceeded"
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        return result

    result.raw_response = final_text

    # Parse JSON verdict from the last line
    verdict = _parse_verdict(final_text)

    # Empty / unparseable response: one retry with an explicit "now emit
    # the JSON" follow-up. This handles the Qwen-thinking-eats-budget
    # case where the visible content is empty.
    if verdict is None:
        try:
            messages.append({
                "role": "user",
                "content": ("Please emit ONLY the JSON verdict object now, "
                            "with no other text. Format: "
                            '{"is_hallucination": true|false, '
                            '"cho_class": ..., "confidence": ..., '
                            '"rationale": ...}'),
            })
            result.n_llm_calls += 1
            resp = client.chat_completion(messages, schemas)
            retry_text = resp["choices"][0]["message"].get("content") or ""
            result.raw_response = (final_text + "\n\n[RETRY]\n" + retry_text)
            verdict = _parse_verdict(retry_text)
        except Exception as e:
            pass

    if verdict is None:
        result.error = "failed_to_parse_verdict"
        return result
    result.is_hallucination_pred = verdict.get("is_hallucination")
    result.cho_class_pred = verdict.get("cho_class")
    result.confidence = verdict.get("confidence")
    result.rationale = verdict.get("rationale", "")
    return result


def _parse_verdict(text: str) -> dict | None:
    """Find the last JSON object in the text and return it as a dict."""
    if not text:
        return None
    # Look for a JSON object on any of the last few lines
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    for line in reversed(lines[-5:]):
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    # Fallback: find a {...} substring
    start = text.rfind("{")
    end = text.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


# =====================================================
# Self-test (mock client)
# =====================================================

if __name__ == "__main__":
    print("Running auditor self-test with MockLLMClient (no real LLM)...")
    client = MockLLMClient()

    test_examples = [
        ("ex1", "Lisinopril treats angioedema."),
        ("ex2", "Metformin treats type 2 diabetes mellitus."),
    ]

    for ex_id, txt in test_examples:
        for arm in ["no_tools", "generic_knowledge", "cho_tools", "cho_ablated"]:
            r = audit_one(client, txt, arm, example_id=ex_id)
            print(f"\n--- {ex_id} | arm={arm} ---")
            print(f"  text: {txt}")
            print(f"  pred: hallucination={r.is_hallucination_pred} "
                  f"class={r.cho_class_pred} conf={r.confidence}")
            print(f"  tool calls: {len(r.tool_calls)} "
                  f"({[tc['name'] for tc in r.tool_calls]})")
            print(f"  llm calls:  {r.n_llm_calls}")
            if r.error:
                print(f"  ERROR: {r.error}")
