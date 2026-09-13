import json
import os
import re
from groq import Groq


# ============================================================
# Groq configuration
# ============================================================

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# These limits are intentionally conservative because your
# current Groq organization has an 8,000 TPM limit.
MAX_CHUNKS_TOTAL = 12
MAX_CHARS_PER_CHUNK = 700
MAX_CONTEXT_CHARS = 7000
MAX_COMPLETION_TOKENS = 1000


def get_api_key():
    """Read the Groq key from Streamlit Secrets or environment variables."""

    try:
        import streamlit as st

        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]

    except Exception:
        pass

    return os.getenv("GROQ_API_KEY")


def build_context(retrieved):
    """
    Build a compact RAG context.

    The RAG engine may retrieve many chunks across multiple risk categories.
    This function:
    1. Combines all retrieved evidence.
    2. Sorts by semantic relevance.
    3. Removes duplicate chunks.
    4. Limits each chunk's size.
    5. Limits the total context size.
    6. Keeps the strongest evidence first.
    """

    all_items = []

    # --------------------------------------------------------
    # Collect all retrieved evidence
    # --------------------------------------------------------

    for category, items in retrieved.items():

        if not items:
            continue

        for item in items:

            text = item.get("text", "").strip()

            if not text:
                continue

            all_items.append({
                "category": category,
                "page": item.get("page"),
                "chunk_id": item.get("chunk_id"),
                "score": float(item.get("score", 0)),
                "text": text,
            })

    # --------------------------------------------------------
    # Strongest semantic evidence first
    # --------------------------------------------------------

    all_items.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    blocks = []
    seen_chunks = set()
    total_chars = 0

    # --------------------------------------------------------
    # Select strongest unique evidence
    # --------------------------------------------------------

    for item in all_items:

        # Unique identifier for deduplication
        chunk_key = (
            item["page"],
            item["chunk_id"]
        )

        if chunk_key in seen_chunks:
            continue

        seen_chunks.add(chunk_key)

        # Clean whitespace
        text = re.sub(
            r"\s+",
            " ",
            item["text"]
        ).strip()

        # Limit individual evidence chunk
        text = text[:MAX_CHARS_PER_CHUNK]

        block = (
            f'[{item["category"]} | '
            f'Page {item["page"]} | '
            f'Chunk {item["chunk_id"]} | '
            f'Relevance {item["score"]:.3f}]\n'
            f'{text}'
        )

        # Stop if adding this evidence would exceed
        # the total context limit.
        if (
            total_chars + len(block)
            > MAX_CONTEXT_CHARS
        ):
            continue

        blocks.append(block)

        total_chars += len(block)

        # Maximum number of evidence chunks
        if len(blocks) >= MAX_CHUNKS_TOTAL:
            break

        # Safety condition
        if total_chars >= MAX_CONTEXT_CHARS:
            break

    return "\n\n".join(blocks)


def analyze_with_groq(retrieved, risk):

    # --------------------------------------------------------
    # API key
    # --------------------------------------------------------

    api_key = get_api_key()

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it under "
            "Streamlit Cloud → App settings → Secrets."
        )

    client = Groq(api_key=api_key)

    # --------------------------------------------------------
    # Compact RAG evidence
    # --------------------------------------------------------

    context = build_context(retrieved)

    if not context:
        raise RuntimeError(
            "No relevant evidence was retrieved from the PDF."
        )

    # --------------------------------------------------------
    # System prompt
    # --------------------------------------------------------

    system_prompt = """
You are a senior construction project controls and delay-risk analyst.

You are working inside a RAG system.

The supplied context consists ONLY of retrieved evidence from a
construction project PDF.

STRICT RULES:

1. Do not invent facts.
2. Do not invent dates.
3. Do not invent quantities.
4. Do not invent costs.
5. Do not invent percentages.
6. Do not invent manpower numbers.
7. Do not invent delay durations.
8. Do not invent project conditions.

9. Every factual problem must be supported by the supplied evidence.

10. Use the page numbers provided in the evidence as sources.

11. If the document does not contain enough information to support
a finding, say:
"Insufficient information in the uploaded document."

12. Separate evidence-based findings from recommendations.

13. Recommendations should be practical construction project
management actions.

14. Do not describe the risk screening score as a statistical
probability.

15. The screening score is an evidence-based screening score.

16. Focus on genuine construction delay drivers including:
- Progress and schedule
- Critical activities
- Labor
- Productivity
- Materials
- Procurement
- Equipment
- Design
- Approvals
- Weather
- Communication
- Coordination
- Cost
- Commercial issues

17. Do not create a problem simply because a risk category exists.

18. Prioritize the strongest and most relevant retrieved evidence.

19. Maximum 6 problems.

20. Maximum 8 recommendations.

RETURN ONLY VALID JSON.

Use exactly this structure:

{
    "summary": "Executive project manager summary.",

    "problems": [
        {
            "title": "Specific problem",
            "severity": "High/Medium/Low",
            "description": "Evidence-based explanation.",
            "impact": "Potential project impact.",
            "sources": ["Page X", "Page Y"]
        }
    ],

    "recommendations": [
        {
            "action": "Specific corrective action",
            "priority": "High/Medium/Low",
            "responsible_area": "Project Management/Planning/Procurement/etc.",
            "expected_impact": "Expected benefit.",
            "timeframe": "Immediate/Within 1 week/etc."
        }
    ]
}
"""

    # --------------------------------------------------------
    # User prompt
    # --------------------------------------------------------

    user_prompt = f"""
The evidence engine calculated:

Risk screening score: {risk["score"]}/100

Risk screening level: {risk["level"]}

Risk areas detected:
{risk["risk_areas"]}

Analyze ONLY the following retrieved project evidence.

Use ONLY the supplied evidence when identifying factual
project problems.

Do not use outside knowledge to create project facts.

RETRIEVED PROJECT EVIDENCE:

{context}
"""

    # --------------------------------------------------------
    # Groq request
    # --------------------------------------------------------

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.1,

        # Keep the generated response small.
        max_completion_tokens=MAX_COMPLETION_TOKENS,

        response_format={
            "type": "json_object"
        },

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            },
        ],
    )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError(
            "Groq returned an empty response."
        )

    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    try:
        return json.loads(content)

    except json.JSONDecodeError:

        # Sometimes models may still wrap JSON in markdown.
        cleaned = content.strip()

        if cleaned.startswith("```"):
            cleaned = re.sub(
                r"^```(?:json)?\s*",
                "",
                cleaned
            )

            cleaned = re.sub(
                r"\s*```$",
                "",
                cleaned
            )

        try:
            return json.loads(cleaned)

        except json.JSONDecodeError as exc:

            raise RuntimeError(
                "Groq returned an invalid JSON response."
            ) from exc
