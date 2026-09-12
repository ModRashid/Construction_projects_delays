import json
import os
from groq import Groq

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
    blocks = []

    for category, items in retrieved.items():
        blocks.append(f"### {category}")

        for item in items:
            blocks.append(
                f'[Page {item["page"]} | Chunk {item["chunk_id"]} | '
                f'Relevance {item["score"]:.2f}]\n{item["text"]}'
            )

    return "\n\n".join(blocks)

def analyze_with_groq(retrieved, risk):
    api_key = get_api_key()

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it under Streamlit Cloud → App settings → Secrets."
        )

    client = Groq(api_key=api_key)
    context = build_context(retrieved)

    system_prompt = """
You are a senior construction project controls and delay-risk analyst.

You are working inside a RAG system. The supplied context consists ONLY of
retrieved evidence from a construction project PDF.

Strict rules:
1. Do not invent facts, dates, quantities, costs, percentages, manpower,
   delays, or project conditions.
2. Every factual problem must be supported by retrieved evidence.
3. Use page references from the context as sources.
4. If the document does not contain enough information, explicitly say
   "Insufficient information in the uploaded document."
5. Separate evidence-based findings from recommendations.
6. Recommendations should be practical for construction project management.
7. Do not claim that the screening score is a statistically trained
   probability.
8. Focus on schedule, critical activities, procurement, resources,
   approvals, design, equipment, productivity, weather, communication,
   commercial issues, and other genuine project-delay drivers.

Return ONLY valid JSON in exactly this structure:

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

    user_prompt = f"""
The evidence engine calculated:
Risk screening score: {risk["score"]}/100
Risk screening level: {risk["level"]}
Risk areas detected: {risk["risk_areas"]}

Analyze the following retrieved project evidence:

{context}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return json.loads(response.choices[0].message.content)
