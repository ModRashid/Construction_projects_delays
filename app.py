import os
import json
import hashlib
import streamlit as st
from pdf_processor import extract_pdf_pages, build_chunks
from rag_engine import build_index, retrieve
from risk_engine import calculate_risk
from llm import analyze_with_groq

st.set_page_config(
    page_title="Construction AI Risk Monitor",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp { background:#f4f6f8; }
.block-container { max-width:1400px; padding-top:1.5rem; }
.hero { background:linear-gradient(135deg,#0b1f33,#173a5e); color:white;
        padding:24px 30px; border-radius:16px; margin-bottom:18px; }
.hero h1 { margin:0; font-size:2.05rem; }
.hero p { margin:6px 0 0; opacity:.85; }
.card { background:white; border:1px solid #e3e7eb; border-radius:14px;
        padding:18px; margin-bottom:12px; }
.high { color:#b42318; font-weight:800; }
.medium { color:#b54708; font-weight:800; }
.low { color:#027a48; font-weight:800; }
.small { color:#667085; font-size:.9rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>🏗️ Construction AI Risk Monitor</h1>
<p>Evidence-based project delay and risk analysis using RAG + Groq AI</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Project Analysis")
    uploaded = st.file_uploader(
        "Upload Construction Project PDF",
        type=["pdf"],
        help="Use a progress report, monthly report, weekly report, risk report, meeting minutes, or similar project document."
    )
    top_k = st.slider(
    "Evidence retrieved per risk area",
    1,
    4,
    2
)
    st.caption("The application retrieves relevant evidence from the PDF before asking the AI for analysis.")

if not uploaded:
    st.info("Upload a construction project PDF from the sidebar to start.")
    st.markdown("""
    ### What the application analyzes
    - Progress & schedule
    - Labor and productivity
    - Materials & procurement
    - Equipment
    - Design changes & approvals
    - Weather
    - Communication & coordination
    - Cost & commercial issues
    """)
    st.stop()

pdf_bytes = uploaded.getvalue()
file_hash = hashlib.md5(pdf_bytes).hexdigest()

# Reset analysis when a different PDF is uploaded
if st.session_state.get("file_hash") != file_hash:
    st.session_state.clear()
    st.session_state["file_hash"] = file_hash

try:
    if "chunks" not in st.session_state:
        with st.spinner("Reading and indexing the project PDF..."):
            pages = extract_pdf_pages(pdf_bytes)
            chunks = build_chunks(pages)
            if not chunks:
                st.error("No readable text was found. The PDF may be scanned/image-only and may require OCR.")
                st.stop()
            index, records = build_index(chunks)
            st.session_state["pages"] = pages
            st.session_state["chunks"] = records
            st.session_state["index"] = index
except Exception as e:
    st.error(f"PDF processing failed: {e}")
    st.stop()

st.success(f"Document ready: {len(st.session_state['pages'])} pages • {len(st.session_state['chunks'])} searchable chunks")

if st.button("🔎 Analyze Construction Project", type="primary", use_container_width=True):
    categories = {
        "Progress & Schedule": "planned progress actual progress schedule delay slippage milestones critical path completion status",
        "Labor & Productivity": "labor manpower workforce shortage staffing productivity worker availability site productivity",
        "Materials & Procurement": "material shortage procurement supplier purchase order delivery lead time late material",
        "Equipment": "construction equipment machinery availability breakdown utilization shortage productivity",
        "Design & Approvals": "design change drawing revision engineering issue RFI approval technical query variation pending approval",
        "Weather": "weather rain heat wind storm temperature weather impact construction activity",
        "Communication": "communication coordination meeting stakeholder decision instruction unresolved issue project management",
        "Cost & Commercial": "budget cost payment cash flow financial commercial variation claim contract",
    }

    retrieved = {}
    progress = st.progress(0)
    for i, (category, query) in enumerate(categories.items(), start=1):
        retrieved[category] = retrieve(
            query,
            st.session_state["index"],
            st.session_state["chunks"],
            top_k=top_k,
        )
        progress.progress(i / len(categories))

    risk = calculate_risk(retrieved)

    try:
        with st.spinner("Generating evidence-based construction analysis..."):
            ai = analyze_with_groq(retrieved, risk)
    except Exception as e:
        ai = {"summary": f"Groq analysis failed: {e}", "problems": [], "recommendations": []}

    st.session_state["retrieved"] = retrieved
    st.session_state["risk"] = risk
    st.session_state["ai"] = ai

if "risk" not in st.session_state:
    st.stop()

risk = st.session_state["risk"]
ai = st.session_state["ai"]

c1, c2, c3 = st.columns(3)
c1.metric("Delay Risk Score", f'{risk["score"]}/100')
c2.metric("Risk Level", risk["level"].upper())
c3.metric("Risk Areas Detected", risk["risk_areas"])

level_class = risk["level"].lower()
st.markdown(
    f'<div class="card"><b>Overall Project Delay Risk</b><br>'
    f'<span class="{level_class}" style="font-size:1.7rem">{risk["level"].upper()}</span>'
    f'<br><span class="small">The score is an evidence-based screening score, not a claimed statistical probability.</span></div>',
    unsafe_allow_html=True,
)

st.subheader("🚨 Key Problems Identified")
problems = ai.get("problems", [])
if problems:
    for p in problems:
        with st.expander(f'{p.get("title", "Problem")} — {p.get("severity", "Unknown")}'):
            st.write(p.get("description", ""))
            st.write(f'**Potential impact:** {p.get("impact", "")}')
            sources = p.get("sources", [])
            if sources:
                st.caption("Source evidence: " + ", ".join(sources))
else:
    st.warning("No structured problems were returned. Review the retrieved evidence below.")

st.subheader("🛠️ Recommended Corrective Actions")
recommendations = ai.get("recommendations", [])
if recommendations:
    for r in recommendations:
        st.markdown(f"""
        <div class="card">
        <b>{r.get("action", "Recommended action")}</b><br>
        <b>Priority:</b> {r.get("priority", "N/A")} &nbsp; | &nbsp;
        <b>Responsible area:</b> {r.get("responsible_area", "N/A")}<br>
        <b>Expected impact:</b> {r.get("expected_impact", "N/A")}<br>
        <b>Timeframe:</b> {r.get("timeframe", "N/A")}
        </div>
        """, unsafe_allow_html=True)
else:
    st.warning("No recommendations were returned.")

st.subheader("📋 Project Manager Summary")
st.write(ai.get("summary", "No summary available."))

st.subheader("📚 Retrieved Source Evidence")
for category, items in st.session_state["retrieved"].items():
    with st.expander(f"{category} ({len(items)} evidence chunks)"):
        if not items:
            st.write("No relevant evidence retrieved.")
        for item in items:
            st.markdown(f'**Page {item["page"]} • Evidence relevance {item["score"]:.2f}**')
            st.write(item["text"])
            st.divider()
