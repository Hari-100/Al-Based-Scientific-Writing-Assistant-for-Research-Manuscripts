import html
import json
import streamlit as st

from citeguard.annotate import LEVELS, annotate
from citeguard.llm import llm_available
from citeguard.loader import load_text
from citeguard.parser import parse_manuscript
from citeguard.pipeline import run_pipeline

st.set_page_config(page_title="CiteGuard", page_icon="📚", layout="wide")

COLORS = {
    "verified": "52,168,83",
    "weak_support": "249,171,0",
    "uncited_claim": "249,171,0",
    "exists_no_abstract": "249,171,0",
    "possible_mismatch": "217,48,37",
    "unresolved": "217,48,37",
}
DECISIONS = ["Pending", "Accepted as is", "Needs revision", "Dismissed"]


def sentence_html(a):
    text = html.escape(a["sentence"].replace("\n", " ")).replace("$", "&#36;")
    if a["status"] is None:
        return f"{text} "
    rgb = COLORS.get(a["status"], "128,128,128")
    level = html.escape(a["level"])
    return (
        f"<span style='background:rgba({rgb},0.18);border-bottom:2px solid rgb({rgb});"
        f"padding:1px 3px;border-radius:3px'>{text}</span> "
        f"<sup style='color:rgb({rgb})'>[{level}]</sup> "
    )


def set_decision(i):
    st.session_state["decisions"][i] = st.session_state[f"decision_{i}"]


def render_finding(i, item):
    status = item["status"]
    with st.container(border=True):
        st.markdown(f"**{i + 1}. {LEVELS.get(status, status)}**")
        st.write(item["sentence"])

        if status in ("verified", "weak_support"):
            box = st.success if status == "verified" else st.warning
            box(f"Source: **{item.get('title', 'Unknown paper')}**")
            if item.get("reason"):
                st.caption(f"LLM reasoning: {item['reason']}")
            elif "similarity" in item:
                st.caption(f"Embedding similarity: {item['similarity']}")

        elif status in ("possible_mismatch", "unresolved", "exists_no_abstract"):
            box = st.warning if status == "exists_no_abstract" else st.error
            box(item.get("note", ""))

        elif status == "uncited_claim":
            st.warning("Factual or empirical claim without a citation.")
            if item.get("llm_note"):
                st.info(f"🤖 LLM assessment: {item['llm_note']}")
            suggestions = item.get("suggestions", [])
            if suggestions:
                with st.expander(f"Potential sources ({len(suggestions)})"):
                    for s in suggestions:
                        st.markdown(f"**{s['title']}**")
                        st.caption(f"Similarity: {s['score']:.3f}")
                        st.write(s["text"])
            else:
                st.caption("No potentially relevant sources were retrieved.")

        current = st.session_state["decisions"].get(i, "Pending")
        st.radio(
            "Review decision",
            DECISIONS,
            index=DECISIONS.index(current),
            key=f"decision_{i}",
            horizontal=True,
            on_change=set_decision,
            args=(i,),
        )


# ---------- Header ----------

st.title("📚 CiteGuard")
st.caption("AI-powered scientific citation and claim verification assistant")

col1, col2, col3 = st.columns(3)
with col1:
    if llm_available():
        st.success("🟢 LLM configured")
    else:
        st.warning("🟡 LLM unavailable (set GROQ_API_KEY), using embedding fallback")
with col2:
    st.success("🟢 RAG enabled")
with col3:
    st.success("🟢 Citation verification enabled")

st.divider()

# ---------- Manuscript input ----------

st.subheader("Manuscript")

if "manuscript" not in st.session_state:
    with open("sample_manuscript.txt", encoding="utf-8") as f:
        st.session_state["manuscript"] = f.read()

uploaded = st.file_uploader("Upload a manuscript (.txt, .docx or .pdf)", type=["txt", "docx", "pdf"])
if uploaded is not None:
    file_key = (uploaded.name, uploaded.size)
    if st.session_state.get("loaded_file") != file_key:
        try:
            st.session_state["manuscript"] = load_text(uploaded.name, uploaded.getvalue())
            st.session_state["loaded_file"] = file_key
        except Exception as e:
            st.error(f"Could not read this file: {e}")

text = st.text_area("Manuscript text", key="manuscript", height=250, label_visibility="collapsed")

if st.button("🔍 Run CiteGuard", type="primary"):
    if not text.strip():
        st.warning("Please enter or upload a manuscript first.")
    else:
        with st.spinner("Analyzing manuscript and verifying citations..."):
            report = run_pipeline(text)
            st.session_state["report"] = report
            st.session_state["annotated"] = annotate(parse_manuscript(text), report)
            st.session_state["decisions"] = {}
        for k in [k for k in st.session_state if k.startswith("decision_")]:
            del st.session_state[k]

# ---------- Results ----------

report = st.session_state.get("report")
if report is None:
    st.stop()
if not report:
    st.info("No citations or flaggable claims were found.")
    st.stop()

decisions = st.session_state["decisions"]
counts = {}
for item in report:
    counts[item["status"]] = counts.get(item["status"], 0) + 1
reviewed = sum(1 for d in decisions.values() if d != "Pending")

st.divider()
st.subheader("Verification Summary")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Supported", counts.get("verified", 0))
m2.metric("Weak support", counts.get("weak_support", 0))
m3.metric("Citation issues", counts.get("possible_mismatch", 0) + counts.get("unresolved", 0))
m4.metric("Uncited claims", counts.get("uncited_claim", 0))
m5.metric("Reviewed", f"{reviewed} / {len(report)}")

tab_text, tab_findings, tab_export = st.tabs(["Annotated manuscript", "Findings & review", "Export"])

with tab_text:
    legend = " ".join(
        f"<span style='background:rgba({COLORS[s]},0.18);border-bottom:2px solid rgb({COLORS[s]});"
        f"padding:1px 6px;border-radius:3px'>{LEVELS[s]}</span>"
        for s in ("verified", "weak_support", "possible_mismatch", "unresolved")
    )
    st.markdown(legend, unsafe_allow_html=True)
    body = "".join(sentence_html(a) for a in st.session_state["annotated"])
    st.markdown(f"<div style='line-height:2.1'>{body}</div>", unsafe_allow_html=True)
    st.caption("Sentences with no citation and no checkable claim are left unmarked.")

with tab_findings:
    view = st.radio("Show", ["Needs attention", "All findings"], horizontal=True)
    shown = 0
    for i, item in enumerate(report):
        if view == "Needs attention" and item["status"] == "verified":
            continue
        render_finding(i, item)
        shown += 1
    if shown == 0:
        st.success("Nothing needs attention.")

with tab_export:
    export = [{**item, "review_decision": decisions.get(i, "Pending")} for i, item in enumerate(report)]
    st.download_button(
        "⬇️ Download JSON Report",
        data=json.dumps(export, indent=2, ensure_ascii=False),
        file_name="citeguard_report.json",
        mime="application/json",
    )