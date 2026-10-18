import html
import json
from collections import OrderedDict

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


def render_finding(i, item, show_sentence=True):
    status = item["status"]
    with st.container(border=True):
        st.markdown(f"**{LEVELS.get(status, status)}**")
        if item.get("citation"):
            st.caption(f"Citation checked: {item['citation']}")
        if show_sentence:
            st.write(item["sentence"])

        if status in ("verified", "weak_support"):
            box = st.success if status == "verified" else st.warning
            box(f"Source: **{item.get('title', 'Unknown paper')}**")
            if item.get("doi"):
                st.caption(f"DOI: {item['doi']}")
            if item.get("reason"):
                st.info(f"🤖 **LLM reasoning**\n\n{item['reason']}")
            elif "similarity" in item:
                st.caption(
                    f"Embedding similarity: {item['similarity']}"
                )
            if item.get("note"):
                st.caption(item["note"])
        elif status == "exists_no_abstract":
            st.warning("**Abstract unavailable. Support was not checked.**")

            if item.get("title"):
                st.write(
                    f"Candidate record: **{item['title']}**"
                )

            st.caption(
                "The candidate record has not been confirmed as "
                "the cited paper. Check its title, author and year "
                "before treating this citation as a match."
            )

        elif status in ("possible_mismatch", "unresolved"):
            st.error(item.get("note", ""))
        elif status == "uncited_claim":
            st.warning("Factual or empirical claim without a citation.")
            if item.get("llm_note"):
                st.info(f"🤖 LLM assessment: {item['llm_note']}")
            suggestions = item.get("suggestions", [])
            if suggestions:
                with st.expander(f"Potential sources ({len(suggestions)})"):
                    for source in suggestions:
                        st.markdown(f"**{source.get('title', 'Untitled paper')}**")
                        if source.get("score") is not None:
                            st.caption(f"Similarity: {source['score']:.3f}")
                        st.write(source.get("text", ""))
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


st.title("📚 CiteGuard")
st.caption("AI-powered scientific citation and claim verification assistant")

col1, col2, col3 = st.columns(3)
with col1:
    if llm_available():
        st.success("🟢 LLM configured")
    else:
        st.warning("🟡 LLM unavailable, using embedding fallback")
with col2:
    st.success("🟢 RAG enabled")
with col3:
    st.success("🟢 Citation verification enabled")

st.divider()
st.subheader("Manuscript")

if "manuscript" not in st.session_state:
    try:
        with open("sample_manuscript.txt", encoding="utf-8") as file:
            st.session_state["manuscript"] = file.read()
    except OSError:
        st.session_state["manuscript"] = ""

uploaded = st.file_uploader(
    "Upload a manuscript (.txt, .docx or .pdf)",
    type=["txt", "docx", "pdf"],
)
if uploaded is not None:
    file_key = (uploaded.name, uploaded.size)
    if st.session_state.get("loaded_file") != file_key:
        try:
            st.session_state["manuscript"] = load_text(uploaded.name, uploaded.getvalue())
            st.session_state["loaded_file"] = file_key
        except Exception as exc:
            st.error(f"Could not read this file: {exc}")

text = st.text_area("Manuscript text", key="manuscript", height=250, label_visibility="collapsed")

if st.button("🔍 Run CiteGuard", type="primary"):
    if not text.strip():
        st.warning("Please enter or upload a manuscript first.")
    else:
        with st.spinner("Searching scholarly metadata and checking citation support..."):
            report = run_pipeline(text)
            st.session_state["report"] = report
            st.session_state["annotated"] = annotate(parse_manuscript(text), report)
            st.session_state["decisions"] = {}
        for key in [key for key in st.session_state if key.startswith("decision_")]:
            del st.session_state[key]

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
reviewed = sum(1 for decision in decisions.values() if decision != "Pending")

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
    body = "".join(sentence_html(item) for item in st.session_state["annotated"])
    st.markdown(f"<div style='line-height:2.1'>{body}</div>", unsafe_allow_html=True)
    st.caption("The same sentence may contain several citations; each citation is shown once in Findings & review.")

with tab_findings:
    view = st.radio("Show", ["Needs attention", "All findings"], horizontal=True)
    grouped = OrderedDict()
    for index, item in enumerate(report):
        if view == "Needs attention" and item["status"] == "verified":
            continue
        grouped.setdefault(item.get("sentence", ""), []).append((index, item))

    if not grouped:
        st.success("Nothing needs attention.")
    else:
        for sentence, rows in grouped.items():
            with st.container(border=True):
                st.markdown(f"**Sentence with {len(rows)} finding(s)**")
                st.write(sentence)
                for index, item in rows:
                    render_finding(index, item, show_sentence=False)

with tab_export:
    export = [
        {**item, "review_decision": decisions.get(index, "Pending")}
        for index, item in enumerate(report)
    ]
    st.download_button(
        "⬇️ Download JSON Report",
        data=json.dumps(export, indent=2, ensure_ascii=False),
        file_name="citeguard_report.json",
        mime="application/json",
    )
