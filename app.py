import streamlit as st
from citeguard.pipeline import run_pipeline
from citeguard.llm import llm_available

st.set_page_config(page_title="CiteGuard", layout="centered")
st.title("CiteGuard")
st.caption("AI based citation and claim verification assistant")

if llm_available():
    st.success("LLM: connected")
else:
    st.warning("LLM: not connected (set the GROQ_API_KEY environment variable), using embedding fallback")

uploaded = st.file_uploader("Upload a .txt manuscript", type="txt")
if uploaded is not None:
    default_text = uploaded.read().decode("utf-8")
else:
    with open("sample_manuscript.txt") as f:
        default_text = f.read()

text = st.text_area("Manuscript text", value=default_text, height=200)

STATUS_KIND = {
    "unresolved": "error",
    "possible_mismatch": "error",
    "weak_support": "warning",
    "exists_no_abstract": "warning",
    "uncited_claim": "warning",
    "verified": "success",
}

if st.button("Run check"):
    with st.spinner("Verifying citations and claims..."):
        report = run_pipeline(text)

    if not report:
        st.info("No citations or flaggable claims found in this text.")

    for item in report:
        kind = STATUS_KIND.get(item["status"], "info")
        box = getattr(st, kind)
        box(f"[{item['status'].upper()}] {item['sentence']}")

        if item["status"] in ("unresolved", "exists_no_abstract", "possible_mismatch"):
            st.caption(item.get("note", ""))
        elif item["status"] in ("verified", "weak_support"):
            if "reason" in item:
                st.caption(f"matched: {item.get('title', '')}, llm reason: {item['reason']}")
            else:
                st.caption(f"matched: {item.get('title', '')} (similarity {item.get('similarity', '?')})")
        elif item["status"] == "uncited_claim":
            if item.get("llm_note"):
                st.caption(f"llm suggestion: {item['llm_note']}")
            elif item["suggestions"]:
                titles = ", ".join(s["title"] for s in item["suggestions"])
                st.caption(f"candidate sources: {titles}")
            else:
                st.caption("candidate sources: none found yet")