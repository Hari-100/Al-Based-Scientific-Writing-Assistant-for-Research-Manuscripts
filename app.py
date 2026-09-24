import json
import streamlit as st

from citeguard.pipeline import run_pipeline
from citeguard.llm import llm_available


st.set_page_config(
    page_title="CiteGuard",
    page_icon="📚",
    layout="wide",
)


# ---------- Header ----------

st.title("📚 CiteGuard")
st.caption(
    "AI-powered scientific citation and claim verification assistant"
)

st.markdown(
    """
    CiteGuard analyzes a manuscript to:
    - verify cited papers
    - detect citation mismatches
    - identify unsupported claims
    - retrieve potentially relevant sources
    """
)


# ---------- System status ----------

col1, col2, col3 = st.columns(3)

with col1:
    if llm_available():
        st.success("🟢 LLM configured")
    else:
        st.warning("🟡 LLM unavailable")

with col2:
    st.success("🟢 RAG enabled")

with col3:
    st.success("🟢 Citation verification enabled")


st.divider()


# ---------- Manuscript input ----------

st.subheader("Manuscript")

uploaded = st.file_uploader(
    "Upload a plain-text manuscript",
    type=["txt"],
)

if uploaded is not None:
    default_text = uploaded.read().decode("utf-8")
else:
    with open("sample_manuscript.txt", encoding="utf-8") as f:
        default_text = f.read()

text = st.text_area(
    "Manuscript text",
    value=default_text,
    height=250,
    label_visibility="collapsed",
)


# ---------- Status configuration ----------

STATUS_KIND = {
    "unresolved": "error",
    "possible_mismatch": "error",
    "weak_support": "warning",
    "exists_no_abstract": "warning",
    "uncited_claim": "warning",
    "verified": "success",
}


STATUS_LABELS = {
    "verified": "Verified",
    "weak_support": "Weak Support",
    "possible_mismatch": "Citation Mismatch",
    "unresolved": "Unresolved Citation",
    "exists_no_abstract": "No Abstract",
    "uncited_claim": "Uncited Claim",
}


# ---------- Run ----------

if st.button(
    "🔍 Run CiteGuard",
    type="primary",
    use_container_width=True,
):

    if not text.strip():
        st.warning("Please enter or upload a manuscript first.")
        st.stop()

    with st.spinner("Analyzing manuscript and verifying citations..."):
        report = run_pipeline(text)

    if not report:
        st.info("No citations or flaggable claims were found.")
        st.stop()

    # ---------- Summary ----------

    counts = {}

    for item in report:
        status = item["status"]
        counts[status] = counts.get(status, 0) + 1

    st.subheader("Verification Summary")

    summary_cols = st.columns(4)

    with summary_cols[0]:
        st.metric(
            "Verified",
            counts.get("verified", 0),
        )

    with summary_cols[1]:
        st.metric(
            "Needs Review",
            counts.get("weak_support", 0),
        )

    with summary_cols[2]:
        st.metric(
            "Citation Issues",
            counts.get("possible_mismatch", 0)
            + counts.get("unresolved", 0),
        )

    with summary_cols[3]:
        st.metric(
            "Uncited Claims",
            counts.get("uncited_claim", 0),
        )

    st.divider()

    # ---------- Detailed results ----------

    st.subheader("Detailed Findings")

    for i, item in enumerate(report, start=1):

        status = item["status"]
        label = STATUS_LABELS.get(
            status,
            status.replace("_", " ").title(),
        )

        kind = STATUS_KIND.get(status, "info")

        with st.container(border=True):

            st.markdown(
                f"**{i}. {label}**"
            )

            st.write(item["sentence"])

            if status == "verified":

                st.success(
                    f"Supported by: **{item.get('title', 'Unknown paper')}**"
                )

                if item.get("reason"):
                    st.caption(
                        f"LLM reasoning: {item['reason']}"
                    )

            elif status == "weak_support":

                st.warning(
                    f"Potentially weak support from: "
                    f"**{item.get('title', 'Unknown paper')}**"
                )

                if item.get("reason"):
                    st.caption(
                        f"LLM reasoning: {item['reason']}"
                    )
                elif "similarity" in item:
                    st.caption(
                        f"Embedding similarity: "
                        f"{item['similarity']}"
                    )

            elif status == "possible_mismatch":

                st.error(
                    f"Possible citation mismatch: "
                    f"**{item.get('title', 'Unknown paper')}**"
                )

                st.caption(
                    item.get("note", "")
                )

            elif status == "unresolved":

                st.error("Citation could not be resolved.")

                st.caption(
                    item.get("note", "")
                )

            elif status == "exists_no_abstract":

                st.warning(
                    f"Paper found: **{item.get('title', 'Unknown paper')}**"
                )

                st.caption(
                    item.get("note", "")
                )

            elif status == "uncited_claim":

                st.warning(
                    "This sentence appears to contain a factual or "
                    "empirical claim without a citation."
                )

                if item.get("llm_note"):

                    st.info(
                        f"🤖 LLM assessment: {item['llm_note']}"
                    )

                suggestions = item.get("suggestions", [])

                if suggestions:

                    with st.expander(
                        f"Potential sources ({len(suggestions)})"
                    ):

                        for suggestion in suggestions:

                            st.markdown(
                                f"**{suggestion['title']}**"
                            )

                            st.caption(
                                f"Similarity: "
                                f"{suggestion['score']:.3f}"
                            )

                            st.write(
                                suggestion["text"]
                            )

                            st.divider()

                else:
                    st.caption(
                        "No potentially relevant sources were retrieved."
                    )


    # ---------- Download ----------

    st.divider()

    st.subheader("Export")

    report_json = json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
    )

    st.download_button(
        "⬇️ Download JSON Report",
        data=report_json,
        file_name="citeguard_report.json",
        mime="application/json",
        use_container_width=True,
    )