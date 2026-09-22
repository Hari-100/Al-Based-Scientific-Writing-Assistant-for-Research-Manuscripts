# CiteGuard - current codebase

This covers the 50 percent checkpoint before Review 3: Input Parser, Citation
Verification Module (now LLM backed), Claim Verification Module (now LLM
backed), Suggestion Aggregator, the shared vector store, and a Streamlit demo
UI. The Outline and Language Module and the full Author Review Interface are
still scoped for later reviews per the project timeline.

## What each file does

- `citeguard/parser.py` - splits manuscript text into sentences and extracts
  citations. Handles both single citations `(Smith, 2020)` and grouped ones
  `(Smith, 2020; Jones et al., 2021)`. Also guards against "et al." being
  misread as a sentence end.
- `citeguard/embeddings.py` - one shared sentence-transformer model
  (`all-MiniLM-L6-v2`, CPU only).
- `citeguard/vector_store.py` - a small FAISS index caching every real
  abstract the citation checker retrieves.
- `citeguard/citation_apis.py` - resolves a citation against three free
  sources: Semantic Scholar, CrossRef, and arXiv, in that order. Also checks
  whether the retrieved paper's year roughly matches the cited year, and
  caches lookups so the same citation is never fetched twice.
- `citeguard/citation_verifier.py` - for each citation: resolves it, flags a
  year mismatch if found, then judges whether the abstract actually supports
  the sentence. This judgment uses the LLM when available, and falls back to
  embedding similarity when it is not.
- `citeguard/claim_verifier.py` - for uncited sentences that look like a
  claim: searches the vector store for related papers, then asks the LLM to
  turn those candidates into a short, specific suggestion for the author.
- `citeguard/llm.py` - open source model inference through the free Hugging
  Face Inference API. One function judges citation support, one turns
  retrieved candidates into a suggestion. Every call is wrapped so a failure
  or missing token never crashes the pipeline, it just falls back.
- `citeguard/aggregator.py` - merges everything into one report, worst
  issues first.
- `citeguard/pipeline.py` - wires the above together.
- `main.py` - command line entry point.
- `app.py` - the live demo UI (Streamlit), kept separate from all pipeline
  logic, it only imports and calls `run_pipeline`.

## What changed since the last checkpoint

- Citation search now checks three sources instead of one, with a year
  sanity check to catch a right-name-wrong-paper mismatch, and caching so a
  repeated citation is not re-fetched.
- The parser now captures multiple citations grouped in one parenthetical,
  which was silently dropped before.
- An actual LLM (open source model, called through Groq's free
  Inference API) now does two things the embedding-only version could not:
  judge whether an abstract really supports a sentence in plain language,
  and write an actual one-sentence suggestion for an uncited claim instead
  of just listing candidate titles. Every LLM call has a working fallback,
  the system still runs correctly with no token set.
- A Streamlit UI (`app.py`) for the live demo, it shows each flagged
  sentence with a colored status and the reasoning behind it, and shows
  plainly whether the LLM is connected or the system is running on the
  fallback path.

## Setting up the LLM for the demo

Uses Groq instead of Hugging Face, Hugging Face's Inference API now requires
enabling specific providers per account (Settings, Inference Providers) and
kept rejecting every model with "not supported by any provider you have
enabled" regardless of which model was requested, that is an account setting,
not something fixable in code. Groq does not have that friction.

1. Create a free account at console.groq.com and generate an API key.
2. Set it as an environment variable before running anything:
   `export GROQ_API_KEY=your_key_here` (or set it in the Colab/Kaggle
   environment secrets panel).
3. `citeguard/llm.py` tries a short list of models in `MODEL_CANDIDATES` and
   caches whichever one works, so a single model being retired does not
   break the whole system. Check console.groq.com/docs/models for the
   current list if all three ever fail, and update the list.
4. With no key set, everything still works, citation support falls back to
   embedding similarity and claim suggestions fall back to listing candidate
   titles. Good for offline development, not for the actual demo.

This was tested with mocked responses (model fallback, caching, and the
graceful no-key path all confirmed working). The actual Groq API call
itself still needs a real key to verify end to end, this sandbox could
reach api.groq.com but had no valid key to authenticate with.

## Running it

Command line:
```
pip install -r requirements.txt
python main.py sample_manuscript.txt
```

Live demo UI:
```
streamlit run app.py
```
This opens a browser page with a text box (prefilled with the sample
manuscript, or upload your own `.txt`), a Run Check button, and a color
coded report. Confirmed this starts cleanly and serves without errors, the
LLM connection itself still needs testing on your end per the note above.

## Known limitations at this stage

- The claim detector is still a simple keyword and number check, not a
  trained classifier.
- The vector store is in-memory only, resets every run.
- Citation extraction handles parenthetical author-year citations, not
  numbered `[1]` style or footnotes.
- The LLM calls were built and logic-tested with mocked responses. The
  sandbox this was built in could reach api.groq.com but had no real key to
  authenticate with, so the actual call still needs a real run on your end.
- Single sample manuscript for testing so far, needs a real excerpt from
  the team.

## Suggested next steps (in order)

1. Test the LLM path for real with your GROQ_API_KEY on Colab or Kaggle,
   update `MODEL_CANDIDATES` if none of the three are available.
2. Run the whole thing against a real manuscript excerpt from the team, not
   just the sample file, and sanity check the flagged results by hand.
3. Rehearse the live demo end to end at least once before Review 3.
4. Improve claim detection past keyword matching, if time allows.
5. Add the Outline and Language Module as its own file, same pattern as the
   others, one clear job, wired into `pipeline.py`.
6. Author Review Interface (accept/edit/reject controls) stays scoped for
   Review 4, the current UI output is a stand-in for it.