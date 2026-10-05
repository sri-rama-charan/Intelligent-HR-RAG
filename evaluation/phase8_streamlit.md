# Phase 8 — Streamlit Chat Interface Evaluation & Technical Report

## 1. Executive Summary

Phase 8 implements a clean, production-minded presentation layer for the Intelligent HR Help Desk RAG system using Streamlit. The application provides an intuitive conversational interface allowing employees to query official company policy documents, view grounded answers with document-level grouped citations, and optionally inspect retrieved chunk context with relevance scores.

The user interface has been refined to eliminate technical clutter:
- The experimental retrieval mode dropdown was removed from the user UI, locking the production assistant strictly to the recommended **Hybrid + Reranker** architecture (`hybrid_rerank`).
- The sidebar was redesigned to be compact and informative without technical jargon.
- Citations were redesigned to group by document and combine page numbers (e.g. `📄 Leave Policy — pp. 2, 3`), significantly reducing visual noise.
- The answer retains primary visual emphasis, with sources secondary and chunk inspection neatly collapsed.

---

## 2. Production UI Architecture

```
                                  Streamlit Web UI (app.py)
                                             │
               ┌─────────────────────────────┴─────────────────────────────┐
               ▼                                                           ▼
       Compact Sidebar                                             Main Chat Interface
   • Title & Subtitle                                          • Chat History Display
   • Static Status:                                            • Grounded Answer (Primary)
     ✓ Hybrid Retrieval + Reranking                            • Grouped Sources (Secondary)
   • System Overview (11 Policies, etc.)                       • Expandable Context Inspector
   • Collapsible "How it works"                                • Chat Input (`st.chat_input`)
   • Clear Conversation Action                                 • Processing Spinner
               │                                                           │
               └─────────────────────────────┬─────────────────────────────┘
                                             │
                                             ▼
                          RAGPipeline.ask(retrieval_mode="hybrid_rerank")
                                             │
                                FAISS Top-20 + BM25 Top-20
                                             │
                                        RRF Fusion
                                             │
                                     Top-20 Candidates
                                             │
                                   Cross-Encoder Reranker
                                             │
                                          Top-5
                                             │
                                     Gemini Generator
                                             │
                                     RAGResponse Object
```

### Decoupling & Caching Principles
- **No Duplicate Retrieval Logic**: `app.py` does not contain any FAISS indexing, BM25 scoring, RRF logic, or Cross-Encoder operations. All retrieval operations are handled by `RAGPipeline`.
- **Resource Caching (`@st.cache_resource`)**: Initializing the 11 PDF documents, building the FAISS vector index, fitting BM25, and loading `sentence-transformers` and `cross-encoder` models occurs once at application startup. Subsequent queries run without reloading models or rebuilding indices.
- **Production Default**: Every user query runs against `retrieval_mode="hybrid_rerank"`. The underlying `RAGPipeline` retains full support for `"faiss"` and `"hybrid"` backends for evaluation and development scripts.

---

## 3. Sidebar Redesign

The sidebar was streamlined to avoid overwhelming non-technical users with implementation metrics:

```markdown
🏢 HR Help Desk
AI-Powered Policy Assistant

────────────────────
STATUS
✓ Hybrid Retrieval + Reranking

────────────────────
SYSTEM
✓ 11 HR Policies  
✓ Hybrid Retrieval  
✓ Cross-Encoder Reranking  
✓ Gemini Generation  

────────────────────
ℹ️ How it works (Expander)
Question -> Hybrid Retrieval -> RRF Fusion -> Reranking -> Grounded Answer

────────────────────
🗑️ Clear Conversation (Button)
```

**Excluded from UI**:
- Removed selectbox / dropdown for retrieval mode.
- Omitted raw embedding dimensions, vector distance formulas, candidate pool numbers, and Python class names.

---

## 4. Document-Level Grouped Citations

Previous naive citation rendering listed individual chunks sequentially, often repeating document titles multiple times (e.g. 4 separate bullet points for 2 policies).

The updated `group_and_format_sources()` logic:
1. **Deduplicates by Document**: Groups chunks by their parent document while preserving order of appearance.
2. **Aggregates Pages**: Collects and sorts distinct page numbers for each document.
3. **Formats Compactly**:
   - Single page: `📄 Leave Policy — p. 2`
   - Multiple pages: `📄 Leave Policy — pp. 2, 3`
   - Clean Titles: Filenames like `06_Compensation_and_Benefits_Policy.pdf` are mapped to human-readable names (`Compensation & Benefits Policy`).
4. **Adaptive Display**:
   - $\le 2$ documents: Displayed cleanly inline (`**📚 Sources · 2 documents**`).
   - $> 2$ documents: Wrapped in a compact collapsed expander (`📚 Sources · 4 documents`).

---

## 5. Visual Hierarchy & Context Inspection

1. **User Question**: Rendered in standard user chat bubble.
2. **Assistant Answer**: Primary focus, formatted in clean markdown text.
3. **Grouped Sources**: Secondary focus, displayed immediately below the answer.
4. **Retrieved Context**: Collapsed by default inside `🔍 View Retrieved Context ({N} chunks)`:
   - For each Top-5 chunk, shows compact metadata:
     `**Chunk #{idx}**`
     `Source: {source} · Page {page}`
     `Score: {score}`
   - Shows the exact chunk text inside a readable quote block (`> ...`).

---

## 6. Error Handling & Guardrails

1. **Empty Query Handling**: Whitespace-only or empty submissions trigger a mild warning without invoking the RAG pipeline or consuming API tokens.
2. **Missing Configuration**: If `GEMINI_API_KEY` is missing from the environment, `initialize_pipeline()` catches the missing variable and displays a user-friendly configuration alert.
3. **Pipeline Failures**: Any network, quota, or retrieval exception during `pipeline.ask()` is captured and displayed via `st.error` without crashing the Streamlit session or wiping conversation history.
4. **Credential Safety**: No API keys, model parameters, or internal file paths are displayed to the user.

---

## 7. Verification Results

Verification was performed using unit tests and end-to-end Python pipeline testing:

| Verification Check | Target / Action | Observed Behavior | Status |
| :--- | :--- | :--- | :--- |
| **Retrieval Mode Selector** | Check UI & sidebar | No dropdown present; static status shows `✓ Hybrid Retrieval + Reranking` | ✅ PASSED |
| **Sidebar Layout** | Compactness | Compact system overview, collapsible how-it-works, clear chat button | ✅ PASSED |
| **Citation Grouping** | Multiple page chunks | Chunks from same policy combined into single line with `pp. 2, 3` | ✅ PASSED |
| **Document Title Cleaning** | File name formatting | `06_Compensation_and_Benefits_Policy.pdf` $\rightarrow$ `Compensation & Benefits Policy` | ✅ PASSED |
| **Context Inspection** | Expandable section | Compact metadata (`Chunk #1`, `Source`, `Page`, `Score: 4.6395`) and chunk text | ✅ PASSED |
| **Live Query (Q02)** | Carry-forward leave question | Returned grounded answer: *"A maximum of 45 days of Earned Leave may be carried forward..."* | ✅ PASSED |
| **Session State** | History persistence | Conversation history maintained in `st.session_state.messages` | ✅ PASSED |

**Live Gemini API Calls During Verification**: Exactly 1 call (Q02 verification query).

---

## 8. Test Suite Results

- **Command**: `python -m unittest discover tests`
- **Total Tests**: **100 tests** (93 existing pipeline tests + 7 UI helper unit tests in `tests/test_ui_helpers.py`)
- **Failures / Errors**: 0
- **Execution Time**: ~29.4s
- **Gemini API Calls in Unit Tests**: 0 (deterministic unit tests use mocks/stubs)

---

## 9. Summary of Files Created / Modified

| File | Action | Purpose |
| :--- | :--- | :--- |
| `app.py` | Modified | Streamlit chat UI: removed mode dropdown, simplified sidebar, integrated grouped sources. |
| `src/ui/helpers.py` | Modified | Added `group_and_format_sources`, `clean_document_title`, `format_grouped_source`, `format_compact_chunk_score`. |
| `src/ui/__init__.py` | Modified | Exported new UI helper functions. |
| `tests/test_ui_helpers.py` | Modified | Added unit tests for citation grouping, page aggregation, title cleaning, and compact score display (7 tests total). |
| `scripts/verify_phase8_ui.py` | Created | Verification script validating citation grouping, title cleaning, and live hybrid rerank execution. |
| `evaluation/phase8_streamlit.md` | Updated | Comprehensive documentation of final UI/UX architecture and verification. |
