# Phase 4B — Top-K = 5 Experiment

## Change

Top-K parameter configured in `RAGPipeline` (and `evaluator.py`) was changed:
- **Previous:** `Top-K = 3`
- **Updated:** `Top-K = 5`

No changes were made to chunking, embeddings, prompt templates, Gemini LLM generation, or advanced retrieval mechanisms (no reranking, MMR, query rewriting, or hybrid search).

---

## Test Results

- **Test Suite:** Complete regression test suite (`tests/`) executed via Python `unittest`.
- **Passed:** **56 / 56 tests passed** (100%).
- **Failed:** 0.
- **External Calls:** Zero Gemini API calls were made during testing.

---

## Retrieval Results

| Question | Correct Chunk Rank | In Top-5? | Observation |
| :--- | :---: | :---: | :--- |
| **Q02** (Earned Leave Carry-Forward) | **#4** (`02_Leave_Policy_p3_c0`)<br>**#5** (`02_Leave_Policy_p2_c0`) | **Yes** | **Fixed at the retrieval level.** In Phase 4A ($K=3$), both carry-forward chunks were excluded (ranks #4 and #5). With $K=5$, both the detailed policy section (45 days rule, score `0.4186`) and the annual entitlement table (score `0.4181`) enter the context window. |
| **Q15** (International Travel Entitlement) | **#1** (`10_Travel_and_Expense_Policy_p2_c0`)<br>**#3** (`10_Travel_and_Expense_Policy_p2_c1`) | **Yes** | **Both relevant chunks are now retrieved together.** Rank #1 (score `0.5389`) provides the table header and grades L3–L6; Rank #3 (score `0.4249`) provides grades L7–L10. However, the table remains split across two chunks due to character chunking. |
| **Q10** (WFH Grade Eligibility) | **#3** (`03_Work_From_Home_Policy_p1_c1`) | **Yes** | **No regression.** The self-contained scope chunk defining L3+ eligibility and L1–L2 ineligibility remains solidly positioned at Rank #3 (score `0.2835`). |

---

## Comparison with Phase 4A

1. **Q02 Retrieval Resolution:**
   - In Phase 4A, Q02 suffered a false refusal because the answer-bearing chunks were at ranks #4 and #5, directly outside the Top-3 boundary.
   - In Phase 4B ($K=5$), both chunks are successfully captured within the retrieval set. This directly resolves the Top-K limitation identified in Phase 4A without requiring complex retrieval architectures.

2. **Q15 Context Availability vs. Chunking Fragmentation:**
   - In Phase 4A, chunk `10_Travel_and_Expense_Policy_p2_c0` was retrieved at rank #1 and `p2_c1` was at rank #3. Both chunks were theoretically in Top-3, but the model refused due to table fragmentation and ambiguous question phrasing ("my allowance" without a grade).
   - In Phase 4B ($K=5$), both chunks remain present (Ranks 1 and 3), and additional travel context (flight rules, per diem limits) is available at Ranks 2, 4, and 5. While retrieval includes all required information, the underlying chunking boundary still severs the International Travel Entitlements table.

3. **Q10 Stability:**
   - Q10 successfully answered under baseline ($K=3$) because chunk `03_Work_From_Home_Policy_p1_c1` was at Rank #3. Under $K=5$, it maintains Rank #3 without being displaced, ensuring zero degradation in precision.

---

## Conclusion

**Top-K = 5 should be retained.**

- **Rationale:**
  1. It immediately cures the retrieval cutoff failure on **Q02**, making both the explicit 45-day carry-forward rule and the summary table available to the LLM.
  2. It introduces zero regression to previously passing queries like **Q10** and retains all 56 passing unit tests.
  3. The increase from 3 to 5 chunks adds only ~1,600 characters of context, which is well within the context window and token budget of `gemini-flash-lite-latest`.
  4. It requires no additional dependencies, latency overhead, or architectural complexity.

- **Remaining Issue:**
  - While $K=5$ solves the retrieval cutoff for Q02, **Q15** still suffers from tabular chunk boundary fragmentation (table headers in chunk 1, subsequent grade rows in chunk 3). Table-aware chunking or increased chunk overlap is the logical next focus.
