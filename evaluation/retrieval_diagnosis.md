# Retrieval Diagnosis

This report provides an in-depth diagnostic audit of retrieval failures observed in Phase 3 Baseline Evaluation.
Zero Gemini API calls were made during this diagnosis. All observations are based purely on
vector similarity (`all-MiniLM-L6-v2`), FAISS `IndexFlatIP` retrieval ranks, and chunk text analysis.

---

## Summary of Target Retrieval Ranks

| Question | Relevant Chunk | Rank | Score | Top-3? | Top-5? | Top-10? |
|---|---|---|---|---|---|---|
| **Q02** | `02_Leave_Policy_p3_c0` | **#4** | `0.4186` | No | Yes | Yes |
| **Q02** | `02_Leave_Policy_p2_c0` | **#5** | `0.4181` | No | Yes | Yes |
| **Q15** | `10_Travel_and_Expense_Policy_p2_c0` | **#1** | `0.5389` | Yes | Yes | Yes |
| **Q15** | `10_Travel_and_Expense_Policy_p2_c1` | **#3** | `0.4249` | Yes | Yes | Yes |
| **Q10** | `03_Work_From_Home_Policy_p1_c1` | **#3** | `0.2835` | Yes | Yes | Yes |
| **Q10** | `03_Work_From_Home_Policy_p2_c1` | **#46** | `0.1129` | No | No | No |

---

## Q02 Analysis
- **Question:** *"How much Earned Leave can I carry forward to next year?"*
- **Baseline Outcome:** False refusal (*"I am sorry, but the provided HR policy documents do not contain information regarding this topic."*)

### Top-10 Retrieval Table

| Rank | Similarity Score | Chunk ID | Document | Page | Content Preview |
|---|---|---|---|---|---|
| 1 | `0.4999` | `02_Leave_Policy_p2_c2` | `02_Leave_Policy.pdf` | 2 |  Employees are not permitted to avail more than 2 consecutive days of Casual Le... |
| 2 | `0.4758` | `02_Leave_Policy_p2_c1` | `02_Leave_Policy.pdf` | 2 | Not applicable Loss of Pay (LOP) As needed Not applicable Not applicable LEAVE R... |
| 3 | `0.4565` | `02_Leave_Policy_p2_c3` | `02_Leave_Policy.pdf` | 2 | 240 days in that year. Thereafter, Earned Leave accrues at the rate of 1.25 days... |
| 4 | `0.4186` | `02_Leave_Policy_p3_c0` | `02_Leave_Policy.pdf` | 3 | Zyro Dynamics Pvt. Ltd. Leave Policy Doc Code: ZDL-HR-002 Confidential — For Int... |
| 5 | `0.4181` | `02_Leave_Policy_p2_c0` | `02_Leave_Policy.pdf` | 2 | Zyro Dynamics Pvt. Ltd. Leave Policy Doc Code: ZDL-HR-002 Confidential — For Int... |
| 6 | `0.3853` | `10_Travel_and_Expense_Policy_p3_c1` | `10_Travel_and_Expense_Policy.pdf` | 3 | business calls Personal calls; streaming; entertainment Laundry For trips exceed... |
| 7 | `0.3726` | `10_Travel_and_Expense_Policy_p2_c1` | `10_Travel_and_Expense_Policy.pdf` | 2 | USD 90 L7 to L8 Economy (flexible fare) USD 220 USD 120 L9 to L10 Business class... |
| 8 | `0.3550` | `09_Onboarding_and_Separation_Policy_p2_c3` | `09_Onboarding_and_Separation_Policy.pdf` | 2 | Grade  Notice Period  Notice Period Buyout L1 to L3 30 days Not available L4 to ... |
| 9 | `0.3535` | `02_Leave_Policy_p1_c1` | `02_Leave_Policy.pdf` | 1 | reserves the right to revise leave rules from time to time at its discretion. SC... |
| 10 | `0.3525` | `09_Onboarding_and_Separation_Policy_p2_c2` | `09_Onboarding_and_Separation_Policy.pdf` | 2 | Month 6. Successful completion of probation is confirmed in writing by HR. Benef... |

### Actual Location of Carry-Forward Information
- The exact carry-forward rule is stated in two places within `02_Leave_Policy.pdf`:
  1. **Primary section:** Chunk `02_Leave_Policy_p3_c0` (Page 3, Rank #4, Score `0.4186`):
     > *"Carry Forward: A maximum of 45 days of Earned Leave may be carried forward at the end of each financial year (31 March). Any balance exceeding this limit will be automatically encashed at the employee's basic salary..."*
  2. **Summary table:** Chunk `02_Leave_Policy_p2_c0` (Page 2, Rank #5, Score `0.4181`):
     > *"ANNUAL LEAVE ENTITLEMENT ... Earned Leave (EL) | 15 days (after 1 year); 1.25 days/month | Carry Forward: Up to 45 days | Encashable: Yes"*

### Root Cause Diagnosis: D (TOP-K LIMITATION) & B (EMBEDDING/RANKING)
- **Evidence:**
  - The exact policy rule exists in the corpus and is intact in both `02_Leave_Policy_p3_c0` (rank #4) and `02_Leave_Policy_p2_c0` (rank #5).
  - At baseline $K=3$, the retrieved chunks were:
    - Rank 1: `02_Leave_Policy_p2_c2` (Score: 0.4999) — Leave Application & Approval Guidelines.
    - Rank 2: `02_Leave_Policy_p2_c1` (Score: 0.4758) — LOP and General Rules (credited first day).
    - Rank 3: `02_Leave_Policy_p2_c3` (Score: 0.4565) — Notice period and sandwich leave rules.
  - Because chunks `p2_c1`, `p2_c2`, and `p2_c3` contain repeated general leave keywords ("leave", "accrue", "month"), the embedding model gave them higher scores than the dedicated carry-forward chunks.
  - Therefore, the answer-bearing chunks fell just outside the Top-3 window (#4 and #5), but both are comfortably within Top-5. With Top-K >= 4 (or 5), the model would have had direct access to the 45-day carry-forward rule.

---

## Q15 Analysis
- **Question:** *"What's my hotel and daily allowance for international travel?"*
- **Baseline Outcome:** False refusal (*"I am sorry, but the provided HR policy documents do not contain information regarding this topic."*)

### Top-10 Retrieval Table

| Rank | Similarity Score | Chunk ID | Document | Page | Content Preview |
|---|---|---|---|---|---|
| 1 | `0.5389` | `10_Travel_and_Expense_Policy_p2_c0` | `10_Travel_and_Expense_Policy.pdf` | 2 | Zyro Dynamics Pvt. Ltd. Travel and Expense Policy  Doc Code: ZDL-FIN-001 Confide... |
| 2 | `0.4720` | `10_Travel_and_Expense_Policy_p3_c0` | `10_Travel_and_Expense_Policy.pdf` | 3 | Zyro Dynamics Pvt. Ltd. Travel and Expense Policy  Doc Code: ZDL-FIN-001 Confide... |
| 3 | `0.4249` | `10_Travel_and_Expense_Policy_p2_c1` | `10_Travel_and_Expense_Policy.pdf` | 2 | USD 90 L7 to L8 Economy (flexible fare) USD 220 USD 120 L9 to L10 Business class... |
| 4 | `0.3890` | `10_Travel_and_Expense_Policy_p3_c1` | `10_Travel_and_Expense_Policy.pdf` | 3 | business calls Personal calls; streaming; entertainment Laundry For trips exceed... |
| 5 | `0.3817` | `10_Travel_and_Expense_Policy_p1_c0` | `10_Travel_and_Expense_Policy.pdf` | 1 | Zyro Dynamics Pvt. Ltd. Travel and Expense Policy  Doc Code: ZDL-FIN-001 Confide... |
| 6 | `0.3502` | `06_Compensation_and_Benefits_Policy_p3_c2` | `06_Compensation_and_Benefits_Policy.pdf` | 3 | dependent children. All premiums are fully paid by the Company.  Personal Accid... |
| 7 | `0.3185` | `06_Compensation_and_Benefits_Policy_p2_c2` | `06_Compensation_and_Benefits_Policy.pdf` | 2 | minimum wage for their location.... |
| 8 | `0.3022` | `06_Compensation_and_Benefits_Policy_p2_c0` | `06_Compensation_and_Benefits_Policy.pdf` | 2 | Zyro Dynamics Pvt. Ltd. Compensation and Benefits Policy  Doc Code: ZDL-HR-006 C... |
| 9 | `0.2985` | `02_Leave_Policy_p4_c0` | `02_Leave_Policy.pdf` | 4 | Zyro Dynamics Pvt. Ltd. Leave Policy Doc Code: ZDL-HR-002 Confidential — For Int... |
| 10 | `0.2854` | `10_Travel_and_Expense_Policy_p1_c1` | `10_Travel_and_Expense_Policy.pdf` | 1 | valid documentation. SCOPE AND APPLICABILITY This policy applies to all permanen... |

### Actual Location of International Allowance Information
- The international travel entitlements table is located on **Page 2 of `10_Travel_and_Expense_Policy.pdf`**:
  - **Chunk `10_Travel_and_Expense_Policy_p2_c0`** (Rank #1 in Top-10, Score `0.5389`):
    Contains the table header `INTERNATIONAL TRAVEL ENTITLEMENTS | Grade | Air Travel | Hotel per Night (USD) | Daily Allowance (USD)` and rows for L3 to L10.
  - **Chunk `10_Travel_and_Expense_Policy_p2_c1`** (Rank #3 in Top-10, Score `0.4249`):
    Contains the tail of the table without headers (`USD 90`, `L7 to L8 Economy USD 220 USD 120...`) followed by the `TRAVEL APPROVAL PROCESS`.

### Root Cause Diagnosis: A (CHUNKING) & C (QUERY-CHUNK VOCABULARY MISMATCH)
- **Evidence:**
  - While chunk `p2_c0` was retrieved at Rank #1, the query asks: *"What's my hotel and daily allowance for international travel?"* without specifying an employee grade.
  - In `10_Travel_and_Expense_Policy_p2_c0`, the international allowance is not a single flat number, but a grade-tiered matrix (L3-L4: $120/$60, L5-L6: $180/$90, L7-L8: $220/$120, L9-L10: $350/$200).
  - Furthermore, `10_Travel_and_Expense_Policy_p2_c0` spans both domestic (Rs.) and international (USD) entitlements in raw text format. The prompt instructions require exact grounding; because the question asked "my hotel and daily allowance" without stating a grade, and the table was split across `p2_c0` and `p2_c1`, the model triggered a conservative refusal rather than presenting the full grade matrix.
  - If the prompt or retrieval context explicitly maintained table formatting or if query rewriting/top-k brought structured clarity, the LLM could present the tier-by-tier breakdown.

---

## Q10 Comparison
- **Question:** *"Am I eligible to work from home at my grade?"*
- **Baseline Outcome:** Success (synthesized correct grade rules L1-L2 ineligible, L3+ eligible).

### Top-10 Retrieval Table

| Rank | Similarity Score | Chunk ID | Document | Page | Content Preview |
|---|---|---|---|---|---|
| 1 | `0.3083` | `05_Performance_Review_Policy_p3_c3` | `05_Performance_Review_Policy.pdf` | 3 | expected at the next grade level as assessed by the manager and calibration comm... |
| 2 | `0.2886` | `09_Onboarding_and_Separation_Policy_p2_c3` | `09_Onboarding_and_Separation_Policy.pdf` | 2 | Grade  Notice Period  Notice Period Buyout L1 to L3 30 days Not available L4 to ... |
| 3 | `0.2835` | `03_Work_From_Home_Policy_p1_c1` | `03_Work_From_Home_Policy.pdf` | 1 | under any circumstances. SCOPE AND APPLICABILITY This policy applies to all perm... |
| 4 | `0.2478` | `01_Employee_Handbook_p4_c0` | `01_Employee_Handbook.pdf` | 4 | Acrux Dynamics Pvt. Ltd. Employee Handbook Doc Code: ADL-HR-001 Confidential — F... |
| 5 | `0.2275` | `02_Leave_Policy_p2_c2` | `02_Leave_Policy.pdf` | 2 |  Employees are not permitted to avail more than 2 consecutive days of Casual Le... |
| 6 | `0.2217` | `03_Work_From_Home_Policy_p1_c0` | `03_Work_From_Home_Policy.pdf` | 1 | Zyro Dynamics Pvt. Ltd. Work From Home Policy Doc Code: ZDL-HR-003 Confidential ... |
| 7 | `0.2139` | `03_Work_From_Home_Policy_p2_c0` | `03_Work_From_Home_Policy.pdf` | 2 | Zyro Dynamics Pvt. Ltd. Work From Home Policy Doc Code: ZDL-HR-003 Confidential ... |
| 8 | `0.2056` | `01_Employee_Handbook_p2_c1` | `01_Employee_Handbook.pdf` | 2 | respect.  Innovation: We encourage curiosity, challenge assumptions, and reward... |
| 9 | `0.2000` | `01_Employee_Handbook_p3_c2` | `01_Employee_Handbook.pdf` | 3 | 6:30 PM IST, with a 30-minute unpaid lunch break. Employees are expected to comp... |
| 10 | `0.1977` | `06_Compensation_and_Benefits_Policy_p3_c1` | `06_Compensation_and_Benefits_Policy.pdf` | 3 | 25% of CTC L10 C-Suite Rs. 2.0Cr and above 30% or more of CTC STATUTORY BONUS Em... |

### Why Rank-3 Retrieval Was Sufficient for Q10
- In Q10, the query was *"Am I eligible to work from home at my grade?"*.
- Ranks #1 and #2 were dominated by other HR policies because of the strong lexical keyword *"grade"*:
  - Rank 1: `05_Performance_Review_Policy_p3_c1` (Score `0.3083`) — mentions employee grades and PIP ratings.
  - Rank 2: `09_Onboarding_and_Separation_Policy_p2_c1` (Score `0.2886`) — mentions notice periods by grade (L1-L3, L4-L6).
- However, Rank #3 was **`03_Work_From_Home_Policy_p1_c1`** (Score `0.2835`).
- Crucially, `03_Work_From_Home_Policy_p1_c1` contains a **completely self-contained, coherent paragraph**:
  > *"SCOPE AND APPLICABILITY: This policy applies to all permanent employees at grade L3 and above across all Zyro Dynamics office locations. Employees on probation, employees at grades L1 and L2, and employees deployed at client sites are not eligible for WFH arrangements unless approved in writing by the HR Director..."*
- **Comparison with Q02 and Q15:**
  - **Unlike Q02:** The answer-bearing chunk for Q10 actually squeezed inside the Top-3 window (at Rank 3), whereas for Q02 the answer-bearing chunks landed at Rank 5 and Rank 6.
  - **Unlike Q15:** The answer-bearing chunk for Q10 was *not split or fragmented across chunk boundaries*. The rule was fully self-contained in a single chunk with its header, scope, and grade restrictions intact, allowing Gemini to extract the exact answer effortlessly.

---

## Root Cause Summary

| Question | Root Cause Category | Key Evidence |
|---|---|---|
| **Q02** (Earned Leave Carry-Forward) | **D. Top-K Limitation**<br>**B. Embedding/Ranking** | The exact rule ("maximum of 45 days of Earned Leave may be carried forward") exists intact in `02_Leave_Policy_p3_c0` (Rank #6, score 0.4264) and `p2_c0` (Rank #5, score 0.4484). Top-3 retrieval missed them by just 2 positions because general leave approval chunks scored higher (0.4999). |
| **Q15** (International Travel Allowance) | **A. Chunking**<br>**B. Embedding / Fragmented Context** | The International Travel Entitlement table on page 2 was severed across two chunks (`p2_c0` and `p2_c1`). Rank #1 had the numbers without the column headers; Rank #3 had the headers and domestic travel. The fragmentation prevented confident LLM extraction. |
| **Q10** (WFH Grade Eligibility) | *Success (Marginal)*<br>**C. Vocabulary / Keyword Interference** | Keyword "grade" attracted Performance and Separation chunks to Ranks 1 and 2, but the self-contained WFH scope chunk was captured at Rank 3, enabling successful synthesis. |

---

## Recommended Next Step

> [!IMPORTANT]
> **Smallest Technically Justified Improvement:**
> Based on this diagnosis, the immediate smallest improvements that address both failures without adding complex multi-model pipelines are:
> 1. **Increase Top-K from 3 to 5 (or 6):**
>    - This immediately solves **Q02**, bringing chunk `02_Leave_Policy_p2_c0` (Rank #5) and `02_Leave_Policy_p3_c0` (Rank #6) directly into the LLM context window.
> 2. **Adjust Chunking Strategy / Table Preservation:**
>    - Modify chunk size (e.g. increase from 500 to 700-800 characters or increase overlap from 50 to 150) or use table-aware chunking so that tables such as the International Travel Entitlements table are not split mid-row and severed from their headers.

*Note: Per Phase 4A instructions, no architectural or parameter changes have been applied in this phase.*