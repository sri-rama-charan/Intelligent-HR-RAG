# Phase 3: Baseline Evaluation Report

## Summary
- **Total Questions Evaluated:** 20
- **Successful Generations:** 19
- **Failed Generations:** 1
- **In-Scope HR Policy Questions (Q01–Q15):** 15
- **Out-of-Scope / Edge-Case Questions (Q16–Q20):** 5

## Local Retrieval Analysis
- **Average Top-1 Similarity (Q01–Q15, In-Scope):** `0.5109`
- **Average Top-1 Similarity (Q16–Q20, Out-of-Scope):** `0.6046`

> [!NOTE]
> Similarity scores represent geometric closeness in the embedding space (`all-MiniLM-L6-v2`),
> not statistical probability or answer correctness.

---

## Q01–Q15: In-Scope Results

### [Q01] How does my Earned Leave accrue every month?
- **Generated Answer:** Earned Leave accrues at the rate of 1.25 days per month for employees who have completed one year of continuous service. Employees in their probation period accrue Earned Leave at the rate of 0.5 days per month, which becomes available for use only after probation confirmation.
- **Top Retrieved Sources:** `02_Leave_Policy.pdf;02_Leave_Policy.pdf;02_Leave_Policy.pdf`
- **Source Pages:** `2;2;2`
- **Retrieval Similarity Scores:** `0.5165;0.4933;0.4774`

### [Q02] How much Earned Leave can I carry forward to next year?
- **Generated Answer:** I am sorry, but the provided HR policy documents do not contain information regarding this topic.
- **Top Retrieved Sources:** `02_Leave_Policy.pdf;02_Leave_Policy.pdf;02_Leave_Policy.pdf`
- **Source Pages:** `2;2;2`
- **Retrieval Similarity Scores:** `0.4999;0.4758;0.4565`

### [Q03] How many weeks of Maternity Leave am I entitled to?
- **Generated Answer:** Female employees who have completed a minimum of 80 days of service in the 12 months preceding the expected date of delivery are entitled to 26 weeks of paid Maternity Leave for the first two live births. For a third child, the entitlement is 12 weeks.
- **Top Retrieved Sources:** `02_Leave_Policy.pdf;02_Leave_Policy.pdf;02_Leave_Policy.pdf`
- **Source Pages:** `3;2;3`
- **Retrieval Similarity Scores:** `0.7397;0.5132;0.4768`

### [Q04] Do I need a medical certificate for sick leave?
- **Generated Answer:** Sick Leave taken for more than 2 consecutive days requires a Medical Certificate from a registered medical practitioner, to be submitted within 3 working days of returning to work.
- **Top Retrieved Sources:** `02_Leave_Policy.pdf;02_Leave_Policy.pdf;02_Leave_Policy.pdf`
- **Source Pages:** `2;2;1`
- **Retrieval Similarity Scores:** `0.6174;0.4393;0.3409`

### [Q05] What date does my salary get credited every month?
- **Generated Answer:** Salaries and professional fees are processed and credited to the employee's registered bank account by the 7th of the following month.
- **Top Retrieved Sources:** `06_Compensation_and_Benefits_Policy.pdf;06_Compensation_and_Benefits_Policy.pdf;06_Compensation_and_Benefits_Policy.pdf`
- **Source Pages:** `1;1;2`
- **Retrieval Similarity Scores:** `0.5314;0.4961;0.3513`

### [Q06] What's the salary range for an L4 Senior employee?
- **Generated Answer:** The salary range (CTC Range) for an L4 Senior employee is Rs. 16.0L to Rs. 26.0L per annum.
- **Top Retrieved Sources:** `06_Compensation_and_Benefits_Policy.pdf;06_Compensation_and_Benefits_Policy.pdf;00_Company_Profile.pdf`
- **Source Pages:** `3;2;4`
- **Retrieval Similarity Scores:** `0.4665;0.4569;0.3886`

### [Q07] What medical insurance coverage does the company provide?
- **Generated Answer:** Based on the provided HR policy documents, the company provides Group Medical Insurance with coverage of up to Rs. 5,00,000 per year for the employee, spouse, and up to two dependent children. All premiums are fully paid by the Company.
- **Top Retrieved Sources:** `06_Compensation_and_Benefits_Policy.pdf;06_Compensation_and_Benefits_Policy.pdf;06_Compensation_and_Benefits_Policy.pdf`
- **Source Pages:** `3;2;3`
- **Retrieval Similarity Scores:** `0.4284;0.3896;0.3873`

### [Q08] When am I put on a Performance Improvement Plan?
- **Generated Answer:** Based on the provided HR policy, an employee is placed on a formal Performance Improvement Plan if they receive a rating of 1 or 2 in two consecutive review cycles.
- **Top Retrieved Sources:** `05_Performance_Review_Policy.pdf;05_Performance_Review_Policy.pdf;05_Performance_Review_Policy.pdf`
- **Source Pages:** `3;1;2`
- **Retrieval Similarity Scores:** `0.5525;0.5401;0.4509`

### [Q09] What's the timeline for the Annual Performance Review?
- **Generated Answer:** Based on the provided HR policy, the Annual Performance Review (APR) takes place annually in March.
- **Top Retrieved Sources:** `05_Performance_Review_Policy.pdf;05_Performance_Review_Policy.pdf;05_Performance_Review_Policy.pdf`
- **Source Pages:** `1;3;2`
- **Retrieval Similarity Scores:** `0.5313;0.4927;0.4662`

### [Q10] Am I eligible to work from home at my grade?
- **Generated Answer:** To determine your eligibility to work from home, it depends on your employment status, grade, and location as outlined in the Work From Home Policy:

* This policy applies to all permanent employees at grade L3 and above across all Zyro Dynamics office locations.
* Employees on probation, employees at grades L1 and L2, and employees deployed at client sites are not eligible for WFH arrangements unless approved in writing by the HR Director on a case-by-case basis.
- **Top Retrieved Sources:** `05_Performance_Review_Policy.pdf;09_Onboarding_and_Separation_Policy.pdf;03_Work_From_Home_Policy.pdf`
- **Source Pages:** `3;2;1`
- **Retrieval Similarity Scores:** `0.3083;0.2886;0.2835`

### [Q11] Can I accept a gift from a client or vendor?
- **Generated Answer:** Based on the Zyro Dynamics Pvt. Ltd. HR policy documents, the rules regarding receiving gifts are as follows:

* **Receiving a gift under Rs. 1,000:** Permitted, but must be declared to your reporting manager.
* **Receiving a gift over Rs. 1,000:** Not permitted; it must be politely declined or surrendered to HR.
* **Accepting cash from any party:** Never allowed; it results in immediate disciplinary action with a zero-tolerance policy.
- **Top Retrieved Sources:** `04_Code_of_Conduct.pdf;01_Employee_Handbook.pdf;10_Travel_and_Expense_Policy.pdf`
- **Source Pages:** `3;1;1`
- **Retrieval Similarity Scores:** `0.3504;0.3266;0.1782`

### [Q12] What are the password rules for company systems?
- **Generated Answer:** Based on the Zyro Dynamics HR policy, the password and access management standards for company systems are as follows:

* **Minimum password length:** 12 characters
* **Password complexity:** Must contain uppercase letters, lowercase letters, at least one number, and one special character
* **Password expiry:** Every 90 days
* **Multi-factor authentication (MFA):** Mandatory for all company systems without exception
* **Password reuse:** The last 12 passwords may not be reused
* **Sharing of credentials:** Strictly prohibited; attracts immediate disciplinary action
- **Top Retrieved Sources:** `07_IT_and_Data_Security_Policy.pdf;07_IT_and_Data_Security_Policy.pdf;07_IT_and_Data_Security_Policy.pdf`
- **Source Pages:** `2;2;2`
- **Retrieval Similarity Scores:** `0.5739;0.5511;0.4419`

### [Q13] How do I raise a POSH complaint and what's the timeline?
- **Generated Answer:** To raise a POSH complaint, the aggrieved person must submit a written complaint to the Internal Complaints Committee (ICC) by email or sealed letter. 

**Timeline:**
* The complaint must be filed within **3 months** of the incident or the last incident in a series of incidents. 
* In exceptional circumstances, the ICC may extend this period by a further **3 months** upon a written application with reasons.
- **Top Retrieved Sources:** `08_Prevention_of_Sexual_Harassment_Policy.pdf;08_Prevention_of_Sexual_Harassment_Policy.pdf;08_Prevention_of_Sexual_Harassment_Policy.pdf`
- **Source Pages:** `3;1;1`
- **Retrieval Similarity Scores:** `0.5060;0.4683;0.4518`

### [Q14] How much notice period do I need to serve to resign?
- **Generated Answer:** Based on the provided HR policy, the notice period required to resign depends on your grade:

* **L1 to L3:** 30 days
* **L4 to L6:** 60 days
* **L7 to L9:** 90 days
* **L10 (C-Suite):** 90 days
- **Top Retrieved Sources:** `09_Onboarding_and_Separation_Policy.pdf;09_Onboarding_and_Separation_Policy.pdf;02_Leave_Policy.pdf`
- **Source Pages:** `2;2;2`
- **Retrieval Similarity Scores:** `0.5028;0.4926;0.4887`

### [Q15] What's my hotel and daily allowance for international travel?
- **Generated Answer:** I am sorry, but the provided HR policy documents do not contain information regarding this topic.
- **Top Retrieved Sources:** `10_Travel_and_Expense_Policy.pdf;10_Travel_and_Expense_Policy.pdf;10_Travel_and_Expense_Policy.pdf`
- **Source Pages:** `2;3;2`
- **Retrieval Similarity Scores:** `0.5389;0.4720;0.4249`

---

## Q16–Q20: Out-of-Scope Results

### [Q16] How do I apply for a job opening at Zyro Dynamics?
- **Refusal Status:** ✅ Polite Refusal (Appropriate)
- **Generated Answer:** I am sorry, but the provided HR policy documents do not contain information regarding this topic.
- **Top Retrieved Sources:** `00_Company_Profile.pdf;10_Travel_and_Expense_Policy.pdf;04_Code_of_Conduct.pdf`
- **Source Pages:** `4;1;2`
- **Retrieval Similarity Scores:** `0.5723;0.5679;0.5595`

### [Q17] Can you tell me my ESOP vesting schedule?
- **Refusal Status:** ⚠️ Answered / Attempted without direct policy
- **Generated Answer:** ERROR: Generation stage failed: Gemini API call failed: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 15, model: gemini-3.5-flash-lite\nPlease retry in 12.13444861s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://ai.google.dev/gemini-api/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-3.5-flash-lite'}, 'quotaValue': '15'}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '12s'}]}}
- **Top Retrieved Sources:** ``
- **Source Pages:** ``
- **Retrieval Similarity Scores:** ``

### [Q18] What was Zyro Dynamics' revenue last year?
- **Refusal Status:** ✅ Polite Refusal (Appropriate)
- **Generated Answer:** I am sorry, but the provided HR policy documents do not contain information regarding this topic.
- **Top Retrieved Sources:** `00_Company_Profile.pdf;00_Company_Profile.pdf;00_Company_Profile.pdf`
- **Source Pages:** `2;1;1`
- **Retrieval Similarity Scores:** `0.6291;0.6077;0.5995`

### [Q19] How does ZyroCRM compare to Salesforce?
- **Refusal Status:** ✅ Polite Refusal (Appropriate)
- **Generated Answer:** I am sorry, but the provided HR policy documents do not contain information regarding this topic.
- **Top Retrieved Sources:** `00_Company_Profile.pdf;00_Company_Profile.pdf;00_Company_Profile.pdf`
- **Source Pages:** `2;1;1`
- **Retrieval Similarity Scores:** `0.5873;0.5613;0.5334`

### [Q20] What is Zoho's leave policy like?
- **Refusal Status:** ✅ Polite Refusal (Appropriate)
- **Generated Answer:** I am sorry, but the provided HR policy documents do not contain information regarding this topic.
- **Top Retrieved Sources:** `02_Leave_Policy.pdf;02_Leave_Policy.pdf;02_Leave_Policy.pdf`
- **Source Pages:** `1;1;2`
- **Retrieval Similarity Scores:** `0.6298;0.6116;0.5309`
