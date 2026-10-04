# Big-4 Technical Interview Defense Guide

This guide prepares engineers, quantitative modelers, and model risk specialists to defend CreditBridge V2 in rigorous technical interviews with Big-4 accounting/consulting firms (Deloitte, PwC, EY, KPMG), global investment banks, and prudential financial regulators (RBI, Federal Reserve, OCC).

Each question is structured with:
- **Crisp Spoken Answer**: A concise, polished 2–3 sentence executive response.
- **Deep Technical Explanation**: In-depth mathematical formulations, architectural decisions, code references, and operational trade-offs.

---

### Q1: Why did you choose an L2-regularized Logistic Regression instead of an advanced gradient boosted tree (e.g., XGBoost, LightGBM) or deep neural network?
**Crisp Spoken Answer:**  
In regulated credit underwriting, monotonic interpretability, auditability, and regulatory compliance (e.g., SR 11-7, FCRA adverse action notices) outweigh marginal non-linear gains. Logistic regression produces linear, log-odds monotonic risk surfaces that guarantee predictable sensitivities and exact, closed-form SHAP attributions without post-hoc approximation errors.

**Deep Technical Explanation:**  
- **Model Card Reference**: `docs/MODEL_CARD.md`, `src/train_model.py`.
- Complex non-linear models like XGBoost frequently capture non-monotonic interactions (e.g., income increasing default probability in localized leaf splits) that violate fundamental underwriting common-sense and fail regulatory model risk validation.
- L2 regularization ($\lambda = 1.0 / C$) controls multicollinearity among correlated transactional proxies (such as UPI inflow volume and estimated income).
- Closed-form SHAP calculations ($\phi_j = w_j \cdot (x_j - E[X_j])$) are exact, instantaneous ($O(d)$ compute), and mathematically aligned with the model's true decision boundaries, unlike KernelSHAP or TreeSHAP interventional approximations.

---

### Q2: How do you address the class imbalance between non-defaulters (~86%) and defaulters (~14%) during model training?
**Crisp Spoken Answer:**  
We train the logistic regression model using balanced class weighting (`class_weight='balanced'`), which scales the loss function inversely proportional to class frequencies. We then apply an exact Bayesian prior odds adjustment at inference to un-bias predictions back to the empirical market default rate of 14%.

**Deep Technical Explanation:**  
- **Code Reference**: `src/train_model.py`, `src/scoring_utils.py` (`probability_to_credit_score`).
- Training with `class_weight='balanced'` assigns sample weight $w_0 = \frac{N}{2 N_0}$ and $w_1 = \frac{N}{2 N_1}$, effectively optimizing the decision boundary as if the training prior were $50/50$. This maximizes minority-class recall and gradient updates.
- However, raw predicted probabilities $p_{\text{raw}}$ reflect the artificial 0.50 prior.
- To produce true calibrated default probabilities $p_{\text{cal}}$, we apply Bayes' theorem to rescale the predicted odds by the ratio of true population odds to training sample odds:
  $$\text{odds}_{\text{raw}} = \frac{p_{\text{raw}}}{1 - p_{\text{raw}}}, \quad \text{odds}_{\text{cal}} = \text{odds}_{\text{raw}} \times \left(\frac{\pi}{1 - \pi}\right) \times \left(\frac{1 - \pi_{\text{train}}}{\pi_{\text{train}}}\right)$$
  Since $\pi_{\text{train}} = 0.50$, the second fraction is 1.0, reducing to:
  $$\text{odds}_{\text{cal}} = \text{odds}_{\text{raw}} \times \frac{0.14}{0.86}, \quad p_{\text{cal}} = \frac{\text{odds}_{\text{cal}}}{1 + \text{odds}_{\text{cal}}}$$

---

### Q3: How is the calibrated probability of default converted into a standard 300–900 credit score?
**Crisp Spoken Answer:**  
We employ the classic industry-standard "Points to Double the Odds" (PDO) logarithmic transformation. The calibrated log-odds of repayment vs. default are scaled linearly using calibrated offset and factor parameters, and clamped to the range of [300, 900].

**Deep Technical Explanation:**  
- **Code Reference**: `src/scoring_utils.py`, `docs/MODEL_CARD.md`.
- Credit score formula:
  $$\text{Score} = \text{Offset} + \text{Factor} \times \ln\left(\frac{1 - p_{\text{cal}}}{p_{\text{cal}}}\right)$$
- The parameters are derived from two anchor points: a score of 680 at odds of 7:1 (12.5% default rate), and a doubling of odds every 66 points ($\text{PDO} = 66$):
  $$\text{Factor} = \frac{\text{PDO}}{\ln(2)} = \frac{66}{0.69315} \approx 95.2 \approx 95.0$$
  $$\text{Offset} = 680.0 - 95.0 \times \ln(7.0) \approx 680.0 - 184.86 = 495.14 \approx 490.0$$
- Implemented as:
  $$\text{Score} = \text{clip}\left(490.0 + 95.0 \times \ln\left(\frac{1 - p_{\text{cal}}}{p_{\text{cal}}}\right), 300, 900\right)$$
- This ensures an applicant with a 14% default probability scores approximately 663 (Moderate Risk tier), perfectly matching domestic bureau conventions (CIBIL/Experian).

---

### Q4: How do you prevent data leakage when engineering features and scaling inputs?
**Crisp Spoken Answer:**  
All preprocessing transformations—including median imputation statistics, standard scaler means, standard deviations, and one-hot categorical encodings—are encapsulated inside a scikit-learn `Pipeline` fitted strictly on the training partition. No test or real-world inference data ever influences transformation parameters.

**Deep Technical Explanation:**  
- **Code Reference**: `src/feature_engineering.py` (`FeaturePipeline`), `src/train_model.py`.
- In naive workflows, fit-transforming scalers or imputers over the entire dataset before train-test splitting leaks test distribution information into training features.
- In CreditBridge, `FeaturePipeline.fit(X_train)` fits all `SimpleImputer` and `StandardScaler` objects exclusively on $X_{\text{train}}$. The fitted pipeline object is saved within `models/credit_model.pkl`.
- During Real Data Mode inference, the pipeline only calls `.transform(X_real)`, applying the frozen training population statistics without refitting.

---

### Q5: Why are 9 of the 21 model features imputed from synthetic medians during Real Data Mode?
**Crisp Spoken Answer:**  
Standard bank statement CSV uploads provide granular cashflow and UPI signals but cannot observe external utility provider on-time payment histories, telecom lapse records, or gig platform telemetry. Rather than forcing applicants to self-report unverified behavioral estimates—which creates extreme adverse selection—we hold these 9 features at population median baselines.

**Deep Technical Explanation:**  
- **Documentation Reference**: `docs/FEATURE_LINEAGE.md`.
- Features such as `electricity_bill_ontime_rate` and `days_since_last_recharge_lapse` exist in the synthetic model to demonstrate full-ecosystem alternative underwriting.
- However, bank statement narrations rarely contain bill due dates or service disconnection dates.
- Holding unobserved features at training median constants ensures they contribute zero net variance to the scaled linear log-odds:
  $$z_j = w_j \times \left(\frac{\text{Median}_j - \mu_j}{\sigma_j}\right) \approx 0$$
- Every imputed feature is explicitly flagged in the audit manifest, and underwriters receive an **Evidence Coverage Ratio** (42.9%) indicating that additional third-party API verification (e.g., Account Aggregator) is needed for high-stakes lending decisions.

---

### Q6: How do you prevent lookahead bias in transaction feature extraction?
**Crisp Spoken Answer:**  
All transaction aggregations are strictly backward-looking from the latest transaction timestamp in the statement. We slice transactions relative to observation windows (e.g., active calendar months) and calculate rolling volatility without accessing future periods.

**Deep Technical Explanation:**  
- **Code Reference**: `src/real_data_features.py` (`extract_real_borrower_features`).
- The observation span is determined by:
  $$\Delta t = \max(t_{\text{tx}}) - \min(t_{\text{tx}})$$
- Monthly transaction frequency and volume averages divide total observed volume by active months:
  $$M = \max\left(\frac{\Delta t_{\text{days}}}{30.0}, 1.0\right)$$
- Volatility is computed from grouped calendar-month buckets ($M_1, M_2, \dots, M_k$). No retrospective smoothing or forward-looking interpolation is permitted.

---

### Q7: What is the purpose of the 90-day transaction history blocker rule?
**Crisp Spoken Answer:**  
Evaluating creditworthiness on short observation windows (< 3 months) is highly susceptible to temporary cashflow surges, seasonal gig demand, or strategic account seasoning. Enforcing a strict 90-day and 15-transaction threshold prevents hallucinated underwriting scores on statistically insufficient data.

**Deep Technical Explanation:**  
- **Code Reference**: `src/real_data_quality.py` (`check_history_sufficiency`), `src/real_data_contracts.py`.
- If $\Delta t_{\text{days}} < 90$ or $N_{\text{tx}} < 15$:
  1. The assessment quality is marked `SufficiencyTier.INSUFFICIENT`.
  2. `is_scoreable` is set to `False`.
  3. Execution halts before model scoring, and the response emits explicit `blocker_reasons`.
- In retail banking, 90 days represents the minimum observation cycle required to verify cash flow stationarity, detect month-end liquidity stress, and observe recurring living expenses (rent, mobile top-ups, family transfers).

---

### Q8: How does your system detect and defend against malicious file uploads?
**Crisp Spoken Answer:**  
We implement a multi-layered defensive boundary: a strict 10 MB payload ceiling, a 50,000-row processing ceiling, file extension whitelisting, directory traversal scrubbing, and binary magic-byte inspection that intercepts executables, zip archives, and disguised PDFs before parsing begins.

**Deep Technical Explanation:**  
- **Code Reference**: `src/privacy_security.py` (`validate_upload_security`).
- Attackers frequently rename executable binaries or zip bombs to `.csv` to exploit backend parsers.
- CreditBridge inspects the initial byte signatures:
  - `b"MZ"` (Windows PE/DLL)
  - `b"\x7fELF"` (Linux binary)
  - `b"PK\x03\x04"` (Zip archives / macro-enabled Office files)
  - `b"%PDF"` (PDF documents)
- Path strings are sanitized against null bytes (`\x00`) and relative traversal tokens (`../`, `..\\`).
- All validation happens in-memory before passing the byte buffer to `csv.reader` or pandas.

---

### Q9: How do you enforce zero disk persistence in a multi-tenant cloud environment?
**Crisp Spoken Answer:**  
The entire parsing, classification, feature mapping, and scoring pipeline executes strictly in-memory using `io.BytesIO` and `io.StringIO` streams. Context managers dereference and wipe data buffers immediately post-assessment, preventing residual data remanence on physical container volumes.

**Deep Technical Explanation:**  
- **Code Reference**: `src/privacy_security.py` (`EphemeralProcessingContext`), `tests/test_privacy_security.py`.
- Writing temporary files to `/tmp` creates severe data breach exposure under India's DPDP Act 2023.
- In CreditBridge, raw bytes received from the client are loaded into an ephemeral memory buffer.
- When execution exits the `EphemeralProcessingContext`:
  1. Internal DataFrame references are set to `None`.
  2. Python's `gc.collect()` is triggered.
  3. Unit tests monitor the filesystem to confirm that zero temporary files are created during end-to-end evaluation.

---

### Q10: How does the transaction classifier disambiguate P2P transfers from Merchant UPI spend?
**Crisp Spoken Answer:**  
We analyze UPI Virtual Payment Address (VPA) handles and transaction narration tokens. Counterparties matching merchant aggregator handles (e.g., `@paytm`, `@ybl`, `@okaxis` with merchant names, or QR scan prefixes) are categorized as merchant consumption, whereas personal mobile number handles or individual names are tagged as P2P.

**Deep Technical Explanation:**  
- **Code Reference**: `src/transaction_classifier.py`.
- In Indian digital payments, personal VPAs typically follow mobile number patterns (`9876543210@upi`, `name@okhdfcbank`), while merchant VPAs include commercial aggregator descriptors (`swiggy@icici`, `bharatpe.9021@yesbank`, `pos.merchant@axis`).
- Our rule engine inspects both counterparty handle strings and narration keywords (`UPI-P2M`, `UPI-P2P`, `QR`, `MERCHANT`).
- When classification confidence is low, the transaction is tagged as `NormalizedCategory.UNKNOWN` with a confidence score of 0.30, avoiding biased financial ratio distortions.

---

### Q11: How do you detect and report distribution shift when scoring real statements against a synthetic model?
**Crisp Spoken Answer:**  
We compare every derived feature value against the 5th and 95th percentiles of the training distribution stored in `data/reference_distributions.json`. Outliers exceeding these bounds are flagged in the assessment report with clear warnings to underwriters.

**Deep Technical Explanation:**  
- **Code Reference**: `src/real_data_quality.py` (`evaluate_distribution_shift`), `data/reference_distributions.json`.
- When an applicant's monthly income or UPI volatility falls outside the 1st–99th percentile of the training universe, linear models can extrapolate into uncalibrated risk regimes.
- CreditBridge tags each feature with a `DistributionStatus`:
  - `within_range`: Between 5th and 95th percentile.
  - `near_boundary`: Between 1st–5th or 95th–99th percentile.
  - `outside_observed_range`: Below 1st or above 99th percentile.
- Outliers do not crash the pipeline but generate structured warnings informing the underwriter that the score reflects out-of-distribution extrapolation.

---

### Q12: How are local SHAP values calculated for your logistic regression model?
**Crisp Spoken Answer:**  
Because logistic regression is an additive linear model in the log-odds space, SHAP attributions can be computed exactly and analytically without sampling approximations. Each feature attribution is simply the product of the standardized model coefficient and the difference between the applicant's feature value and the training baseline expectation.

**Deep Technical Explanation:**  
- **Code Reference**: `src/explain.py` (`explain_borrower_record`).
- In log-odds space:
  $$\ln\left(\frac{p}{1 - p}\right) = \beta_0 + \sum_{j=1}^d \beta_j \tilde{x}_j$$
- By definition, the Shapley value $\phi_j$ of feature $j$ represents its contribution relative to the expected log-odds $E[f(X)]$:
  $$\phi_j = \beta_j \cdot (\tilde{x}_j - E[\tilde{X}_j])$$
  Since standardized features have $E[\tilde{X}_j] = 0$:
  $$\phi_j = \beta_j \cdot \tilde{x}_j$$
- This satisfies efficiency ($\sum \phi_j = f(x) - E[f(X)]$), symmetry, and additivity exactly. No Monte Carlo sampling or surrogate models are needed.

---

### Q13: Why is SHAP explainability non-causal, and why must underwriters be cautioned about this?
**Crisp Spoken Answer:**  
SHAP decomposes statistical correlation within the model's fitted equation, not real-world causal mechanisms. For example, if a borrower artificially inflates their recharge frequency by buying multiple small top-ups, their SHAP score will improve even though their true default risk has not changed.

**Deep Technical Explanation:**  
- **Documentation Reference**: `docs/MODEL_CARD.md`.
- SHAP values answer: *"How much did the model's mathematical function shift based on this input?"* They do NOT answer: *"What happens if the borrower intervenes to change this behavior?"*
- Transactional proxies are subject to Goodhart’s Law: *"When a measure becomes a target, it ceases to be a good measure."*
- Adverse action notices must distinguish between descriptive statistical explanations and actionable causal guidance.

---

### Q14: How does the system handle circular transaction fraud or UPI round-tripping?
**Crisp Spoken Answer:**  
Currently, our rule-based transaction classifier flags internal self-transfers between accounts of the same holder and filters out reversals. However, detecting sophisticated multi-party circular UPI loops (round-tripping) requires full graph network analysis, which is flagged as an explicit limitation in our production roadmap.

**Deep Technical Explanation:**  
- **Code Reference**: `src/transaction_classifier.py`, `docs/HOSTILE_REVIEW.md`.
- Single-statement heuristic parsing can detect simple self-transfers (matching account holder names) and exclude them from income calculations.
- However, two colluding gig workers repeatedly sending ₹10,000 back and forth to inflate UPI inflow volumes cannot be identified from a single statement in isolation.
- Addressing this requires a centralized Account Aggregator (AA) pipeline or multi-account graph network analysis (see `docs/PRODUCTION_ROADMAP.md`).

---

### Q15: What is the purpose of the SHA-256 cryptographic audit manifest?
**Crisp Spoken Answer:**  
The audit manifest provides complete regulatory non-repudiation and traceability without storing sensitive customer financial records. It cryptographically binds the raw statement hash to the feature provenance records, data quality warnings, and final credit score.

**Deep Technical Explanation:**  
- **Code Reference**: `src/privacy_security.py` (`create_audit_manifest`).
- Under prudential model audit standards, lenders must prove that an underwriting decision was reached deterministically from verified source data.
- Storing full bank statements violates data minimization principles.
- CreditBridge computes $\text{SHA-256}(\text{raw\_statement\_bytes})$ at ingestion. The resulting 64-character hex digest is logged in the `AuditManifest` alongside the timestamp, model version, and score.
- If a borrower or regulator later contests an adverse decision, the original statement can be re-hashed to prove that the exact file was processed by the exact model version without post-hoc tampering.

---

### Q16: How do you evaluate algorithmic fairness across demographic groups?
**Crisp Spoken Answer:**  
We compute demographic parity ratio, disparate impact ratio, and equalized odds differences across age cohorts, occupations, and city tiers. Protected attributes like gender, caste, and religion are strictly excluded from the model feature contract.

**Deep Technical Explanation:**  
- **Code Reference**: `src/fairness_diagnostics.py`, `docs/MODEL_CARD.md`.
- Disparate Impact Ratio (DIR) compares the acceptance rate of an unprivileged group to a privileged group:
  $$\text{DIR} = \frac{P(\hat{Y}=0 \mid D = \text{unprivileged})}{P(\hat{Y}=0 \mid D = \text{privileged})}$$
- Under standard US EEOC and global fair-lending thresholds (the four-fifths rule), a DIR $< 0.80$ signals potential adverse impact.
- Our fairness suite also evaluates Equalized Odds by checking False Positive Rate (FPR) parity across groups to ensure the model does not disproportionately deny creditworthy individuals in Tier-3 cities.

---

### Q17: What are the primary failure modes of your multi-bank CSV parser?
**Crisp Spoken Answer:**  
Primary failure modes include ambiguous date formats (`05/06/2025`), merged narration columns, negative signed values vs. debit/credit columns, and multi-line narrations. We mitigate these with regex header scoring, date heuristic resolvers, and deterministic amount cleaning.

**Deep Technical Explanation:**  
- **Code Reference**: `src/real_data_parser.py`.
- Indian banks lack a unified CSV standard (e.g., SBI uses separate `Txn Date` and `Value Date`; HDFC uses `Narration` and `Withdrawal Amt.`).
- If an ambiguous date like `04/05/2025` appears, the parser inspects the entire column: if subsequent rows contain `15/05/2025`, the format is deterministically resolved as `DD/MM/YYYY`.
- Corrupted or unparseable rows are isolated rather than causing a pipeline crash, and row-level parse success rates are tracked in the data quality report.

---

### Q18: Why can't you claim that legacy AUC of 0.755 or current baseline AUC of 0.624 represent real-world credit performance?
**Crisp Spoken Answer:**  
Those performance metrics were evaluated entirely on synthetic test sets generated from mathematical assumptions. Earlier prototype documentation cited a historical run of 0.755 AUC / 38.8% KS, whereas the current reproducible baseline artifact (`models/credit_model.pkl`) achieves 0.6240 ROC-AUC and 21.08% KS on `synthetic_borrowers.csv`. Real-world credit risk includes unmodeled macroeconomic shocks, strategic defaults, and fraud that synthetic distributions cannot replicate. Claiming real-world readiness without empirical repayment validation would be a serious model governance violation.

**Deep Technical Explanation:**  
- **Documentation Reference**: `docs/MODEL_CARD.md`, `README.md`, `BASELINE_REPORT.md`.
- Synthetic data generator `data/generate_synthetic_data.py` created ground truth default labels based on a logistic link function of engineered proxies.
- A model trained on that data reflects the generator's specific assumptions. The historical run achieved $\text{AUC} = 0.755, \text{KS} = 38.8\%$, while the current verified, reproducible baseline produces $\text{AUC} = 0.6240, \text{KS} = 21.08\%$.
- In real-world micro-lending, true label assignment is subject to behavioral unpredictability, health emergencies, and lender collection efficacy. The current model serves as a validated architectural prototype, not a commercially validated scorecard.

---

### Q19: How do you identify salary or earnings inflows for gig workers with irregular payout cycles?
**Crisp Spoken Answer:**  
We combine regex keyword matching for gig aggregators (e.g., Zepto, Blinkit, Zomato, Uber) with frequency and amount clustering. Rather than expecting a monthly fixed salary, we recognize weekly payouts and aggregate them over calendar months to compute income stability.

**Deep Technical Explanation:**  
- **Code Reference**: `src/transaction_classifier.py`, `src/real_data_features.py`.
- Traditional salaried workers receive a single monthly NEFT credit. Gig delivery partners frequently receive weekly payouts on Tuesdays or daily cashouts.
- Our classifier matches employer tokens (`ZOMATO HYPERPURE`, `ZEPTO DELIVERY`, `BLINKIT LOGISTICS`) and tags them as `NormalizedCategory.GIG_INCOME_LIKE`.
- Inflow aggregation sums all valid gig earnings across the observation period and divides by the active month count, preventing gig workers from being penalized by legacy "one salary deposit per month" underwriting heuristics.

---

### Q20: What are the mandatory milestones required before this backend can be deployed in a regulated lending environment?
**Crisp Spoken Answer:**  
Production deployment requires four fundamental pillars: replacing synthetic training data with real loan repayment outcomes, integrating with the RBI Account Aggregator framework for tamper-proof data retrieval, implementing an automated drift monitoring pipeline, and securing independent Model Risk Management (MRM) validation.

**Deep Technical Explanation:**  
- **Documentation Reference**: `docs/PRODUCTION_ROADMAP.md`.
- Specifically:
  1. **Empirical Calibration**: Train on $\ge 50,000$ real micro-loan repayment records with confirmed 90+ DPD default labels.
  2. **Account Aggregator (FIP/FIU)**: Discontinue unverified CSV uploads in favor of cryptographically signed financial data via RBI-regulated Account Aggregators.
  3. **Continuous Monitoring**: Deploy Population Stability Index (PSI) and Characteristic Analysis monitoring with automated alerts for PSI $> 0.25$.
  4. **Regulatory Governance**: Complete full SR 11-7 model documentation, independent review by accredited model risk auditors, and compliance certification under RBI Digital Lending Guidelines.
