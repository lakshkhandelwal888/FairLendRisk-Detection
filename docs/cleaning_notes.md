# Cleaning notes — Loan_Default.csv

Source: `data/raw/Loan_Default.csv`
Output: `data/processed/clean.csv`

## Shape

- Before: 148,670 rows x 34 columns
- After:  148,670 rows x 34 columns
- Columns dropped: 0 (none — all 34 columns preserved)

## Duplicates

- Exact duplicate rows removed: 0

## Missing values per column (before -> after)

| Column | Missing before | Missing after | Change |
|---|---:|---:|---|
| ID | 0 | 0 | — |
| year | 0 | 0 | — |
| loan_limit | 3,344 | 3,344 | — |
| Gender | 0 | 37,659 | +37659 (placeholder/impossible values nulled) |
| approv_in_adv | 908 | 908 | — |
| loan_type | 0 | 0 | — |
| loan_purpose | 134 | 134 | — |
| Credit_Worthiness | 0 | 0 | — |
| open_credit | 0 | 0 | — |
| business_or_commercial | 0 | 0 | — |
| loan_amount | 0 | 0 | — |
| rate_of_interest | 36,439 | 36,439 | — |
| Interest_rate_spread | 36,639 | 36,639 | — |
| Upfront_charges | 39,642 | 39,642 | — |
| term | 41 | 41 | — |
| Neg_ammortization | 121 | 121 | — |
| interest_only | 0 | 0 | — |
| lump_sum_payment | 0 | 0 | — |
| property_value | 15,098 | 15,104 | +6 (placeholder/impossible values nulled) |
| construction_type | 0 | 0 | — |
| occupancy_type | 0 | 0 | — |
| Secured_by | 0 | 0 | — |
| total_units | 0 | 0 | — |
| income | 9,150 | 9,150 | — |
| credit_type | 0 | 0 | — |
| Credit_Score | 0 | 0 | — |
| co-applicant_credit_type | 0 | 0 | — |
| age | 200 | 200 | — |
| submission_of_application | 200 | 200 | — |
| LTV | 15,098 | 15,104 | +6 (placeholder/impossible values nulled) |
| Region | 0 | 0 | — |
| Security_Type | 0 | 0 | — |
| Status | 0 | 0 | — |
| dtir1 | 24,121 | 24,121 | — |

## Column-by-column changes

**term** — stored as `float64` (e.g. `360.0`) despite always holding a whole number of months, because missing values forced a float dtype. Converted to a nullable integer type (`Int64`). No values changed, 148,629 non-null rows re-typed; missing values (41) left as NaN.

**Gender** — the category `"Sex Not Available"` is a placeholder for unknown gender, not a real category. Converted to NaN so it's treated as missing rather than as its own group. Rows affected: 37,659.

**Region** — inconsistent capitalization (`"south"` vs `"North"`, `"central"`, `"North-East"`). Standardized to title case (`South`, `North`, `Central`, `North-East`) so the same region isn't split across differently-cased labels. Rows affected: 72,713.

**property_value, LTV** — 6 rows had `property_value` of about $8,000 against loan amounts in the hundreds of thousands, producing Loan-to-Value ratios above 1000% (up to ~7,831%). This is not a plausible real-world LTV (even aggressive high-LTV mortgage programs top out well under 200%), so `property_value` and `LTV` were treated as data-entry errors and set to NaN on those rows. The rows themselves were kept (every other field is usable) rather than dropped. Rows affected: 6.

## Columns checked and left unchanged

These were inspected for impossible values / inconsistent labels and found to be within a plausible range or already consistent, so nothing was changed:

- `income` — includes 0 values (no declared income), which is plausible for some applicants; left as-is.
- `Interest_rate_spread` — includes negative values, which is a normal, legitimate outcome of a below-par interest rate; left as-is.
- `Credit_Score` — consistently ranges 500-900 across all rows on this dataset's own scoring scale (not standard FICO); no impossible values.
- `age` — categorical brackets (e.g. `25-34`, `>74`), already a single consistent format.
- `dtir1`, `rate_of_interest`, `Upfront_charges`, `term`, `loan_amount` — ranges checked, no negative values or implausible extremes found.

## Rows dropped

- None. No exact duplicate rows were found in this dataset, and no other rows were dropped — impossible values were nulled out in place instead of removing the row (see property_value/LTV above).

## Judgment calls (for reference / faculty discussion)

- **LTV impossibility threshold set at 1000%.** LTV can legitimately exceed 100% for some loan programs, so we didn't flag everything over 100 — only values so extreme (>1000%) that they can only be explained by a data-entry error in `property_value`. 6 rows were affected; the raw values are documented above so the threshold can be revisited.
- **`"Sex Not Available"` treated as missing, not as its own gender category.** This affects a large share of rows and was a judgment call — an alternative view would keep it as a legitimate fourth category (e.g. 'declined to state'). We chose NaN because it does not describe an applicant's gender, it describes an absence of that information.
- **Zero income and zero/low interest rates were left alone** rather than treated as impossible, since both are plausible in a real lending dataset.
- **No imputation was performed.** All genuine missing values (income, rate_of_interest, property_value, LTV, dtir1, Upfront_charges, Interest_rate_spread, loan_limit, approv_in_adv, age, submission_of_application, loan_purpose, Neg_ammortization, term) are left as NaN for the modeling pipeline to handle.
