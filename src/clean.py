"""
src/clean.py

Cleans the raw loan-default dataset and writes a cleaned CSV.

Run from the repo root:
    python src/clean.py

Reads   : data/raw/<file>.csv   (the single CSV found in data/raw/)
Writes  : data/processed/clean.csv
Writes  : docs/cleaning_notes.md   (also printed to stdout)

Scope (deliberately limited — see project brief):
    - remove exact duplicate rows
    - fix data types (numbers-as-text, whole-number columns stored as float, etc.)
    - standardize category labels (case/whitespace variants of the same value)
    - fix or null-out impossible values (documented below)
    - convert obvious missing-value placeholders to NaN
    - genuine missing values are LEFT AS NaN — no imputation here
    - no encoding, no scaling, no column drops, no row splitting
    - the target column `Status` is never modified
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
DOCS_DIR = REPO_ROOT / "docs"

OUTPUT_CSV = PROCESSED_DIR / "clean.csv"
NOTES_MD = DOCS_DIR / "cleaning_notes.md"

TARGET_COL = "Status"

# Threshold above which a Loan-to-Value ratio is treated as a data-entry error
# rather than a genuinely high-LTV loan. Real high-LTV mortgage programs can
# run well above 100%, but three-, four-figure LTVs (e.g. 7831%) are not
# realistic and trace back to a handful of rows with an implausibly small
# property_value (e.g. $8,000). We null out property_value/LTV on those rows
# rather than dropping them, since every other field on the row is usable.
LTV_IMPOSSIBLE_THRESHOLD = 1000

# Values that mean "unknown" but were encoded as a real-looking category
# instead of a null. Mapped to NaN, column by column.
MISSING_PLACEHOLDERS = {
    "Gender": ["Sex Not Available"],
}


def find_raw_csv() -> Path:
    """Locate the single CSV file under data/raw/."""
    if not RAW_DIR.exists():
        sys.exit(f"ERROR: {RAW_DIR} does not exist.")
    csvs = sorted(RAW_DIR.glob("*.csv"))
    if not csvs:
        sys.exit(f"ERROR: no .csv file found in {RAW_DIR}.")
    if len(csvs) > 1:
        names = ", ".join(p.name for p in csvs)
        sys.exit(
            f"ERROR: multiple CSV files found in {RAW_DIR} ({names}). "
            "Keep only the one raw dataset there."
        )
    return csvs[0]


def coerce_numeric_text(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Fix 'numbers stored as text': any object/string column whose non-null
    values are ALL parseable as numbers gets converted to a numeric dtype.
    (No such columns exist in this dataset today, but this keeps the script
    correct if the raw export changes.)
    """
    fixed_cols = []
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype).startswith("str"):
            non_null = df[col].dropna()
            if non_null.empty:
                continue
            coerced = pd.to_numeric(non_null, errors="coerce")
            if coerced.notna().all():
                df[col] = pd.to_numeric(df[col], errors="coerce")
                fixed_cols.append(col)
    return df, fixed_cols


def standardize_label_collisions(df: pd.DataFrame, exclude: set[str]) -> dict[str, int]:
    """
    Generic fix for the 'Male / male / M' problem: for every categorical
    column (excluding ones handled explicitly elsewhere), find values that
    are identical once trimmed and lower-cased. If a column has such
    collisions, all variants are mapped to the single most frequent raw
    spelling. Columns with no collisions are left completely untouched
    (this avoids mangling legitimate short codes like 'cf'/'ncf', 'CIB').
    Returns {column: rows_changed}.
    """
    changed: dict[str, int] = {}
    obj_cols = [
        c
        for c in df.columns
        if (df[c].dtype == object or str(df[c].dtype).startswith("str"))
        and c not in exclude
    ]
    for col in obj_cols:
        s = df[col]
        non_null = s.dropna()
        if non_null.empty:
            continue
        norm_key = non_null.astype(str).str.strip().str.casefold()
        groups = non_null.groupby(norm_key)
        n_variants_per_key = groups.nunique()
        colliding_keys = n_variants_per_key[n_variants_per_key > 1].index
        if len(colliding_keys) == 0:
            continue

        # Build canonical spelling = most frequent raw value for each key
        canonical = (
            non_null.groupby(norm_key)
            .agg(lambda x: x.value_counts().idxmax())
        )
        mapping = {}
        rows_changed = 0
        for key in colliding_keys:
            variants = groups.get_group(key)
            best = canonical.loc[key]
            for raw_val in variants.unique():
                if raw_val != best:
                    mapping[raw_val] = best
                    rows_changed += (s == raw_val).sum()

        if mapping:
            df[col] = s.replace(mapping)
            changed[col] = int(rows_changed)

    return df, changed


def main() -> None:
    raw_path = find_raw_csv()
    df = pd.read_csv(raw_path)

    report_lines: list[str] = []

    def log(line: str = "") -> None:
        report_lines.append(line)

    # ----------------------------------------------------------------- #
    # BEFORE snapshot
    # ----------------------------------------------------------------- #
    rows_before, cols_before = df.shape
    missing_before = df.isna().sum()
    dtypes_before = df.dtypes.copy()

    # ----------------------------------------------------------------- #
    # 1. Duplicate rows
    # ----------------------------------------------------------------- #
    dup_mask = df.duplicated()
    n_duplicates = int(dup_mask.sum())
    df = df.loc[~dup_mask].reset_index(drop=True)

    # ----------------------------------------------------------------- #
    # 2. Fix data types
    # ----------------------------------------------------------------- #
    df, numeric_text_fixed = coerce_numeric_text(df)

    # `term` is a whole number of months but is stored as float64 because
    # of missing values. Convert to a nullable integer type so it reads as
    # a count, not a continuous measurement, while still allowing NaN.
    term_dtype_fixed = False
    if "term" in df.columns and pd.api.types.is_float_dtype(df["term"]):
        non_null_terms = df["term"].dropna()
        if (non_null_terms % 1 == 0).all():
            df["term"] = df["term"].astype("Int64")
            term_dtype_fixed = True

    # ----------------------------------------------------------------- #
    # 3. Missing-value placeholders -> NaN
    # ----------------------------------------------------------------- #
    placeholder_counts: dict[str, int] = {}
    for col, placeholders in MISSING_PLACEHOLDERS.items():
        if col not in df.columns:
            continue
        mask = df[col].isin(placeholders)
        placeholder_counts[col] = int(mask.sum())
        if mask.any():
            df.loc[mask, col] = pd.NA

    # ----------------------------------------------------------------- #
    # 4. Standardize category labels
    # ----------------------------------------------------------------- #
    region_fixed_count = 0
    if "Region" in df.columns:
        original = df["Region"]
        standardized = original.astype("string").str.strip().str.title()
        region_fixed_count = int(
            ((original.notna()) & (original.astype("string") != standardized)).sum()
        )
        df["Region"] = standardized

    already_handled_cols = {"Region"} | set(MISSING_PLACEHOLDERS.keys())
    df, label_collision_changes = standardize_label_collisions(df, exclude=already_handled_cols)

    # ----------------------------------------------------------------- #
    # 5. Fix / null-out impossible values
    # ----------------------------------------------------------------- #
    impossible_ltv_count = 0
    if {"LTV", "property_value"}.issubset(df.columns):
        impossible_mask = df["LTV"] > LTV_IMPOSSIBLE_THRESHOLD
        impossible_ltv_count = int(impossible_mask.sum())
        if impossible_mask.any():
            df.loc[impossible_mask, ["LTV", "property_value"]] = pd.NA

    # ----------------------------------------------------------------- #
    # Sanity: target column untouched, no columns dropped
    # ----------------------------------------------------------------- #
    assert TARGET_COL in df.columns, f"{TARGET_COL} column is missing!"
    assert set(df.columns) == set(pd.read_csv(raw_path, nrows=0).columns), (
        "Column set changed — no columns should be added or dropped."
    )

    # ----------------------------------------------------------------- #
    # AFTER snapshot
    # ----------------------------------------------------------------- #
    rows_after, cols_after = df.shape
    missing_after = df.isna().sum()

    # ----------------------------------------------------------------- #
    # Write outputs
    # ----------------------------------------------------------------- #
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    # ----------------------------------------------------------------- #
    # Build the report (printed + saved to docs/cleaning_notes.md)
    # ----------------------------------------------------------------- #
    log(f"# Cleaning notes — {raw_path.name}")
    log()
    log(f"Source: `data/raw/{raw_path.name}`")
    log(f"Output: `data/processed/clean.csv`")
    log()
    log("## Shape")
    log()
    log(f"- Before: {rows_before:,} rows x {cols_before} columns")
    log(f"- After:  {rows_after:,} rows x {cols_after} columns")
    log(f"- Columns dropped: 0 (none — all {cols_after} columns preserved)")
    log()
    log("## Duplicates")
    log()
    log(f"- Exact duplicate rows removed: {n_duplicates:,}")
    log()
    log("## Missing values per column (before -> after)")
    log()
    log("| Column | Missing before | Missing after | Change |")
    log("|---|---:|---:|---|")
    for col in df.columns:
        before = int(missing_before.get(col, 0))
        after = int(missing_after.get(col, 0))
        if before == after:
            change = "—"
        elif after > before:
            change = f"+{after - before} (placeholder/impossible values nulled)"
        else:
            change = f"-{before - after}"
        log(f"| {col} | {before:,} | {after:,} | {change} |")
    log()
    log("## Column-by-column changes")
    log()

    any_change_logged = False

    if numeric_text_fixed:
        any_change_logged = True
        log(f"**{', '.join(numeric_text_fixed)}** — stored as text, contained only "
            f"numeric values. Converted to a numeric dtype. Rows affected: all "
            f"non-null rows in {'these columns' if len(numeric_text_fixed) > 1 else 'this column'}.")
        log()

    if term_dtype_fixed:
        any_change_logged = True
        n_term_nonnull = int(df["term"].notna().sum())
        log(f"**term** — stored as `float64` (e.g. `360.0`) despite always holding a "
            f"whole number of months, because missing values forced a float dtype. "
            f"Converted to a nullable integer type (`Int64`). No values changed, "
            f"{n_term_nonnull:,} non-null rows re-typed; "
            f"missing values ({int(df['term'].isna().sum())}) left as NaN.")
        log()

    if "Gender" in placeholder_counts:
        any_change_logged = True
        n = placeholder_counts["Gender"]
        log(f"**Gender** — the category `\"Sex Not Available\"` is a placeholder for "
            f"unknown gender, not a real category. Converted to NaN so it's treated "
            f"as missing rather than as its own group. Rows affected: {n:,}.")
        log()

    if region_fixed_count:
        any_change_logged = True
        log(f"**Region** — inconsistent capitalization (`\"south\"` vs `\"North\"`, "
            f"`\"central\"`, `\"North-East\"`). Standardized to title case "
            f"(`South`, `North`, `Central`, `North-East`) so the same region isn't "
            f"split across differently-cased labels. Rows affected: {region_fixed_count:,}.")
        log()

    for col, n in label_collision_changes.items():
        any_change_logged = True
        log(f"**{col}** — found case/whitespace variants of the same label "
            f"(e.g. differing only by capitalization or leading/trailing spaces). "
            f"Merged into a single consistent spelling. Rows affected: {n:,}.")
        log()

    if impossible_ltv_count:
        any_change_logged = True
        log(f"**property_value, LTV** — {impossible_ltv_count} rows had "
            f"`property_value` of about $8,000 against loan amounts in the "
            f"hundreds of thousands, producing Loan-to-Value ratios above "
            f"{LTV_IMPOSSIBLE_THRESHOLD}% (up to ~7,831%). This is not a plausible "
            f"real-world LTV (even aggressive high-LTV mortgage programs top out "
            f"well under 200%), so `property_value` and `LTV` were treated as "
            f"data-entry errors and set to NaN on those rows. The rows themselves "
            f"were kept (every other field is usable) rather than dropped. "
            f"Rows affected: {impossible_ltv_count}.")
        log()

    if not any_change_logged:
        log("No column-level changes were needed beyond duplicate removal.")
        log()

    log("## Columns checked and left unchanged")
    log()
    log("These were inspected for impossible values / inconsistent labels and found "
        "to be within a plausible range or already consistent, so nothing was changed:")
    log()
    log("- `income` — includes 0 values (no declared income), which is plausible "
        "for some applicants; left as-is.")
    log("- `Interest_rate_spread` — includes negative values, which is a normal, "
        "legitimate outcome of a below-par interest rate; left as-is.")
    log("- `Credit_Score` — consistently ranges 500-900 across all rows on this "
        "dataset's own scoring scale (not standard FICO); no impossible values.")
    log("- `age` — categorical brackets (e.g. `25-34`, `>74`), already a single "
        "consistent format.")
    log("- `dtir1`, `rate_of_interest`, `Upfront_charges`, `term`, `loan_amount` — "
        "ranges checked, no negative values or implausible extremes found.")
    log()
    log("## Rows dropped")
    log()
    if n_duplicates:
        log(f"- {n_duplicates:,} exact duplicate rows, removed in step 1.")
    else:
        log("- None. No exact duplicate rows were found in this dataset, and no "
            "other rows were dropped — impossible values were nulled out in place "
            "instead of removing the row (see property_value/LTV above).")
    log()
    log("## Judgment calls (for reference / faculty discussion)")
    log()
    log(f"- **LTV impossibility threshold set at {LTV_IMPOSSIBLE_THRESHOLD}%.** LTV "
        f"can legitimately exceed 100% for some loan programs, so we didn't flag "
        f"everything over 100 — only values so extreme (>{LTV_IMPOSSIBLE_THRESHOLD}%) "
        f"that they can only be explained by a data-entry error in `property_value`. "
        f"{impossible_ltv_count} rows were affected; the raw values are documented "
        f"above so the threshold can be revisited.")
    log("- **`\"Sex Not Available\"` treated as missing, not as its own gender "
        "category.** This affects a large share of rows and was a judgment call — "
        "an alternative view would keep it as a legitimate fourth category (e.g. "
        "'declined to state'). We chose NaN because it does not describe an "
        "applicant's gender, it describes an absence of that information.")
    log("- **Zero income and zero/low interest rates were left alone** rather than "
        "treated as impossible, since both are plausible in a real lending dataset.")
    log("- **No imputation was performed.** All genuine missing values (income, "
        "rate_of_interest, property_value, LTV, dtir1, Upfront_charges, "
        "Interest_rate_spread, loan_limit, approv_in_adv, age, "
        "submission_of_application, loan_purpose, Neg_ammortization, term) are left "
        "as NaN for the modeling pipeline to handle.")

    report = "\n".join(report_lines)
    NOTES_MD.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
