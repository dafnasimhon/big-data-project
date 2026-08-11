"""Shared feature/target column names (PLAN.md §2, §10, §11).

Single source of truth so exploration, cleaning, and the feature pipeline can't drift
apart on which columns matter.
"""

CANDIDATE_INPUT_FEATURES = [
    "Country",
    "Age",
    "EdLevel",
    "Employment",
    "RemoteWork",
    "YearsCodePro",
    "DevType",
    "OrgSize",
    "Industry",
    "LanguageHaveWorkedWith",
    "DatabaseHaveWorkedWith",
    "PlatformHaveWorkedWith",
]

TARGET_COLUMN = "ConvertedCompYearly"

SINGLE_VALUE_CATEGORICAL_COLUMNS = [
    "Country",
    "Age",
    "EdLevel",
    "RemoteWork",
    "DevType",
    "OrgSize",
    "Industry",
]

# `;`-separated multi-select fields, handled via RegexTokenizer + CountVectorizer in
# feature_pipeline.py rather than StringIndexer + OneHotEncoder. `Employment` moved here
# (2026-08-11) after a real OutOfMemoryError training RandomForestRegressor on the VM:
# it was originally in SINGLE_VALUE_CATEGORICAL_COLUMNS, but it's actually a multi-select
# field like "Employed, full-time;Student, part-time" — treating each unique COMBINATION
# as its own opaque one-hot category inflated it to ~107 columns (vs. the ~9 real
# underlying statuses), which is both wasteful (most of that "cardinality" is redundant
# combinations of the same handful of statuses) and a meaningfully large chunk of the
# feature-vector width that was pushing RandomForestRegressor's split-finding out of
# memory. Tokenizing it like the skill columns fixes both: far fewer dimensions, and each
# one now means something (e.g. "is employed full-time" as its own boolean signal)
# instead of an arbitrary combination id.
MULTI_VALUE_COLUMNS = [
    "Employment",
    "LanguageHaveWorkedWith",
    "DatabaseHaveWorkedWith",
    "PlatformHaveWorkedWith",
]
