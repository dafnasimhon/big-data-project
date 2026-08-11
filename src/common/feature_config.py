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
    "Employment",
    "RemoteWork",
    "DevType",
    "OrgSize",
    "Industry",
]

SKILL_COLUMNS = [
    "LanguageHaveWorkedWith",
    "DatabaseHaveWorkedWith",
    "PlatformHaveWorkedWith",
]
