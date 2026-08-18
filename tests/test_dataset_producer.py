from src.producers.dataset_producer import build_event


def _row(**overrides) -> dict:
    base = {
        "Country": "Israel",
        "Age": "25-34 years old",
        "EdLevel": "Bachelor's degree",
        "Employment": "Employed, full-time",
        "RemoteWork": "Hybrid",
        "YearsCodePro": "5",
        "DevType": "Developer, back-end",
        "OrgSize": "100 to 499 employees",
        "Industry": "Information Services, IT, Software Development",
        "LanguageHaveWorkedWith": "Python;SQL",
        "DatabaseHaveWorkedWith": "PostgreSQL",
        "PlatformHaveWorkedWith": "AWS",
        "ConvertedCompYearly": "125000",
    }
    base.update(overrides)
    return base


def test_build_event_adds_id_and_time():
    event = build_event(_row())

    assert len(event["event_id"]) == 36  # uuid4 string
    assert event["event_time"].endswith("Z")


def test_build_event_selects_relevant_fields():
    event = build_event(_row())

    assert event["Country"] == "Israel"
    assert event["LanguageHaveWorkedWith"] == "Python;SQL"
    assert event["YearsCodePro"] == 5.0
    assert event["ConvertedCompYearly"] == 125000.0


def test_build_event_handles_years_code_pro_special_values():
    assert build_event(_row(YearsCodePro="Less than 1 year"))["YearsCodePro"] == 0.5
    assert build_event(_row(YearsCodePro="More than 50 years"))["YearsCodePro"] == 51.0


def test_build_event_treats_na_and_blank_as_missing():
    event = build_event(_row(Country="NA", EdLevel="", YearsCodePro="NA", ConvertedCompYearly="NA"))

    assert event["Country"] is None
    assert event["EdLevel"] is None
    assert event["YearsCodePro"] is None
    assert event["ConvertedCompYearly"] is None


def test_build_event_strips_commas_from_target():
    event = build_event(_row(ConvertedCompYearly="1,250,000"))

    assert event["ConvertedCompYearly"] == 1250000.0


def test_build_event_handles_non_numeric_target_gracefully():
    event = build_event(_row(ConvertedCompYearly="not a number"))

    assert event["ConvertedCompYearly"] is None
