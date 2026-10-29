"""
test_predictor.py - automated tests for validations and business rules.
Run:  pytest -v
Requires the models/ folder (run the notebook first).
"""
from datetime import date

import pandas as pd
import pytest

import predictor as P


@pytest.fixture(scope="module")
def bundle():
    return P.ModelBundle()


# ---------- Derived features ----------
@pytest.mark.parametrize("d, season", [
    (date(2012, 1, 15), 1), (date(2012, 3, 20), 1), (date(2012, 3, 21), 2),
    (date(2012, 6, 21), 3), (date(2012, 9, 23), 4), (date(2012, 12, 21), 1),
])
def test_season_matches_dataset(d, season):
    assert P.season_of(d) == season


def test_season_matches_every_row_of_dataset():
    df = pd.read_csv("day.csv", parse_dates=["dteday"])
    derived = df["dteday"].dt.date.map(P.season_of)
    assert (derived == df["season"]).all()


def test_weekday_and_workingday_match_dataset():
    df = pd.read_csv("day.csv", parse_dates=["dteday"])
    for _, r in df.iterrows():
        d = r["dteday"].date()
        assert P.weekday_of(d) == r["weekday"]
        wd = int(P.weekday_of(d) not in (0, 6) and not r["holiday"])
        assert wd == r["workingday"]


# ---------- Hard validation rules ----------
@pytest.mark.parametrize("kwargs, msg", [
    (dict(day=""), "Date is required"),
    (dict(day="31/02/2012"), "not a valid date"),
    (dict(temp_c=-5), "Temperature must be"),
    (dict(temp_c=50), "Temperature must be"),
    (dict(temp_c="hot"), "must be a number"),
    (dict(humidity=120), "Humidity must be"),
    (dict(windspeed=-1), "Wind speed must be"),
    (dict(weather="Tornado"), "Unknown weather"),
    (dict(weather=7), "Weather code"),
    (dict(day="2012-06-16", holiday=True), "public holidays"),   # Saturday
    (dict(holiday="maybe"), "Holiday must be"),
])
def test_hard_rules_reject(bundle, kwargs, msg):
    args = dict(day="2012-06-18", holiday=False, weather=1, temp_c=25, humidity=60, windspeed=12)
    args.update(kwargs)
    with pytest.raises(P.ValidationError, match=msg):
        P.validate(**args, ranges=bundle.ranges)


# ---------- Soft rules (warnings) ----------
def test_future_year_warns(bundle):
    inp = P.validate("2026-06-18", False, 1, 25, 60, 12, bundle.ranges)
    assert any("2012 demand level" in w for w in inp.warnings)


def test_heavy_weather_warns(bundle):
    inp = P.validate("2012-06-18", False, 4, 25, 60, 12, bundle.ranges)
    assert any("never appears" in w for w in inp.warnings)


def test_normal_day_has_no_warnings(bundle):
    inp = P.validate("2012-06-18", False, 1, 25, 60, 12, bundle.ranges)
    assert inp.warnings == []


# ---------- Predictions ----------
def test_prediction_is_sensible(bundle):
    res = P.predict_day(bundle, "2012-06-18", False, "Clear / few clouds", 27, 55, 10)
    assert 0 <= res["mlr"] <= 10000 and 0 <= res["ann"] <= 10000
    assert res["plan"]["level"] in ("Low", "Medium", "High")
    assert res["plan"]["staff"] >= P.MIN_STAFF


def test_bad_weather_lowers_demand(bundle):
    good = P.predict_day(bundle, "2012-06-18", False, 1, 27, 55, 10)
    bad = P.predict_day(bundle, "2012-06-18", False, 3, 27, 55, 10)
    assert bad["plan"]["pred"] < good["plan"]["pred"]


def test_plan_never_negative():
    p = P.plan(-300, 800, 3700, 5200)
    assert p["pred"] == 0 and p["low"] == 0 and p["level"] == "Low"


# ---------- Weekly planner ----------
def test_week_sample_file(bundle):
    out = P.predict_week(bundle, pd.read_csv("sample_week.csv"))
    assert len(out) == 7
    assert not out["status"].str.startswith("Error").any()


def test_week_bad_rows_are_reported_not_fatal(bundle):
    df = pd.DataFrame({"date": ["2012-07-02", "not-a-date", "2012-07-02"],
                       "holiday": [0, 0, 0], "weather": [1, 1, 1],
                       "temp_c": [25, 25, 25], "humidity": [60, 60, 60], "windspeed": [10, 10, 10]})
    out = P.predict_week(bundle, df)
    assert out.loc[0, "status"] == "OK"
    assert out.loc[1, "status"].startswith("Error")
    assert "Duplicate" in out.loc[2, "status"]


def test_week_missing_column(bundle):
    with pytest.raises(P.ValidationError, match="Missing column"):
        P.predict_week(bundle, pd.DataFrame({"date": ["2012-07-02"]}))


def test_week_too_many_rows(bundle):
    df = pd.read_csv("sample_week.csv")
    with pytest.raises(P.ValidationError, match="Too many rows"):
        P.predict_week(bundle, pd.concat([df] * 5, ignore_index=True))


def test_missing_model_folder():
    with pytest.raises(FileNotFoundError):
        P.ModelBundle(model_dir="does_not_exist")


# ---------- Hourly ANN ----------
def test_hourly_profile_shape_and_sum(bundle):
    res = P.predict_day(bundle, "2012-06-18", False, 1, 27, 55, 10)
    assert len(res["hourly"]) == 24
    assert (res["hourly"] >= 0).all()
    assert abs(res["hourly"].sum() - res["hann"]) < 1e-6


def test_working_day_has_commute_peaks(bundle):
    res = P.predict_day(bundle, "2012-06-18", False, 1, 27, 55, 10)   # Monday
    assert any(7 <= h <= 9 for h in res["peaks"])
    assert any(16 <= h <= 19 for h in res["peaks"])


def test_weekend_has_no_morning_commute_peak(bundle):
    res = P.predict_day(bundle, "2012-06-16", False, 1, 27, 55, 10)   # Saturday
    assert not any(7 <= h <= 8 for h in res["peaks"])


def test_peak_windows_edge_cases():
    assert P.peak_windows([0] * 24) == []
    flat = [10] * 24
    assert len(P.peak_windows(flat)) == 2


def test_recommended_model_is_served(bundle):
    assert bundle.best_model in ("MLR", "ANN", "Hourly ANN")


def test_adjacent_peaks_are_merged():
    hourly = [0] * 24
    hourly[11], hourly[12] = 250, 280
    assert P.peak_windows(hourly) == [12]
