"""
app.py - Bike Demand Planner (Gradio interface)
Run:  python app.py   then open the link shown in the terminal.

Tabs:
  1. Single-day prediction  - MLR vs ANN vs Hourly ANN + hourly profile + operations plan
  2. Weekly planner         - upload a weather forecast CSV, get a 7-day plan
  3. Model performance      - test-set and cross-validation metrics
"""
import os
import sys
import traceback

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import predictor as P

try:
    BUNDLE = P.ModelBundle()
except (FileNotFoundError, RuntimeError) as e:
    print(f"[ERROR] {e}")
    sys.exit(1)

META = BUNDLE.meta
BEST = BUNDLE.best_model
LC_PATH = os.path.join(P.MODEL_DIR, "learning_curve.png")


# ---------------------------------------------------------------------------
# Tab 1: single day
# ---------------------------------------------------------------------------
def ui_single(day, holiday, weather, temp_c, humidity, windspeed):
    try:
        res = P.predict_day(BUNDLE, day, holiday, weather, temp_c, humidity, windspeed)
    except P.ValidationError as e:
        gr.Warning(str(e))
        return f"### ❌ Input error\n{e}", None, None
    except Exception:
        traceback.print_exc()
        return "### ❌ Unexpected error\nSomething went wrong while predicting. Check the terminal for details.", None, None

    inp, p = res["input"], res["plan"]
    d = inp.day
    f = res["features"]
    lines = [
        f"## {d:%A, %d %B %Y}",
        f"**Season:** {P.SEASON_NAMES[f['season']]} &nbsp;|&nbsp; "
        f"**Working day:** {'Yes' if f['workingday'] else 'No'} &nbsp;|&nbsp; "
        f"**Weather:** {P.WEATHER_BY_CODE[inp.weather]}",
        "",
        "| Model | Predicted rentals |",
        "|---|---|",
        f"| Multiple Linear Regression (daily data) | {res['mlr']:,.0f} |",
        f"| Artificial Neural Network (daily data) | {res['ann']:,.0f} |",
        f"| Artificial Neural Network (hourly data, summed) | {res['hann']:,.0f} |",
        "",
        f"### Recommended forecast ({BEST}, lowest cross-validated error): **{p['pred']:,.0f} rentals**",
        f"Likely range: {p['low']:,.0f} - {p['high']:,.0f} (± test RMSE)",
        "",
        f"**Demand level:** {p['level']}  ",
        f"**Field staff needed:** {p['staff']} (assumes 1 per {P.RENTALS_PER_STAFF:,} rentals, minimum {P.MIN_STAFF})  ",
        f"**Action:** {p['action']}  ",
        "**Rebalance stations by:** " + (", ".join(f"{max(h - 1, 0):02d}:00 (peak at {h:02d}:00)" for h in res["peaks"])
                                         if res["peaks"] else "no clear peak"),
    ]
    if inp.warnings:
        lines += ["", "### ⚠️ Warnings"] + [f"- {w}" for w in inp.warnings]
    return "\n".join(lines), chart_single(res), chart_hourly(res)


def chart_single(res):
    p = res["plan"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    names = ["MLR", "ANN", "Hourly ANN"]
    vals = [res["mlr"], res["ann"], res["hann"]]
    bars = ax.barh(names, vals, color=["#4C72B0", "#DD8452", "#C44E52"])
    ax.bar_label(bars, labels=[f"{v:,.0f}" for v in vals], padding=4)
    ax.axvspan(0, META["cnt_q33"], color="#e8f4ea", zorder=0, label="Low")
    ax.axvspan(META["cnt_q33"], META["cnt_q66"], color="#fff6d9", zorder=0, label="Medium")
    ax.axvspan(META["cnt_q66"], META["cnt_max"] * 1.1, color="#fbe3e1", zorder=0, label="High")
    ax.set_xlim(0, META["cnt_max"] * 1.1)
    ax.set_xlabel("Predicted daily rentals")
    ax.set_title("Prediction vs historical demand levels")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def chart_hourly(res):
    hourly = res["hourly"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(range(24), hourly, color="#C44E52", alpha=0.85)
    for h in res["peaks"]:
        ax.annotate(f"peak {h:02d}:00", (h, hourly[h]), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=8)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Predicted rentals")
    ax.set_ylim(0, max(hourly.max() * 1.15, 1))
    ax.set_title("Hourly demand profile (Hourly ANN)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Tab 2: weekly planner
# ---------------------------------------------------------------------------
def ui_week(file):
    if file is None:
        return "Upload a CSV file first. Download `sample_week.csv` to see the format.", None, None, None
    path = file if isinstance(file, str) else file.name
    if not str(path).lower().endswith(".csv"):
        return "### ❌ Only .csv files are accepted.", None, None, None
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return f"### ❌ Could not read the file\n{e}", None, None, None
    try:
        out = P.predict_week(BUNDLE, df)
    except P.ValidationError as e:
        return f"### ❌ File error\n{e}", None, None, None
    except Exception:
        traceback.print_exc()
        return "### ❌ Unexpected error. Check the terminal for details.", None, None, None

    ok = out[~out["status"].str.startswith("Error")]
    n_err = len(out) - len(ok)
    summary = [f"### Plan for {len(ok)} day(s)"]
    if len(ok):
        summary += [f"- **Total expected rentals:** {ok['prediction'].sum():,.0f}",
                    f"- **Busiest day:** {ok.loc[ok['prediction'].idxmax(), 'date']} "
                    f"({ok['prediction'].max():,.0f})",
                    f"- **Best day for maintenance:** {ok.loc[ok['prediction'].idxmin(), 'date']} "
                    f"({ok['prediction'].min():,.0f})",
                    f"- **Peak staff needed:** {ok['staff'].max()}"]
    if n_err:
        summary.append(f"- ❌ **{n_err} row(s) rejected** - see the status column.")
    n_warn = out["status"].str.startswith("Warning").sum()
    if n_warn:
        summary.append(f"- ⚠️ {n_warn} row(s) have warnings - see the status column.")

    csv_path = "weekly_plan.csv"
    out.to_csv(csv_path, index=False)
    return "\n".join(summary), out, chart_week(ok), csv_path


def chart_week(ok):
    if ok is None or len(ok) == 0:
        return None
    fig, ax = plt.subplots(figsize=(8, 3.5))
    x = range(len(ok))
    ax.plot(x, ok["MLR"], "o-", label="MLR", color="#4C72B0")
    ax.plot(x, ok["ANN"], "s-", label="ANN", color="#DD8452")
    ax.plot(x, ok["Hourly ANN"], "^-", label="Hourly ANN", color="#C44E52")
    ax.axhline(META["cnt_q33"], ls=":", color="green", lw=1)
    ax.axhline(META["cnt_q66"], ls=":", color="red", lw=1)
    ax.set_xticks(list(x), [f"{d[5:]}\n{w[:3]}" for d, w in zip(ok["date"], ok["day"])])
    ax.set_ylabel("Predicted rentals")
    ax.set_title("Weekly demand forecast (dotted lines = Low/High thresholds)")
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Tab 3: model performance
# ---------------------------------------------------------------------------
NAMES = {"MLR": "Daily MLR", "ANN": "Daily ANN", "Hourly MLR -> daily": "Hourly MLR (summed)",
         "Hourly ANN": "Hourly ANN (summed)"}


def perf_table():
    rows = []
    for key, label in NAMES.items():
        t, c = META["test_metrics"][key], META["cv_metrics"][key]
        rows.append({"Approach": label, "Test R²": round(t["R2"], 3), "Test MAE": round(t["MAE"], 1),
                     "Test RMSE": round(t["RMSE"], 1), "Test WAPE %": round(t["WAPE"], 2),
                     "5-fold CV RMSE (mean ± sd)": f"{c['RMSE_mean']:.1f} ± {c['RMSE_std']:.1f}"})
    return pd.DataFrame(rows).sort_values("Test RMSE").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
with gr.Blocks(title="Bike Demand Planner") as demo:
    gr.Markdown("# 🚲 Bike Demand Planner\nPredicts total daily bike rentals with **Multiple Linear Regression** "
                "and **Artificial Neural Networks** (daily and hourly), then turns the forecast into an operations plan.")

    with gr.Tab("Single-day prediction"):
        with gr.Row():
            with gr.Column():
                day = gr.Textbox(label="Date (YYYY-MM-DD)", value="2012-06-18")
                holiday = gr.Checkbox(label="Public holiday")
                weather = gr.Dropdown(list(P.WEATHER), value="Clear / few clouds", label="Weather")
                temp = gr.Slider(0, 41, value=25, step=0.5, label="Temperature (°C)")
                hum = gr.Slider(0, 100, value=60, step=1, label="Humidity (%)")
                wind = gr.Slider(0, 67, value=12, step=0.5, label="Wind speed")
                btn = gr.Button("Predict", variant="primary")
                gr.Markdown("Season, month, weekday and working-day status are derived from the date automatically.")
            with gr.Column():
                out_md = gr.Markdown()
                out_plot = gr.Plot()
                out_hourly = gr.Plot()
        btn.click(ui_single, [day, holiday, weather, temp, hum, wind], [out_md, out_plot, out_hourly])
        gr.Examples(
            examples=[["2012-06-18", False, "Clear / few clouds", 27, 55, 10],
                      ["2012-01-14", False, "Light rain / light snow", 6, 85, 25],
                      ["2012-12-25", True, "Mist / cloudy", 8, 70, 12]],
            inputs=[day, holiday, weather, temp, hum, wind])

    with gr.Tab("Weekly planner"):
        gr.Markdown("Upload a weather forecast CSV with columns: "
                    "`date, holiday, weather, temp_c, humidity, windspeed` (weather = 1-4). Max 31 rows.")
        with gr.Row():
            f_in = gr.File(label="Forecast CSV", file_types=[".csv"])
            gr.File(value="sample_week.csv", label="Sample file (download)", interactive=False)
        w_btn = gr.Button("Build weekly plan", variant="primary")
        w_md = gr.Markdown()
        w_plot = gr.Plot()
        w_tbl = gr.Dataframe(wrap=True)
        w_dl = gr.File(label="Download plan")
        w_btn.click(ui_week, f_in, [w_md, w_tbl, w_plot, w_dl])

    with gr.Tab("Model performance"):
        gr.Markdown(f"Trained on {META['n_train']} days, tested on the same {META['n_test']} unseen days "
                    f"(UCI Bike Sharing, 2011-2012). Errors are for **daily totals**. "
                    f"**{BEST}** has the lowest cross-validated error and is used for the recommended forecast.")
        gr.Dataframe(perf_table(), interactive=False)
        gr.Markdown(
            f"- Daily ANN architecture (chosen by validation RMSE): **{META['ann_config']}**\n"
            f"- Paired t-test on CV folds, Daily MLR vs Daily ANN: p = {META['ttest_daily_mlr_vs_daily_ann_p']:.3f}\n"
            f"- Paired t-test on CV folds, Daily MLR vs Hourly ANN: p = {META['ttest_daily_mlr_vs_hourly_ann_p']:.3f} "
            f"(5 folds only, so treat as indicative)")
        if os.path.exists(LC_PATH):
            gr.Image(LC_PATH, label="Learning curve", show_label=True, interactive=False)

if __name__ == "__main__":
    demo.launch()
