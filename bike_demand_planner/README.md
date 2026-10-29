# Bike Demand Planner — MLR vs ANN

DDS2213 Deep Learning and Neural Networks, Coursework 1. Author: Liang Choong.

Predicts total daily bike rentals (`cnt`) with Multiple Linear Regression and an Artificial Neural Network, compares them, and turns the forecast into an operations plan (demand level, staff needed, maintenance advice, hourly peaks).

**Extension (beyond the given project):** an ANN trained on the hourly data, with its 24 hourly predictions summed to a daily total. On the same 147 test days it cuts daily RMSE from 798 (daily MLR) to 505, and also gives an hourly demand profile used to time station rebalancing.

## Files

| File | Purpose |
|---|---|
| `bike_mlr_vs_ann.ipynb` | CRISP-DM notebook: EDA, preparation, MLR, ANN tuning, evaluation, saves models |
| `day.csv` | UCI Bike Sharing dataset (daily) |
| `hour.csv` | UCI Bike Sharing dataset (hourly, used by the extension) |
| `predictor.py` | Core logic: model loading, input validation, business rules, prediction, planning |
| `app.py` | Gradio web app (single-day prediction, weekly planner, model performance) |
| `sample_week.csv` | Example 7-day weather forecast for the weekly planner |
| `test_predictor.py` | 36 automated tests for validations, business rules and the hourly model |
| `models/` | Saved models created by the notebook |

## How to run

1. `pip install -r requirements.txt`
2. Optional: run every cell in `bike_mlr_vs_ann.ipynb` to retrain (5-10 minutes). Trained models are already in `models/`, and retraining changes the numbers slightly from the report
3. `pytest -v` (optional, checks all rules)
4. `python app.py`, then open http://127.0.0.1:7860

## Validations and business rules

Hard rules (input rejected with a clear message):
- Date required and must be valid (YYYY-MM-DD)
- Temperature 0–41 °C, humidity 0–100 %, wind 0–67 (dataset scale)
- Numbers must be numeric and finite
- Weather must be one of the 4 dataset categories
- A public holiday cannot be on a weekend (true for every holiday in the dataset)
- Weekly file: required columns, max 31 rows, duplicate dates rejected, bad rows reported without stopping the others

Soft rules (prediction allowed, warning shown):
- Year outside 2011–2012 (model treats it as the nearest year)
- Heavy rain/snow (never seen in training data)
- Temperature, humidity or wind outside the training range
- Humidity below 10 %
- Negative predictions are clipped to 0

Derived automatically from the date (user cannot enter inconsistent values):
- Season (dataset definition: 1 = 21 Dec–20 Mar, 2 = 21 Mar–20 Jun, 3 = 21 Jun–22 Sep, 4 = 23 Sep–20 Dec)
- Month, weekday, working day

Planning assumptions (edit in `predictor.py`):
- 1 field staff per 1,500 rentals, minimum 2
- Demand level thresholds = 33rd and 66th percentile of historical daily rentals
- Likely range = prediction ± test RMSE of the recommended model
- Recommended model = lowest 5-fold cross-validated RMSE
- Rebalance time = 1 hour before each main peak (morning peak 05:00-11:00, afternoon peak 12:00-21:00, counted only if at least half the day's highest hour)
- The hourly profile uses the day's weather for every hour (hourly forecasts are not entered)
