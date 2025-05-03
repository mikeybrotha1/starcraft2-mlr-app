# StarCraft II Game Time Predictor

This Streamlit app uses a Multiple Linear Regression (MLR) model to predict the **game duration** of a StarCraft II match based on in-game performance metrics.

## Features

- Predict **game time** based on KPIs like:
  - Average Unspent Resources
  - Time Supply Capped
  - Workers Created
- Visualize results with dynamic charts
- Load and analyze your own StarCraft II match data
- Download predictions to Excel (coming soon)

## Model Details

The model is trained using a regression analysis on match data exported from PHStat and Excel. It uses:

- Multiple Linear Regression (MLR)
- KPIs sourced from real matches
- `pandas`, `numpy`, `scikit-learn`, and `streamlit`

## Files

- `streamlit_app.py` – Main Streamlit interface
- `Starcraft II MLR.xlsx` – Source data used to build and test the model
- `requirements.txt` – Python dependencies

## Run It Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
