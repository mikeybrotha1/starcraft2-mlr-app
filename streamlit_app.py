import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.stats import t
import matplotlib.ticker as ticker

# Streamlit setup
st.set_page_config(page_title="Starcraft II Game Time Predictor", layout="wide")

# Header
st.title("⏳ Starcraft II Game Time Predictor")
st.write("Predict how long a match might last based on your resource management.")
st.caption("Built using multiple linear regression trained on over 100 StarCraft II matches.")
st.markdown("---")

# Sidebar links
st.sidebar.markdown("📂 [View Project on GitHub](https://github.com/mikeybrotha1/starcraft2-mlr-app")
st.sidebar.markdown("👾 [Chat on Reddit](https://www.reddit.com/user/Glad-Ad-7611)")
st.markdown("🚀 Powered by Python, Streamlit, and Scikit-learn. Not affiliated with Blizzard Entertainment.")

# Load data
try:
    df = pd.read_excel("Starcraft II MLR.xlsx", sheet_name="Raw Data")
except FileNotFoundError:
    st.error("❌ Could not find 'Starcraft II MLR.xlsx'. Please make sure the file is in the correct directory.")
    st.stop()

# Preprocess
df.rename(columns={
    "Game Time (Seconds)": "Game_Time",
    "Average Unspent Resources": "Avg_Unspent",
    "Time Supply Capped": "Supply_Capped",
    "Workers Created": "Workers_Created"
}, inplace=True)

X = df[["Avg_Unspent", "Supply_Capped", "Workers_Created"]]
y = df["Game_Time"]
model = LinearRegression().fit(X, y)

# User Inputs
st.subheader("🎮 Enter Your In-Game Stats")
col1, col2, col3 = st.columns(3)
with col1:
    avg_unspent = st.number_input("Avg Unspent Resources", 0, 3000, 500)
with col2:
    supply_capped = st.number_input("Supply Capped Time (s)", 0, 300, 30)
with col3:
    workers_created = st.number_input("Workers Created", 0, 100, 40)

# Prediction
if st.button("Predict Game Time"):
    input_data = np.array([[avg_unspent, supply_capped, workers_created]])
    pred_seconds = model.predict(input_data)[0]

    # Confidence & prediction intervals
    y_pred = model.predict(X)
    residuals = y - y_pred
    mse = np.mean(residuals ** 2)

    n = len(df)
    p = X.shape[1]
    X_design = np.column_stack((np.ones(n), X))
    x_new = np.array([[1, avg_unspent, supply_capped, workers_created]])

    XTX_inv = np.linalg.inv(X_design.T @ X_design)
    se_pred = np.sqrt(mse * (x_new @ XTX_inv @ x_new.T)[0, 0])
    se_total = np.sqrt(mse * (1 + (x_new @ XTX_inv @ x_new.T)[0, 0]))
    t_value = t.ppf(0.975, n - p - 1)

    ci_lower = pred_seconds - t_value * se_pred
    ci_upper = pred_seconds + t_value * se_pred
    pi_lower = pred_seconds - t_value * se_total
    pi_upper = pred_seconds + t_value * se_total

    def to_mmss(seconds):
        return int(seconds // 60), int(seconds % 60)

    pred_m, pred_s = to_mmss(pred_seconds)
    st.success(f"🎯 Predicted Game Time: {pred_m} minutes and {pred_s} seconds")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 2.5))

    def seconds_to_mmss(x, pos):
        m = int(x // 60)
        s = int(x % 60)
        return f"{m}:{s:02d}"

    ax.axvspan(ci_lower, ci_upper, color="green", alpha=0.3, label="Confidence Interval")
    ax.axvspan(pi_lower, pi_upper, color="blue", alpha=0.2, label="Prediction Interval")
    ax.plot(pred_seconds, 0.5, 'ro', markersize=10, label="Predicted Game Time")

    ax.set_yticks([])
    ax.set_xlabel("Game Time")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(seconds_to_mmss))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=3)
    st.pyplot(fig)

    # Strategy Output
    st.markdown("---")
    st.subheader("🧙 Strategy Wizard")
    st.markdown(f"""
    **Your prediction places this game around {pred_m}:{pred_s:02d} minutes.** Here's how to use that:

    🔵 **Prediction Interval** (~{int(pi_lower // 60)}:{int(pi_lower % 60):02d} to {int(pi_upper // 60)}:{int(pi_upper % 60):02d})  
    > This wide range shows the possible *volatility* of the match. Use it to time:
    - Early **harassment or cheese** attacks before {int(ci_lower // 60)}:{int(ci_lower % 60):02d}
    - **Defensive preparations** before {int(pi_upper // 60)}:{int(pi_upper % 60):02d}

    🟢 **Confidence Interval** (~{int(ci_lower // 60)}:{int(ci_lower % 60):02d} to {int(ci_upper // 60)}:{int(ci_upper % 60):02d})  
    > More stable games land here. Consider:
    - **Tech transitions** (e.g., Cloak, Medivacs, Spire) around this timing
    - Mid-game **economic push** or **third base** safely between these minutes

    🔴 **Max-Aggression 'All-in' Strategy:**  
    > Since your game is expected to end around **{pred_m}:{pred_s:02d}**, it's perfect for:
    - **2-base all-ins** (e.g., Ravager-Ling, Marine-Tank push)
    - Committing to **timing windows** that peak just before this mark

    🧠 Pro Tip:  
    > Lower predicted times usually mean **you’re inefficient or aggressive**. Lean into it and finish the game before your eco starts falling behind!
    """)
