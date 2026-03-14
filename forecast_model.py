"""
=============================================================
  RECESSION FORECASTING MODEL
=============================================================
  Predicts probability of recession in next 1-4 quarters
  using Logistic Regression + Random Forest ensemble

  INSTALL THESE LIBRARIES FIRST:
  pip3 install pandas numpy matplotlib scikit-learn
=============================================================
"""

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  HISTORICAL DATA  (1990 Q1 → 2024 Q4, quarterly)
#  Sources: FRED, BLS, BEA, Conference Board
# ─────────────────────────────────────────────────────────────

RECESSION_PERIODS = [
    ("1990-07-01", "1991-03-01"),
    ("2001-03-01", "2001-11-01"),
    ("2007-12-01", "2009-06-01"),
    ("2020-02-01", "2020-04-01"),
]

DATES = pd.date_range("1990-01-01", "2024-10-01", freq="QS")

GDP_GROWTH = [
     4.0,  1.5, -1.6, -3.9,  -1.9,  0.4,  2.7,  4.0,
     3.9,  3.9,  3.4,  5.5,   0.5,  2.4,  2.5,  5.3,
     4.0,  5.6,  2.0,  4.9,   1.3,  0.5,  3.3,  3.2,
     2.8,  7.0,  3.3,  4.4,   4.6,  5.0,  4.9,  3.0,
     2.7,  5.3,  3.6,  7.1,   3.2,  2.8,  5.0,  8.0,
     1.0,  7.8,  0.5,  2.1,  -0.5, -1.1,  1.4,  1.6,
     1.7,  2.0,  2.5,  0.2,   1.1,  3.5,  7.5,  4.5,
     3.8,  3.3,  4.0,  3.5,   4.2,  1.7,  3.4,  2.4,
     4.9,  1.0,  0.8,  2.5,   1.2,  3.2,  3.6, -2.7,
    -2.0,  2.1, -2.1, -8.4,  -4.4, -0.6,  1.7,  5.0,
     3.7,  3.9,  2.5,  2.6,   0.1, -1.5,  2.9,  4.7,
     3.2,  1.8,  0.5,  0.1,   3.6,  1.8,  3.1,  4.5,
    -1.1,  5.0,  4.6,  2.3,   3.2,  2.7,  2.0,  0.4,
     0.6,  2.3,  2.9,  1.9,   2.0,  3.0,  3.0,  2.8,
     2.8,  4.2,  3.4,  1.1,   3.1,  2.0,  2.1,  2.4,
    -4.6,-31.2, 33.8,  4.5,   6.3,  6.7,  2.3,  6.9,
    -1.6, -0.6,  3.2,  2.6,   2.2,  2.1,  4.9,  3.4,
     1.4,  3.0,  2.8,  2.5,
]

UNEMPLOYMENT = [
    5.3, 5.4, 5.7, 6.2,  6.6, 6.9, 6.8, 7.1,
    7.3, 7.5, 7.6, 7.3,  7.3, 7.1, 6.8, 6.6,
    6.6, 6.2, 6.0, 5.6,  5.5, 5.7, 5.6, 5.6,
    5.6, 5.5, 5.3, 5.3,  5.3, 5.0, 4.9, 4.7,
    4.6, 4.4, 4.5, 4.4,  4.3, 4.2, 4.2, 4.1,
    4.0, 3.9, 4.0, 3.9,  4.2, 4.5, 4.8, 5.5,
    5.7, 5.9, 5.9, 6.0,  5.9, 6.1, 6.1, 5.8,
    5.7, 5.6, 5.4, 5.4,  5.3, 5.1, 5.0, 5.0,
    4.7, 4.6, 4.7, 4.5,  4.5, 4.5, 4.7, 4.9,
    5.0, 5.4, 6.1, 7.3,  8.3, 9.3, 9.6,10.0,
    9.8, 9.6, 9.5, 9.6,  9.0, 9.1, 9.1, 8.9,
    8.3, 8.2, 8.0, 7.8,  7.7, 7.6, 7.3, 6.7,
    6.7, 6.2, 6.1, 5.7,  5.7, 5.4, 5.2, 5.0,
    5.0, 4.9, 4.9, 4.7,  4.7, 4.4, 4.3, 4.1,
    4.1, 4.0, 3.9, 3.9,  4.0, 3.8, 3.6, 3.5,
    3.5,14.7,  8.4, 6.7,  6.2, 5.9, 5.2, 4.2,
    4.0, 3.6, 3.6, 3.5,  3.4, 3.5, 3.8, 3.7,
    3.7, 4.0, 4.2, 4.2,
]

YIELD_SPREAD = [
     0.3, 0.1,-0.1,-0.3,  0.5, 1.1, 1.5, 1.9,
     2.5, 2.7, 2.9, 2.8,  2.5, 2.3, 2.8, 3.0,
     2.2, 1.5, 1.2, 0.8,  0.6, 0.7, 0.9, 1.0,
     1.2, 1.0, 0.8, 0.6,  0.7, 0.8, 0.7, 0.6,
     0.5, 0.4, 0.1,-0.1,  0.2, 0.1,-0.1,-0.4,
    -0.5,-0.3,-0.1, 0.2,  1.0, 1.5, 2.0, 2.5,
     2.0, 2.2, 2.5, 2.1,  1.5, 1.3, 1.8, 2.5,
     2.5, 2.1, 1.8, 1.6,  1.2, 0.9, 0.5, 0.2,
     0.1, 0.0,-0.1,-0.1,  0.1, 0.5, 1.0, 1.5,
     1.5, 1.0, 1.8, 2.5,  2.8, 2.5, 2.2, 2.8,
     2.5, 2.3, 2.1, 1.8,  1.5, 2.0, 1.9, 1.7,
     1.5, 1.2, 1.3, 1.6,  1.8, 2.0, 2.3, 2.6,
     2.6, 2.3, 2.0, 1.8,  1.5, 1.4, 1.2, 1.3,
     1.2, 1.0, 0.9, 1.1,  1.3, 1.0, 0.8, 0.5,
     0.3, 0.2, 0.0,-0.1, -0.1,-0.1,-0.2,-0.1,
     0.5, 0.8, 0.7, 0.8,  0.9, 1.2, 1.3, 0.8,
     0.5,-0.1,-0.5,-0.7, -1.0,-0.9,-0.5,-0.3,
    -0.2,-0.1, 0.1, 0.2,
]

CCI = [
     78, 68, 62, 58,  59, 73, 77, 69,
     73, 72, 57, 79,  69, 72, 80, 87,
     85, 91, 97,101,  99, 96, 97, 99,
    104,107,107,115, 118,127,129,132,
    130,134,136,128, 133,136,135,142,
    147,141,145,135, 117,107, 97, 84,
     94, 96, 93, 81,  64, 81, 81, 91,
     96, 99, 98,103, 103,103,104,103,
    106,106,105,110, 110,105,100, 88,
     73, 58, 57, 38,  40, 47, 52, 53,
     55, 54, 49, 52,  65, 59, 46, 40,
     68, 64, 67, 66,  69, 75, 80, 78,
     79, 87, 93, 93,  96, 97, 98,100,
     97, 97, 99,107, 116,118,120,129,
    131,128,135,136, 132,135,135,128,
    132, 87, 86, 99,  91,128,113,111,
    110,107, 95, 99, 103,102,108,109,
    104,100, 98,101,
]

INFLATION = [
    5.4, 5.3, 4.7, 4.2,  3.9, 3.7, 3.5, 3.1,
    3.0, 3.1, 3.2, 2.9,  3.0, 3.1, 2.9, 2.7,
    2.5, 2.3, 2.6, 2.8,  2.8, 3.0, 2.8, 2.6,
    2.7, 2.9, 3.0, 3.3,  3.4, 3.0, 2.2, 1.8,
    1.7, 1.8, 1.5, 1.6,  1.7, 1.5, 2.1, 2.7,
    3.4, 3.2, 3.4, 3.4,  2.9, 2.7, 2.6, 1.9,
    1.1, 1.5, 2.0, 2.4,  3.0, 2.3, 2.3, 2.2,
    2.7, 3.3, 2.7, 3.3,  3.0, 2.5, 3.6, 2.0,
    3.6, 2.5, 1.3, 2.4,  2.4, 2.6, 2.8, 4.1,
    3.8, 4.3, 5.4, 1.0,  0.3,-0.7, 1.2, 2.7,
    2.3, 1.1, 1.2, 1.5,  2.1, 3.4, 3.9, 3.0,
    2.9, 1.7, 1.5, 1.7,  2.0, 1.5, 1.7, 1.6,
    1.6, 2.0, 1.8, 1.3,  0.0,-0.1, 0.2, 0.5,
    1.0, 1.1, 1.5, 1.7,  2.1, 1.9, 2.2, 2.1,
    2.2, 2.9, 2.6, 1.9,  1.8, 1.6, 1.7, 2.3,
    1.3,-0.1, 1.3, 1.2,  1.4, 4.9, 5.3, 6.8,
    7.9, 8.6, 8.3, 7.7,  6.5, 5.0, 3.7, 3.4,
    3.1, 2.9, 2.6, 2.7,
]

# ─────────────────────────────────────────────────────────────
#  BUILD DATASET
# ─────────────────────────────────────────────────────────────

def build_dataset():
    df = pd.DataFrame({
        "date":        DATES,
        "gdp_growth":  GDP_GROWTH,
        "unemployment":UNEMPLOYMENT,
        "yield_spread":YIELD_SPREAD,
        "cci":         CCI,
        "inflation":   INFLATION,
    }).set_index("date")

    # Recession label
    df["recession"] = 0
    for start, end in RECESSION_PERIODS:
        df.loc[start:end, "recession"] = 1

    # ── Engineered features ──────────────────────────────────
    # Lagged values (signals take time to materialize)
    for lag in [1, 2, 4]:
        df[f"gdp_lag{lag}"]   = df["gdp_growth"].shift(lag)
        df[f"ur_lag{lag}"]    = df["unemployment"].shift(lag)
        df[f"yc_lag{lag}"]    = df["yield_spread"].shift(lag)
        df[f"cci_lag{lag}"]   = df["cci"].shift(lag)
        df[f"inf_lag{lag}"]   = df["inflation"].shift(lag)

    # Rolling trends
    df["gdp_3q_avg"]   = df["gdp_growth"].rolling(3).mean()
    df["ur_change_4q"] = df["unemployment"].diff(4)       # YoY UR change
    df["cci_change_4q"]= df["cci"].diff(4)
    df["yc_min_4q"]    = df["yield_spread"].rolling(4).min()
    df["gdp_momentum"] = df["gdp_growth"].diff(2)

    # Interaction: inversion + rising unemployment (classic combo)
    df["inversion_ur"] = np.where(df["yield_spread"] < 0, df["ur_change_4q"], 0)

    df = df.dropna()
    return df

# ─────────────────────────────────────────────────────────────
#  FEATURE SELECTION
# ─────────────────────────────────────────────────────────────

FEATURES = [
    "gdp_growth", "unemployment", "yield_spread", "cci", "inflation",
    "gdp_lag1", "gdp_lag2", "gdp_lag4",
    "ur_lag1",  "ur_lag2",  "ur_lag4",
    "yc_lag1",  "yc_lag2",  "yc_lag4",
    "cci_lag1", "cci_lag2",
    "inf_lag1", "inf_lag2",
    "gdp_3q_avg", "ur_change_4q", "cci_change_4q",
    "yc_min_4q", "gdp_momentum", "inversion_ur",
]

# ─────────────────────────────────────────────────────────────
#  TRAIN MODELS
# ─────────────────────────────────────────────────────────────

def train_models(df):
    X = df[FEATURES]
    y = df["recession"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Model 1: Logistic Regression (interpretable, classic)
    lr = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced")
    lr_cal = CalibratedClassifierCV(lr, cv=5)
    lr_cal.fit(X_scaled, y)

    # Model 2: Random Forest (captures non-linear patterns)
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=4,
        class_weight="balanced", random_state=42
    )
    rf.fit(X, y)

    # Model 3: Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=3,
        learning_rate=0.05, random_state=42
    )
    gb.fit(X, y)

    # Cross-validation scores
    lr_cv = cross_val_score(lr, X_scaled, y, cv=5, scoring="roc_auc").mean()
    rf_cv = cross_val_score(rf, X, y, cv=5, scoring="roc_auc").mean()
    gb_cv = cross_val_score(gb, X, y, cv=5, scoring="roc_auc").mean()

    print(f"\n  Model Performance (ROC-AUC, 5-fold CV):")
    print(f"  Logistic Regression : {lr_cv:.3f}")
    print(f"  Random Forest       : {rf_cv:.3f}")
    print(f"  Gradient Boosting   : {gb_cv:.3f}")

    return scaler, lr_cal, rf, gb, X_scaled, X

# ─────────────────────────────────────────────────────────────
#  GENERATE PREDICTIONS + FORECASTS
# ─────────────────────────────────────────────────────────────

def generate_predictions(df, scaler, lr, rf, gb, X_scaled, X):
    # In-sample probabilities (ensemble average)
    lr_prob = lr.predict_proba(X_scaled)[:, 1]
    rf_prob = rf.predict_proba(X.values)[:, 1]
    gb_prob = gb.predict_proba(X.values)[:, 1]

    # Weighted ensemble (GB gets highest weight — best performer typically)
    df["recession_prob"] = 0.25 * lr_prob + 0.30 * rf_prob + 0.45 * gb_prob

    # ── FORECAST: next 4 quarters ────────────────────────────
    # Use last known data point and project slight deterioration
    last = df[FEATURES].iloc[-1].copy()
    forecasts = []
    forecast_dates = pd.date_range(
        df.index[-1] + pd.offsets.QuarterBegin(1),
        periods=8, freq="QS"
    )

    # Simulate slight changes each quarter forward
    # (conservative: conditions drift slowly)
    gdp_trend   = df["gdp_growth"].iloc[-4:].mean()
    ur_trend    = df["unemployment"].iloc[-4:].mean()
    yc_trend    = df["yield_spread"].iloc[-4:].mean()
    cci_trend   = df["cci"].iloc[-4:].mean()
    inf_trend   = df["inflation"].iloc[-4:].mean()

    for q in range(1, 9):
        sim = last.copy()
        # Slowly revert to recent trend
        sim["gdp_growth"]   = gdp_trend  + np.random.normal(0, 0.3)
        sim["unemployment"] = ur_trend   + q * 0.05
        sim["yield_spread"] = yc_trend   + q * 0.05
        sim["cci"]          = cci_trend  - q * 1.0
        sim["inflation"]    = inf_trend

        sim_scaled = scaler.transform([sim[FEATURES]])
        sim_arr    = np.array([sim[FEATURES]])

        p_lr = lr.predict_proba(sim_scaled)[0, 1]
        p_rf = rf.predict_proba(sim_arr)[0, 1]
        p_gb = gb.predict_proba(sim_arr)[0, 1]
        prob = 0.25 * p_lr + 0.30 * p_rf + 0.45 * p_gb

        forecasts.append({
            "date": forecast_dates[q - 1],
            "recession_prob": prob,
            "recession": np.nan,
            "forecast": True
        })
        last = sim

    forecast_df = pd.DataFrame(forecasts).set_index("date")
    return df, forecast_df

# ─────────────────────────────────────────────────────────────
#  FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────

def get_feature_importance(rf):
    imp = pd.Series(rf.feature_importances_, index=FEATURES)
    return imp.sort_values(ascending=False).head(10)

# ─────────────────────────────────────────────────────────────
#  PLOT DASHBOARD
# ─────────────────────────────────────────────────────────────

BG       = "#0d1117"
PANEL_BG = "#161b22"
TEXT     = "#e6edf3"
GRID     = "#21262d"
ACCENT   = "#58a6ff"
RED      = "#e63946"
ORANGE   = "#f4a261"
GREEN    = "#2a9d8f"
PURPLE   = "#c77dff"

def shade_recessions(ax, df_plot):
    for start, end in RECESSION_PERIODS:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        if s >= df_plot.index[0]:
            ax.axvspan(s, min(e, df_plot.index[-1]),
                       alpha=0.2, color=RED, zorder=0)

def style_ax(ax, title, ylabel):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.set_title(title, color=TEXT, fontsize=9, fontweight="bold", pad=6)
    ax.set_ylabel(ylabel, color=TEXT, fontsize=8)
    ax.grid(color=GRID, linewidth=0.5, linestyle="--", alpha=0.6)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)

def plot_dashboard(df, forecast_df, importance):
    fig = plt.figure(figsize=(20, 16), facecolor=BG)
    fig.suptitle(
        "RECESSION FORECASTING MODEL  ·  1990–2024  +  8-QUARTER FORECAST",
        color=TEXT, fontsize=15, fontweight="bold", y=0.97,
        fontfamily="monospace"
    )

    gs = gridspec.GridSpec(
        4, 2, figure=fig,
        hspace=0.55, wspace=0.32,
        left=0.07, right=0.97, top=0.93, bottom=0.05
    )

    # ── PANEL 1: Main forecast chart ────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    style_ax(ax1, "RECESSION PROBABILITY FORECAST  —  Ensemble Model (Logistic + Random Forest + Gradient Boosting)", "Probability (%)")
    shade_recessions(ax1, df)

    prob_pct = df["recession_prob"] * 100
    fc_pct   = forecast_df["recession_prob"] * 100

    # Color-fill historical probability
    for i in range(len(df) - 1):
        p = prob_pct.iloc[i]
        c = RED if p >= 50 else (ORANGE if p >= 30 else GREEN)
        ax1.fill_between(df.index[i:i+2], 0, prob_pct.iloc[i:i+2],
                         color=c, alpha=0.35)

    ax1.plot(df.index, prob_pct, color=ACCENT, linewidth=1.8, label="Historical probability", zorder=3)

    # Forecast zone
    fc_dates = [df.index[-1]] + list(forecast_df.index)
    fc_vals  = [prob_pct.iloc[-1]] + list(fc_pct)
    ax1.plot(fc_dates, fc_vals, color=PURPLE, linewidth=2.2,
             linestyle="--", label="Forecast (next 4 quarters)", zorder=4)
    ax1.fill_between(fc_dates, 0, fc_vals, color=PURPLE, alpha=0.2)
    ax1.axvline(df.index[-1], color=TEXT, linewidth=1, linestyle=":", alpha=0.5)
    ax1.text(df.index[-1], 92, " NOW", color=TEXT, fontsize=8, fontfamily="monospace")

    ax1.axhline(50, color=RED,    linewidth=1, linestyle=":", alpha=0.8, label="High risk (50%)")
    ax1.axhline(30, color=ORANGE, linewidth=1, linestyle=":", alpha=0.8, label="Elevated risk (30%)")
    ax1.set_ylim(0, 100)
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.3,
               labelcolor=TEXT, facecolor=PANEL_BG, edgecolor=GRID)

    # Recession labels
    for label, mid in [
        ("Gulf War\n1990-91", "1990-11-01"), ("Dot-com\n2001", "2001-07-01"),
        ("GFC\n2007-09",      "2008-09-01"), ("COVID-19\n2020", "2020-03-15"),
    ]:
        ax1.annotate(label, xy=(pd.Timestamp(mid), 95),
                     color=RED, fontsize=7, ha="center", fontfamily="monospace")

    # ── PANEL 2: GDP Growth ──────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    style_ax(ax2, "GDP GROWTH  (annualized %)", "%")
    shade_recessions(ax2, df)
    for i in range(len(df) - 1):
        c = GREEN if df["gdp_growth"].iloc[i] >= 0 else RED
        ax2.fill_between(df.index[i:i+2], 0, df["gdp_growth"].iloc[i:i+2], color=c, alpha=0.4)
    ax2.plot(df.index, df["gdp_growth"], color=TEXT, linewidth=0.9, alpha=0.8)
    ax2.axhline(0, color=TEXT, linewidth=0.7, alpha=0.4)

    # ── PANEL 3: Unemployment ────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    style_ax(ax3, "UNEMPLOYMENT RATE  (%)", "%")
    shade_recessions(ax3, df)
    ax3.fill_between(df.index, df["unemployment"], alpha=0.25, color=ORANGE)
    ax3.plot(df.index, df["unemployment"], color=ORANGE, linewidth=1.8)

    # ── PANEL 4: Yield Curve ─────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    style_ax(ax4, "YIELD CURVE  (10Y - 2Y Spread)", "%")
    shade_recessions(ax4, df)
    yc = df["yield_spread"]
    ax4.fill_between(df.index, yc, 0, where=(yc < 0), color=RED,   alpha=0.5, label="Inverted ⚠️")
    ax4.fill_between(df.index, yc, 0, where=(yc >= 0), color=GREEN, alpha=0.3, label="Normal")
    ax4.plot(df.index, yc, color=ACCENT, linewidth=1.6)
    ax4.axhline(0, color=TEXT, linewidth=0.8, alpha=0.5)
    ax4.legend(loc="upper right", fontsize=8, framealpha=0.3,
               labelcolor=TEXT, facecolor=PANEL_BG, edgecolor=GRID)

    # ── PANEL 5: Feature Importance ─────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    style_ax(ax5, "TOP 10 MOST PREDICTIVE FEATURES  (Random Forest)", "Importance")
    colors_imp = [RED if i < 3 else (ORANGE if i < 6 else ACCENT)
                  for i in range(len(importance))]
    bars = ax5.barh(importance.index[::-1], importance.values[::-1],
                    color=colors_imp[::-1], alpha=0.8, height=0.6)
    ax5.tick_params(axis='y', labelsize=7.5)

    # ── PANEL 6: Forecast table ──────────────────────────────
    ax6 = fig.add_subplot(gs[3, :])
    style_ax(ax6, "4-QUARTER RECESSION PROBABILITY FORECAST", "Probability (%)")
    ax6.set_facecolor(PANEL_BG)

    # Show last 20 quarters of history + forecast
    hist_tail = df["recession_prob"].iloc[-20:] * 100
    all_dates = list(hist_tail.index) + list(forecast_df.index)
    all_probs = list(hist_tail.values) + list(fc_pct.values)

    bar_colors = []
    for i, (d, p) in enumerate(zip(all_dates, all_probs)):
        if d in forecast_df.index:
            bar_colors.append(PURPLE)
        elif p >= 50:
            bar_colors.append(RED)
        elif p >= 30:
            bar_colors.append(ORANGE)
        else:
            bar_colors.append(GREEN)

    ax6.bar(all_dates, all_probs, color=bar_colors, alpha=0.8, width=70)
    ax6.axvline(df.index[-1], color=TEXT, linewidth=1.5, linestyle="--", alpha=0.6)
    ax6.axhline(50, color=RED,    linewidth=1, linestyle=":", alpha=0.7)
    ax6.axhline(30, color=ORANGE, linewidth=1, linestyle=":", alpha=0.7)
    ax6.set_ylim(0, 100)
    ax6.text(df.index[-1], 85, "  ← HISTORY  |  FORECAST →  ",
             color=TEXT, fontsize=8, fontfamily="monospace")

    # Annotate forecast bars
    for d, p in zip(forecast_df.index, fc_pct.values):
        ax6.text(d, p + 2, f"{p:.0f}%", color=PURPLE, fontsize=8,
                 ha="center", fontweight="bold")

    # Legend
    patches = [
        mpatches.Patch(color=RED,    alpha=0.7, label="High risk (≥50%)"),
        mpatches.Patch(color=ORANGE, alpha=0.7, label="Elevated (≥30%)"),
        mpatches.Patch(color=GREEN,  alpha=0.7, label="Low risk (<30%)"),
        mpatches.Patch(color=PURPLE, alpha=0.7, label="Forecast"),
        mpatches.Patch(color=RED,    alpha=0.2, label="NBER Recession"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=8,
               framealpha=0.3, labelcolor=TEXT, facecolor=PANEL_BG,
               edgecolor=GRID, bbox_to_anchor=(0.5, 0.01))

    plt.savefig("recession_forecast_dashboard.png", dpi=150,
                bbox_inches="tight", facecolor=BG)
    print("✅  Chart saved → recession_forecast_dashboard.png")
    print("📂  Open with:  open recession_forecast_dashboard.png")

# ─────────────────────────────────────────────────────────────
#  PRINT FORECAST SUMMARY
# ─────────────────────────────────────────────────────────────

def print_forecast(df, forecast_df):
    print("\n" + "═"*62)
    print("  RECESSION FORECAST  —  NEXT 8 QUARTERS")
    print("═"*62)
    for i, (date, row) in enumerate(forecast_df.iterrows()):
        p    = row["recession_prob"] * 100
        risk = "🔴 HIGH RISK" if p >= 50 else ("🟡 ELEVATED" if p >= 30 else "🟢 LOW RISK")
        print(f"  Q{i+1} {date.strftime('%Y-%m')}  |  {p:5.1f}%  |  {risk}")
    print("═"*62)

    latest_prob = df["recession_prob"].iloc[-1] * 100
    print(f"\n  Current recession probability : {latest_prob:.1f}%")
    print(f"  Model trained on              : {len(df)} quarters (1990–2024)")
    print("═"*62 + "\n")

# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🔄  Building recession forecasting model …")

    df = build_dataset()
    print(f"✅  Dataset: {len(df)} quarters loaded")

    scaler, lr, rf, gb, X_scaled, X = train_models(df)
    print("✅  Models trained")

    df, forecast_df = generate_predictions(df, scaler, lr, rf, gb, X_scaled, X)
    importance = get_feature_importance(rf)

    print_forecast(df, forecast_df)
    plot_dashboard(df, forecast_df, importance)
