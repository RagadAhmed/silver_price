import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Silver Price Forecaster | AI-Powered Predictions",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #111128 30%, #0d1b2a 60%, #1a1a2e 100%);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111128 0%, #0d1b2a 100%);
        border-right: 1px solid rgba(192, 192, 192, 0.1);
    }

    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #C0C0C0 0%, #E8E8E8 30%, #A8A8A8 60%, #D4D4D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
        letter-spacing: -1px;
        text-shadow: 0 0 40px rgba(192,192,192,0.3);
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: #8892b0;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }

    .metric-card {
        background: linear-gradient(135deg, rgba(192,192,192,0.08) 0%, rgba(192,192,192,0.03) 100%);
        border: 1px solid rgba(192,192,192,0.15);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }

    .metric-card:hover {
        border-color: rgba(192,192,192,0.35);
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(192,192,192,0.1);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #C0C0C0, #E8E8E8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 0.3rem;
        font-weight: 500;
    }

    .metric-delta-up {
        color: #64ffda;
        font-size: 0.9rem;
        font-weight: 600;
    }

    .metric-delta-down {
        color: #ff6b6b;
        font-size: 0.9rem;
        font-weight: 600;
    }

    .section-header {
        font-size: 1.6rem;
        font-weight: 700;
        color: #ccd6f6;
        margin: 2rem 0 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(192,192,192,0.2);
    }

    .rmse-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(100,255,218,0.15), rgba(100,255,218,0.05));
        border: 1px solid rgba(100,255,218,0.3);
        border-radius: 24px;
        padding: 0.5rem 1.5rem;
        color: #64ffda;
        font-weight: 600;
        font-size: 1rem;
        margin: 0.5rem;
    }

    .insight-box {
        background: linear-gradient(135deg, rgba(100,255,218,0.05) 0%, rgba(100,255,218,0.02) 100%);
        border-left: 3px solid #64ffda;
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        color: #a8b2d1;
    }

    .model-tag {
        display: inline-block;
        background: rgba(192,192,192,0.1);
        border: 1px solid rgba(192,192,192,0.2);
        border-radius: 8px;
        padding: 0.3rem 0.8rem;
        color: #ccd6f6;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 0.2rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(192,192,192,0.05);
        border-radius: 12px;
        padding: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #8892b0;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(192,192,192,0.15) !important;
        color: #ccd6f6 !important;
    }

    div[data-testid="stMetricValue"] { color: #C0C0C0; }
    div[data-testid="stMetricLabel"] { color: #8892b0; }

    .footer {
        text-align: center;
        color: #4a5568;
        padding: 2rem 0 1rem;
        font-size: 0.8rem;
        border-top: 1px solid rgba(192,192,192,0.1);
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Helper Functions ────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_silver_data(period_years=5):
    """Fetch silver futures data from Yahoo Finance."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_years * 365)
    ticker = yf.Ticker("SI=F")
    df = ticker.history(start=start_date, end=end_date, interval="1d")
    df = df.reset_index()
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.dropna(subset=["Close"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def create_features(df):
    """Create time-series features for XGBoost."""
    data = df.copy()
    data["day_of_week"] = data["Date"].dt.dayofweek
    data["day_of_month"] = data["Date"].dt.day
    data["month"] = data["Date"].dt.month
    data["quarter"] = data["Date"].dt.quarter
    data["year"] = data["Date"].dt.year
    data["day_of_year"] = data["Date"].dt.dayofyear
    data["week_of_year"] = data["Date"].dt.isocalendar().week.astype(int)

    # Lag features
    for lag in [1, 3, 5, 7, 14, 21, 30, 60, 90]:
        data[f"lag_{lag}"] = data["Close"].shift(lag)

    # Rolling statistics
    for window in [7, 14, 30, 60, 90]:
        data[f"rolling_mean_{window}"] = data["Close"].rolling(window=window).mean()
        data[f"rolling_std_{window}"] = data["Close"].rolling(window=window).std()

    # Exponential moving averages
    for span in [7, 21, 50]:
        data[f"ema_{span}"] = data["Close"].ewm(span=span, adjust=False).mean()

    # Price momentum
    for period in [7, 14, 30]:
        data[f"momentum_{period}"] = data["Close"].pct_change(periods=period)

    # Volatility
    data["volatility_30"] = data["Close"].rolling(window=30).std() / data["Close"].rolling(window=30).mean()

    data = data.dropna()
    return data


def train_prophet_model(df):
    """Train Facebook Prophet model."""
    prophet_df = df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
        seasonality_mode="multiplicative",
        n_changepoints=30,
    )
    model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    model.add_seasonality(name="quarterly", period=91.25, fourier_order=3)
    model.fit(prophet_df)
    return model


def train_xgboost_model(df):
    """Train XGBoost model with engineered features."""
    data = create_features(df)

    feature_cols = [c for c in data.columns if c not in ["Date", "Close", "Open", "High", "Low", "Volume"]]

    # Use last 20% as validation
    split_idx = int(len(data) * 0.8)
    train = data.iloc[:split_idx]
    val = data.iloc[split_idx:]

    X_train, y_train = train[feature_cols], train["Close"]
    X_val, y_val = val[feature_cols], val["Close"]

    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    val_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    mae = mean_absolute_error(y_val, val_pred)
    r2 = r2_score(y_val, val_pred)

    return model, feature_cols, rmse, mae, r2, val, val_pred


def forecast_prophet(model, periods=365):
    """Generate Prophet forecast."""
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast


def forecast_xgboost(model, df, feature_cols, periods=365):
    """Generate XGBoost forecast iteratively."""
    data = create_features(df)
    last_data = data.copy()

    predictions = []
    current_date = df["Date"].max()

    for i in range(periods):
        current_date += timedelta(days=1)
        # Skip weekends
        if current_date.weekday() >= 5:
            continue

        new_row = pd.DataFrame({"Date": [current_date], "Close": [np.nan]})
        # Copy last known OHLV
        new_row["Open"] = last_data["Close"].iloc[-1]
        new_row["High"] = last_data["Close"].iloc[-1]
        new_row["Low"] = last_data["Close"].iloc[-1]
        new_row["Volume"] = last_data["Volume"].iloc[-1] if "Volume" in last_data.columns else 0

        temp_df = pd.concat([df, pd.DataFrame({
            "Date": [d for d, _ in predictions] + [current_date],
            "Close": [p for _, p in predictions] + [last_data["Close"].iloc[-1]],
            "Open": [last_data["Open"].iloc[-1]] * (len(predictions) + 1),
            "High": [last_data["High"].iloc[-1]] * (len(predictions) + 1),
            "Low": [last_data["Low"].iloc[-1]] * (len(predictions) + 1),
            "Volume": [0] * (len(predictions) + 1),
        })], ignore_index=True)

        temp_features = create_features(temp_df)
        if len(temp_features) == 0:
            continue

        last_features = temp_features[feature_cols].iloc[-1:]
        pred = model.predict(last_features)[0]
        predictions.append((current_date, pred))

    forecast_df = pd.DataFrame(predictions, columns=["Date", "Predicted"])
    return forecast_df


def ensemble_forecast(prophet_fc, xgb_fc, weight_prophet=0.4, weight_xgb=0.6):
    """Combine Prophet and XGBoost forecasts with weighted average."""
    prophet_future = prophet_fc[prophet_fc["ds"] > prophet_fc["ds"].iloc[-366]].copy()
    prophet_future = prophet_future.rename(columns={"ds": "Date", "yhat": "Prophet_Pred"})
    prophet_future = prophet_future[["Date", "Prophet_Pred"]]

    xgb_future = xgb_fc.rename(columns={"Predicted": "XGB_Pred"})

    merged = pd.merge_asof(
        prophet_future.sort_values("Date"),
        xgb_future.sort_values("Date"),
        on="Date",
        direction="nearest",
        tolerance=pd.Timedelta("1D"),
    )
    merged = merged.dropna()
    merged["Ensemble_Pred"] = (
        weight_prophet * merged["Prophet_Pred"] +
        weight_xgb * merged["XGB_Pred"]
    )
    return merged


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
        <span style="font-size: 3rem;">🪙</span>
        <h2 style="color: #C0C0C0; margin: 0.5rem 0 0;">Silver Forecaster</h2>
        <p style="color: #8892b0; font-size: 0.85rem;">AI-Powered Price Prediction</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### ⚙️ Configuration")
    forecast_days = st.slider("📅 Forecast Horizon (Days)", 30, 365, 365, step=30)

    st.markdown("### 🧠 Model Weights")
    prophet_weight = st.slider("Prophet Weight", 0.0, 1.0, 0.4, 0.05)
    xgb_weight = 1.0 - prophet_weight
    st.caption(f"XGBoost Weight: **{xgb_weight:.2f}**")

    st.markdown("---")
    st.markdown("### 📊 Data Source")
    st.markdown("""
    <div style="background: rgba(192,192,192,0.05); border-radius:12px; padding:1rem; border: 1px solid rgba(192,192,192,0.1);">
        <p style="color:#8892b0; font-size:0.8rem; margin:0;">
            <strong style="color:#C0C0C0;">Ticker:</strong> SI=F (Silver Futures)<br/>
            <strong style="color:#C0C0C0;">Source:</strong> Yahoo Finance<br/>
            <strong style="color:#C0C0C0;">Period:</strong> Last 5 Years<br/>
            <strong style="color:#C0C0C0;">Models:</strong> Prophet + XGBoost
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ─── Main Content ────────────────────────────────────────────────────────────
st.markdown('<h1 class="hero-title">🪙 Silver Price Forecaster</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">AI-powered silver futures prediction engine using Prophet & XGBoost ensemble modeling</p>', unsafe_allow_html=True)

# Load data
with st.spinner("🔄 Fetching silver price data from Yahoo Finance..."):
    df = fetch_silver_data(5)

if df.empty or len(df) < 100:
    st.error("⚠️ Unable to fetch sufficient silver price data. Please try again later.")
    st.stop()

# ─── Key Metrics ─────────────────────────────────────────────────────────────
latest_price = df["Close"].iloc[-1]
prev_price = df["Close"].iloc[-2]
price_change = latest_price - prev_price
price_change_pct = (price_change / prev_price) * 100
high_52w = df["Close"].tail(252).max()
low_52w = df["Close"].tail(252).min()
avg_volume = df["Volume"].tail(30).mean()

col1, col2, col3, col4 = st.columns(4)

with col1:
    delta_class = "metric-delta-up" if price_change >= 0 else "metric-delta-down"
    delta_sign = "+" if price_change >= 0 else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">${latest_price:.2f}</div>
        <div class="metric-label">Current Price</div>
        <div class="{delta_class}">{delta_sign}{price_change:.2f} ({delta_sign}{price_change_pct:.2f}%)</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">${high_52w:.2f}</div>
        <div class="metric-label">52-Week High</div>
        <div style="color:#8892b0; font-size:0.85rem;">{((latest_price/high_52w)-1)*100:.1f}% from high</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">${low_52w:.2f}</div>
        <div class="metric-label">52-Week Low</div>
        <div style="color:#64ffda; font-size:0.85rem;">+{((latest_price/low_52w)-1)*100:.1f}% from low</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(df):,}</div>
        <div class="metric-label">Data Points</div>
        <div style="color:#8892b0; font-size:0.85rem;">{df['Date'].min().strftime('%b %Y')} — {df['Date'].max().strftime('%b %Y')}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# ─── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📈 Historical Analysis", "🔮 Forecast", "🏆 Model Performance", "📋 Data Explorer"])

# ─── Tab 1: Historical Analysis ─────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">📈 Silver Price — Historical Overview</div>', unsafe_allow_html=True)

    # Candlestick chart
    fig_candle = go.Figure(data=[go.Candlestick(
        x=df["Date"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        increasing_line_color="#64ffda",
        decreasing_line_color="#ff6b6b",
        increasing_fillcolor="rgba(100,255,218,0.3)",
        decreasing_fillcolor="rgba(255,107,107,0.3)",
    )])

    # Add moving averages
    df["MA_50"] = df["Close"].rolling(50).mean()
    df["MA_200"] = df["Close"].rolling(200).mean()

    fig_candle.add_trace(go.Scatter(
        x=df["Date"], y=df["MA_50"],
        name="50-Day MA", line=dict(color="#ffd700", width=1.5, dash="dot"),
    ))
    fig_candle.add_trace(go.Scatter(
        x=df["Date"], y=df["MA_200"],
        name="200-Day MA", line=dict(color="#ff69b4", width=1.5, dash="dot"),
    ))

    fig_candle.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,26,0.8)",
        height=500,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(gridcolor="rgba(192,192,192,0.05)"),
        yaxis=dict(gridcolor="rgba(192,192,192,0.05)", title="Price (USD)"),
    )
    st.plotly_chart(fig_candle, use_container_width=True)

    # Volume chart
    col_v1, col_v2 = st.columns([2, 1])

    with col_v1:
        st.markdown('<div class="section-header">📊 Trading Volume</div>', unsafe_allow_html=True)
        colors = ["#64ffda" if df["Close"].iloc[i] >= df["Open"].iloc[i] else "#ff6b6b" for i in range(len(df))]
        fig_vol = go.Figure(data=[go.Bar(
            x=df["Date"], y=df["Volume"],
            marker_color=colors, opacity=0.6,
        )])
        fig_vol.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10,10,26,0.8)",
            height=250,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(gridcolor="rgba(192,192,192,0.05)"),
            yaxis=dict(gridcolor="rgba(192,192,192,0.05)", title="Volume"),
            showlegend=False,
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    with col_v2:
        st.markdown('<div class="section-header">📉 Return Distribution</div>', unsafe_allow_html=True)
        daily_returns = df["Close"].pct_change().dropna() * 100
        fig_dist = go.Figure(data=[go.Histogram(
            x=daily_returns, nbinsx=80,
            marker_color="rgba(192,192,192,0.5)",
            marker_line_color="rgba(192,192,192,0.8)",
            marker_line_width=0.5,
        )])
        fig_dist.add_vline(x=0, line_dash="dash", line_color="#ffd700", line_width=1)
        fig_dist.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10,10,26,0.8)",
            height=250,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(title="Daily Return (%)", gridcolor="rgba(192,192,192,0.05)"),
            yaxis=dict(title="Frequency", gridcolor="rgba(192,192,192,0.05)"),
            showlegend=False,
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    # Monthly heatmap
    st.markdown('<div class="section-header">🗓️ Monthly Return Heatmap</div>', unsafe_allow_html=True)
    df_monthly = df.set_index("Date")["Close"].resample("ME").last().pct_change() * 100
    heatmap_data = pd.DataFrame({
        "Year": df_monthly.index.year,
        "Month": df_monthly.index.month,
        "Return": df_monthly.values,
    }).dropna()

    pivot = heatmap_data.pivot_table(values="Return", index="Year", columns="Month", aggfunc="mean")
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig_heat = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=month_names[:pivot.shape[1]],
        y=pivot.index.astype(str),
        colorscale=[[0, "#ff6b6b"], [0.5, "#1a1a2e"], [1, "#64ffda"]],
        text=np.round(pivot.values, 1),
        texttemplate="%{text}%",
        textfont=dict(size=11),
        hoverongaps=False,
        colorbar=dict(title="Return %"),
    ))
    fig_heat.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,26,0.8)",
        height=250,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_heat, use_container_width=True)


# ─── Tab 2: Forecast ────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">🔮 AI-Powered Silver Price Forecast</div>', unsafe_allow_html=True)

    with st.spinner("🧠 Training Prophet model..."):
        prophet_model = train_prophet_model(df)
        prophet_forecast = forecast_prophet(prophet_model, periods=forecast_days)

    with st.spinner("⚡ Training XGBoost model..."):
        xgb_model, feature_cols, xgb_rmse, xgb_mae, xgb_r2, xgb_val, xgb_val_pred = train_xgboost_model(df)
        xgb_forecast = forecast_xgboost(xgb_model, df, feature_cols, periods=forecast_days)

    # Prophet validation RMSE
    prophet_train_df = df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    split_idx = int(len(prophet_train_df) * 0.8)
    prophet_val = prophet_train_df.iloc[split_idx:]
    prophet_pred_val = prophet_forecast[prophet_forecast["ds"].isin(prophet_val["ds"])]

    if len(prophet_pred_val) > 0:
        merged_val = prophet_val.merge(prophet_pred_val[["ds", "yhat"]], on="ds", how="inner")
        prophet_rmse = np.sqrt(mean_squared_error(merged_val["y"], merged_val["yhat"]))
        prophet_mae = mean_absolute_error(merged_val["y"], merged_val["yhat"])
        prophet_r2 = r2_score(merged_val["y"], merged_val["yhat"])
    else:
        prophet_rmse, prophet_mae, prophet_r2 = 0, 0, 0

    # Ensemble
    with st.spinner("🔗 Creating ensemble forecast..."):
        ensemble_fc = ensemble_forecast(prophet_forecast, xgb_forecast, prophet_weight, xgb_weight)

    # Forecast chart
    fig_forecast = go.Figure()

    # Historical
    fig_forecast.add_trace(go.Scatter(
        x=df["Date"], y=df["Close"],
        name="Historical Price",
        line=dict(color="#C0C0C0", width=2),
        fill="tozeroy",
        fillcolor="rgba(192,192,192,0.05)",
    ))

    # Prophet forecast
    future_prophet = prophet_forecast[prophet_forecast["ds"] > df["Date"].max()]
    fig_forecast.add_trace(go.Scatter(
        x=future_prophet["ds"], y=future_prophet["yhat"],
        name="Prophet Forecast",
        line=dict(color="#ffd700", width=2, dash="dot"),
    ))

    # Prophet confidence interval
    fig_forecast.add_trace(go.Scatter(
        x=pd.concat([future_prophet["ds"], future_prophet["ds"][::-1]]),
        y=pd.concat([future_prophet["yhat_upper"], future_prophet["yhat_lower"][::-1]]),
        fill="toself",
        fillcolor="rgba(255,215,0,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Prophet CI (95%)",
        showlegend=True,
    ))

    # XGBoost forecast
    if not xgb_forecast.empty:
        fig_forecast.add_trace(go.Scatter(
            x=xgb_forecast["Date"], y=xgb_forecast["Predicted"],
            name="XGBoost Forecast",
            line=dict(color="#ff69b4", width=2, dash="dash"),
        ))

    # Ensemble forecast
    if not ensemble_fc.empty:
        fig_forecast.add_trace(go.Scatter(
            x=ensemble_fc["Date"], y=ensemble_fc["Ensemble_Pred"],
            name="Ensemble Forecast",
            line=dict(color="#64ffda", width=3),
        ))

    # Vertical line at forecast start
    fig_forecast.add_vline(
        x=df["Date"].max(),
        line_dash="dash",
        line_color="rgba(192,192,192,0.4)",
        line_width=1,
    )
    fig_forecast.add_annotation(
        x=df["Date"].max(),
        y=latest_price * 1.05,
        text="Forecast Start →",
        showarrow=False,
        font=dict(color="#8892b0", size=11),
    )

    fig_forecast.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,26,0.8)",
        height=550,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=11),
        ),
        xaxis=dict(gridcolor="rgba(192,192,192,0.05)", title="Date"),
        yaxis=dict(gridcolor="rgba(192,192,192,0.05)", title="Price (USD)"),
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

    # Forecast summary cards
    if not ensemble_fc.empty:
        fc_end = ensemble_fc["Ensemble_Pred"].iloc[-1]
        fc_min = ensemble_fc["Ensemble_Pred"].min()
        fc_max = ensemble_fc["Ensemble_Pred"].max()
        fc_change = ((fc_end - latest_price) / latest_price) * 100

        st.markdown("<br/>", unsafe_allow_html=True)
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            sign = "+" if fc_change >= 0 else ""
            delta_cls = "metric-delta-up" if fc_change >= 0 else "metric-delta-down"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">${fc_end:.2f}</div>
                <div class="metric-label">Predicted Price (End)</div>
                <div class="{delta_cls}">{sign}{fc_change:.1f}% from current</div>
            </div>
            """, unsafe_allow_html=True)
        with fc2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">${fc_max:.2f}</div>
                <div class="metric-label">Forecast High</div>
                <div style="color:#64ffda; font-size:0.85rem;">Peak predicted price</div>
            </div>
            """, unsafe_allow_html=True)
        with fc3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">${fc_min:.2f}</div>
                <div class="metric-label">Forecast Low</div>
                <div style="color:#ff6b6b; font-size:0.85rem;">Trough predicted price</div>
            </div>
            """, unsafe_allow_html=True)
        with fc4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{forecast_days}</div>
                <div class="metric-label">Forecast Days</div>
                <div style="color:#8892b0; font-size:0.85rem;">≈ {forecast_days//30} months ahead</div>
            </div>
            """, unsafe_allow_html=True)


# ─── Tab 3: Model Performance ───────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">🏆 Model Performance Comparison</div>', unsafe_allow_html=True)

    # RMSE badges
    st.markdown(f"""
    <div style="text-align:center; margin: 1.5rem 0;">
        <span class="rmse-badge">📊 Prophet RMSE: ${prophet_rmse:.4f}</span>
        <span class="rmse-badge">⚡ XGBoost RMSE: ${xgb_rmse:.4f}</span>
        <span class="rmse-badge">🏆 Best: {'XGBoost' if xgb_rmse < prophet_rmse else 'Prophet'}</span>
    </div>
    """, unsafe_allow_html=True)

    # Comparison table
    perf_col1, perf_col2 = st.columns(2)

    with perf_col1:
        st.markdown("""
        <div class="metric-card" style="text-align:left;">
            <h3 style="color:#ffd700; margin-top:0;">📊 Facebook Prophet</h3>
        """, unsafe_allow_html=True)
        st.markdown(f"""
            <table style="width:100%; color:#a8b2d1;">
                <tr><td>RMSE</td><td style="text-align:right; color:#64ffda; font-weight:600;">${prophet_rmse:.4f}</td></tr>
                <tr><td>MAE</td><td style="text-align:right; color:#64ffda; font-weight:600;">${prophet_mae:.4f}</td></tr>
                <tr><td>R² Score</td><td style="text-align:right; color:#64ffda; font-weight:600;">{prophet_r2:.4f}</td></tr>
                <tr><td>Seasonality</td><td style="text-align:right; color:#C0C0C0;">Multiplicative</td></tr>
                <tr><td>Changepoints</td><td style="text-align:right; color:#C0C0C0;">30</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with perf_col2:
        st.markdown("""
        <div class="metric-card" style="text-align:left;">
            <h3 style="color:#ff69b4; margin-top:0;">⚡ XGBoost</h3>
        """, unsafe_allow_html=True)
        st.markdown(f"""
            <table style="width:100%; color:#a8b2d1;">
                <tr><td>RMSE</td><td style="text-align:right; color:#64ffda; font-weight:600;">${xgb_rmse:.4f}</td></tr>
                <tr><td>MAE</td><td style="text-align:right; color:#64ffda; font-weight:600;">${xgb_mae:.4f}</td></tr>
                <tr><td>R² Score</td><td style="text-align:right; color:#64ffda; font-weight:600;">{xgb_r2:.4f}</td></tr>
                <tr><td>Estimators</td><td style="text-align:right; color:#C0C0C0;">500</td></tr>
                <tr><td>Learning Rate</td><td style="text-align:right; color:#C0C0C0;">0.03</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    # Validation chart
    st.markdown('<div class="section-header">📉 Validation: Actual vs Predicted</div>', unsafe_allow_html=True)

    fig_val = go.Figure()
    fig_val.add_trace(go.Scatter(
        x=xgb_val["Date"], y=xgb_val["Close"],
        name="Actual Price", line=dict(color="#C0C0C0", width=2),
    ))
    fig_val.add_trace(go.Scatter(
        x=xgb_val["Date"], y=xgb_val["Close"].values,  # actual for reference
        name="", showlegend=False, line=dict(color="rgba(0,0,0,0)"),
    ))
    fig_val.add_trace(go.Scatter(
        x=xgb_val["Date"], y=xgb_val_pred,
        name="XGBoost Predicted", line=dict(color="#ff69b4", width=2, dash="dot"),
    ))

    # Prophet validation
    if len(prophet_pred_val) > 0 and len(merged_val) > 0:
        fig_val.add_trace(go.Scatter(
            x=merged_val["ds"], y=merged_val["yhat"],
            name="Prophet Predicted", line=dict(color="#ffd700", width=2, dash="dash"),
        ))

    fig_val.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,26,0.8)",
        height=400,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(gridcolor="rgba(192,192,192,0.05)"),
        yaxis=dict(gridcolor="rgba(192,192,192,0.05)", title="Price (USD)"),
    )
    st.plotly_chart(fig_val, use_container_width=True)

    # Feature importance (XGBoost)
    st.markdown('<div class="section-header">🧬 XGBoost Feature Importance (Top 15)</div>', unsafe_allow_html=True)
    importance = pd.Series(xgb_model.feature_importances_, index=feature_cols)
    top_features = importance.nlargest(15).sort_values()

    fig_imp = go.Figure(data=[go.Bar(
        y=top_features.index,
        x=top_features.values,
        orientation="h",
        marker=dict(
            color=top_features.values,
            colorscale=[[0, "#1a1a2e"], [0.5, "#C0C0C0"], [1, "#64ffda"]],
        ),
    )])
    fig_imp.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,26,0.8)",
        height=400,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(title="Importance Score", gridcolor="rgba(192,192,192,0.05)"),
        yaxis=dict(gridcolor="rgba(192,192,192,0.05)"),
    )
    st.plotly_chart(fig_imp, use_container_width=True)


# ─── Tab 4: Data Explorer ───────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">📋 Silver Price Data Explorer</div>', unsafe_allow_html=True)

    col_stats1, col_stats2 = st.columns(2)

    with col_stats1:
        st.markdown("#### 📊 Descriptive Statistics")
        stats = df[["Close", "Open", "High", "Low", "Volume"]].describe()
        st.dataframe(stats.style.format("{:.2f}"), use_container_width=True)

    with col_stats2:
        st.markdown("#### 📈 Yearly Performance")
        yearly = df.set_index("Date")["Close"].resample("YE").agg(["first", "last", "max", "min"])
        yearly["Return %"] = ((yearly["last"] - yearly["first"]) / yearly["first"]) * 100
        yearly.index = yearly.index.year
        yearly.columns = ["Open", "Close", "High", "Low", "Return %"]
        st.dataframe(yearly.style.format("{:.2f}").map(
            lambda v: "color: #64ffda" if isinstance(v, (int, float)) and v > 0 else "color: #ff6b6b",
            subset=["Return %"],
        ), use_container_width=True)

    st.markdown("#### 🔍 Raw Data (Latest 100 Records)")
    display_df = df.tail(100).sort_values("Date", ascending=False).copy()
    display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(
        display_df.style.format({
            "Open": "${:.2f}", "High": "${:.2f}",
            "Low": "${:.2f}", "Close": "${:.2f}",
            "Volume": "{:,.0f}",
        }),
        use_container_width=True,
        height=400,
    )

    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Full Dataset (CSV)",
        data=csv,
        file_name=f"silver_prices_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ─── Footer ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <p>🪙 Silver Price Forecaster — Powered by Prophet & XGBoost | Data from Yahoo Finance</p>
    <p>⚠️ This is for educational purposes only. Not financial advice. Past performance does not guarantee future results.</p>
</div>
""", unsafe_allow_html=True)
