import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 페이지 기본 설정
st.set_page_config(page_title="서울 기온 예측기 - 모델 평가", layout="wide")
st.title("🌡️ 서울 연평균 기온 회귀 모델 비교 및 평가")

# 데이터 불러오기 함수
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"


@st.cache_data
def load_and_process_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year

    # 연도별 관측일수 및 평균기온 계산
    yearly_summary = (
        df.groupby("연도")
        .agg(관측일수=("평균기온", "count"), 평균기온=("평균기온", "mean"))
        .reset_index()
    )

    # 필터링: 2025년 이하 & 관측일수 300일 이상인 해만 추출
    filtered_df = yearly_summary[
        (yearly_summary["연도"] <= 2025) & (yearly_summary["관측일수"] >= 300)
    ].copy()

    # 독립변수: 1908년부터 지난 연수 (Year - 1908)
    filtered_df["지난연수"] = filtered_df["연도"] - 1908

    return filtered_df


df = load_and_process_data()

# ---------------------------------------------------------
# 데이터 분할 (테스트 데이터: 최근 20년 2006~2025)
# ---------------------------------------------------------
df_test = df[(df["연도"] >= 2006) & (df["연도"] <= 2025)].copy()
X_test = df_test["지난연수"].values
y_test = df_test["평균기온"].values

# 학습 데이터셋 3가지 준비
# 1. 전체 데이터 (2005년 이하 학습)
df_train_all = df[df["연도"] <= 2005].copy()
# 2. 최근 100년 (1906~2005)
df_train_100 = df[(df["연도"] >= 1906) & (df["연도"] <= 2005)].copy()
# 3. 최근 50년 (1956~2005)
df_train_50 = df[(df["연도"] >= 1956) & (df["연도"] <= 2005)].copy()


# 회귀 모델 학습 및 평가 함수
def fit_and_evaluate(df_train, X_test, y_test, name):
    X_train = df_train["지난연수"].values
    y_train = df_train["평균기온"].values

    # 선형 회귀 학습
    slope, intercept = np.polyfit(X_train, y_train, 1)

    # 테스트 데이터 예측
    y_pred = slope * X_test + intercept

    # 평가 지표 계산
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "Name": name,
        "Slope": slope,
        "Slope_100y": slope * 100,
        "Intercept": intercept,
        "Train_Count": len(df_train),
        "Start_Year": int(df_train["연도"].min()),
        "End_Year": int(df_train["연도"].max()),
        "MAE": mae,
        "MSE": mse,
        "R2": r2,
        "y_pred": y_pred,
    }


# 3가지 학습 데이터로 모델 생성 및 테스트 데이터 평가
res_all = fit_and_evaluate(df_train_all, X_test, y_test, "전체 (1908~2005)")
res_100 = fit_and_evaluate(df_train_100, X_test, y_test, "최근 100년 (1906~2005)")
res_50 = fit_and_evaluate(df_train_50, X_test, y_test, "최근 50년 (1956~2005)")

# ---------------------------------------------------------
# UI 구성: 1. 모델 성능 및 기울기 비교 표
# ---------------------------------------------------------
st.subheader("📊 최근 20년(2006~2025년) 테스트 데이터에 대한 모델 예측 성능 평가")

metrics_df = pd.DataFrame(
    [
        {
            "학습 기간": res_all["Name"],
            "학습 데이터 수": f"{res_all['Train_Count']}개",
            "기울기 (100년당 상승량)": f"+{res_all['Slope_100y']:.3f} °C",
            "MAE (평균 절대 오차)": f"{res_all['MAE']:.4f}",
            "MSE (평균 제곱 오차)": f"{res_all['MSE']:.4f}",
            "R² (결정계수)": f"{res_all['R2']:.4f}",
        },
        {
            "학습 기간": res_100["Name"],
            "학습 데이터 수": f"{res_100['Train_Count']}개",
            "기울기 (100년당 상승량)": f"+{res_100['Slope_100y']:.3f} °C",
            "MAE (평균 절대 오차)": f"{res_100['MAE']:.4f}",
            "MSE (평균 제곱 오차)": f"{res_100['MSE']:.4f}",
            "R² (결정계수)": f"{res_100['R2']:.4f}",
        },
        {
            "학습 기간": res_50["Name"],
            "학습 데이터 수": f"{res_50['Train_Count']}개",
            "기울기 (100년당 상승량)": f"+{res_50['Slope_100y']:.3f} °C",
            "MAE (평균 절대 오차)": f"{res_50['MAE']:.4f}",
            "MSE (평균 제곱 오차)": f"{res_50['MSE']:.4f}",
            "R² (결정계수)": f"{res_50['R2']:.4f}",
        },
    ]
)

st.dataframe(metrics_df, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# UI 구성: 2. 주요 변화 핵심 분석 카드
# ---------------------------------------------------------
st.subheader("🔍 학습 기간에 따른 기울기 및 예측 성능 비교")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
        <div style="background-color: #f0f4f8; padding: 18px; border-radius: 10px;">
            <h4 style="margin:0; color:#1f77b4;">전체 (1908~2005)</h4>
            <p style="margin:5px 0;"><b>100년당 상승 폭:</b> +{res_all['Slope_100y']:.2f} °C</p>
            <p style="margin:5px 0;"><b>MAE:</b> {res_all['MAE']:.3f} | <b>R²:</b> {res_all['R2']:.3f}</p>
            <small style="color:#666;">과거 전체 데이터를 반영하여 완만한 상승률을 보임</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div style="background-color: #f0f7f0; padding: 18px; border-radius: 10px;">
            <h4 style="margin:0; color:#2ca02c;">최근 100년 (1906~2005)</h4>
            <p style="margin:5px 0;"><b>100년당 상승 폭:</b> +{res_100['Slope_100y']:.2f} °C</p>
            <p style="margin:5px 0;"><b>MAE:</b> {res_100['MAE']:.3f} | <b>R²:</b> {res_100['R2']:.3f}</p>
            <small style="color:#666;">전체 기간과 비슷하나 관측 누락 구간 제외로 안정적</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div style="background-color: #fff5f0; padding: 18px; border-radius: 10px;">
            <h4 style="margin:0; color:#d62728;">최근 50년 (1956~2005)</h4>
            <p style="margin:5px 0;"><b>100년당 상승 폭:</b> +{res_50['Slope_100y']:.2f} °C</p>
            <p style="margin:5px 0;"><b>MAE:</b> {res_50['MAE']:.3f} | <b>R²:</b> {res_50['R2']:.3f}</p>
            <small style="color:#666;">기울기가 가파르며, 최근 20년 실제 기온에 가장 가깝게 예측 (MAE 낮음)</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# ---------------------------------------------------------
# UI 구성: 3. 예측 슬라이더 및 시각화
# ---------------------------------------------------------
selected_year = st.slider(
    "예측하고 싶은 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2026,
    step=1,
)

# 선택한 연도의 각 모델별 예측값
pred_val_all = res_all["Slope"] * (selected_year - 1908) + res_all["Intercept"]
pred_val_100 = res_100["Slope"] * (selected_year - 1908) + res_100["Intercept"]
pred_val_50 = res_50["Slope"] * (selected_year - 1908) + res_50["Intercept"]

p_col1, p_col2, p_col3 = st.columns(3)
p_col1.metric("전체 모델 예측값", f"{pred_val_all:.2f} °C")
p_col2.metric("최근 100년 모델 예측값", f"{pred_val_100:.2f} °C")
p_col3.metric("최근 50년 모델 예측값", f"{pred_val_50:.2f} °C")

# Plotly 그래프 시각화
fig = go.Figure()

# 1. 학습 데이터 (1908~2005)
fig.add_trace(
    go.Scatter(
        x=df[df["연도"] <= 2005]["연도"],
        y=df[df["연도"] <= 2005]["평균기온"],
        mode="markers",
        name="학습 데이터 (1908~2005)",
        marker=dict(color="gray", size=6, opacity=0.5),
    )
)

# 2. 테스트 데이터 (2006~2025)
fig.add_trace(
    go.Scatter(
        x=df_test["연도"],
        y=df_test["평균기온"],
        mode="markers",
        name="테스트 데이터 (2006~2025)",
        marker=dict(color="black", size=8, symbol="diamond"),
    )
)

# X축 범위 설정 (1900~2100)
years_range = np.arange(1900, 2101)

# 회귀선 그리기
fig.add_trace(
    go.Scatter(
        x=years_range,
        y=res_all["Slope"] * (years_range - 1908) + res_all["Intercept"],
        mode="lines",
        name=f"전체 학습 (+{res_all['Slope_100y']:.2f}°C/100년)",
        line=dict(color="#1f77b4", width=2),
    )
)

fig.add_trace(
    go.Scatter(
        x=years_range,
        y=res_100["Slope"] * (years_range - 1908) + res_100["Intercept"],
        mode="lines",
        name=f"최근 100년 학습 (+{res_100['Slope_100y']:.2f}°C/100년)",
        line=dict(color="#2ca02c", width=2, dash="dash"),
    )
)

fig.add_trace(
    go.Scatter(
        x=years_range,
        y=res_50["Slope"] * (years_range - 1908) + res_50["Intercept"],
        mode="lines",
        name=f"최근 50년 학습 (+{res_50['Slope_100y']:.2f}°C/100년)",
        line=dict(color="#d62728", width=2.5),
    )
)

# 선택 연도 예측점 표시
fig.add_trace(
    go.Scatter(
        x=[selected_year, selected_year, selected_year],
        y=[pred_val_all, pred_val_100, pred_val_50],
        mode="markers",
        name="선택 연도 예측점",
        marker=dict(
            color=["#1f77b4", "#2ca02c", "#d62728"], size=10, symbol="star"
        ),
    )
)

fig.update_layout(
    title="학습 기간별 회귀선 비교 및 최근 20년 테스트 데이터 예측 검증",
    xaxis_title="연도",
    yaxis_title="평균 기온 (°C)",
    hovermode="x unified",
    template="plotly_white",
    xaxis=dict(range=[1895, 2105]),
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
)

st.plotly_chart(fig, use_container_width=True)
