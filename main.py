import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# 페이지 기본 설정
st.set_page_config(page_title="서울 기온 예측기", layout="wide")
st.title("🌡️ 서울 연평균 기온 및 온난화 추세 분석기")

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
# 회귀선 계산
# ---------------------------------------------------------
# 1. 전체 기간 회귀 모델 (1908년 기준 지난 연수)
X_all = df["지난연수"].values
Y_all = df["평균기온"].values
slope_all, intercept_all = np.polyfit(X_all, Y_all, 1)

# 2. 최근 20년 회귀 모델
max_year = int(df["연도"].max())
df_recent20 = df[df["연도"] >= (max_year - 19)].copy()
X_recent = df_recent20["지난연수"].values
Y_recent = df_recent20["평균기온"].values
slope_recent, intercept_recent = np.polyfit(X_recent, Y_recent, 1)

# 상관계수 (전체 기간)
corr_all = np.corrcoef(df["연도"], Y_all)[0, 1]

# 100년당 기온 상승량 (slope * 100)
slope_100y_all = slope_all * 100
slope_100y_recent = slope_recent * 100

# ---------------------------------------------------------
# UI 구성: 1. 데이터 개요 및 100년당 상승 폭
# ---------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("분석 대상 연도 수", f"{len(df)}개 해")
col2.metric("시작 연도", f"{int(df['연도'].min())}년")
col3.metric("끝 연도", f"{int(df['연도'].max())}년")
col4.metric("상관계수 (전체 기간)", f"{corr_all:.4f}")

st.markdown("---")

# 100년당 기온 상승량 크게 표시
st.subheader("📈 서울 기온 상승 속도 (100년당 상승량)")
m_col1, m_col2 = st.columns(2)

with m_col1:
    st.markdown(
        """
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;">
            <h4 style="margin:0; color: #333;">전체 기간 기준 (1908년~2025년)</h4>
            <h1 style="color: #1f77b4; margin: 10px 0;">+{:0.2f} °C / 100년</h1>
            <p style="margin:0; color: #666;">연간 상승 폭: 약 {:0.4f} °C/년</p>
        </div>
        """.format(slope_100y_all, slope_all),
        unsafe_allow_html=True,
    )

with m_col2:
    delta_diff = slope_100y_recent - slope_100y_all
    st.markdown(
        """
        <div style="background-color: #fff0f0; padding: 20px; border-radius: 10px; text-align: center;">
            <h4 style="margin:0; color: #333;">최근 20년 기준 ({0}년~{1}년)</h4>
            <h1 style="color: #d62728; margin: 10px 0;">+{2:0.2f} °C / 100년</h1>
            <p style="margin:0; color: #666;">연간 상승 폭: 약 {3:0.4f} °C/년 (전체 대비 +{4:0.2f}°C 격차)</p>
        </div>
        """.format(
            max_year - 19,
            max_year,
            slope_100y_recent,
            slope_recent,
            delta_diff,
        ),
        unsafe_allow_html=True,
    )

st.markdown("---")

# ---------------------------------------------------------
# UI 구성: 2. 예측 슬라이더 및 결과
# ---------------------------------------------------------
selected_year = st.slider(
    "예측하고 싶은 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2026,
    step=1,
)

# 예측값 계산 (전체 추세 기준)
predicted_temp_all = slope_all * (selected_year - 1908) + intercept_all
predicted_temp_recent = (
    slope_recent * (selected_year - 1908) + intercept_recent
)

p_col1, p_col2 = st.columns(2)

with p_col1:
    st.subheader(f"🔮 {selected_year}년 예상 기온 (전체 추세)")
    st.markdown(
        f"<h2 style='text-align: center; color: #1f77b4;'>{predicted_temp_all:.2f} °C</h2>",
        unsafe_allow_html=True,
    )

with p_col2:
    st.subheader(f"🔥 {selected_year}년 예상 기온 (최근 20년 추세)")
    st.markdown(
        f"<h2 style='text-align: center; color: #d62728;'>{predicted_temp_recent:.2f} °C</h2>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ---------------------------------------------------------
# UI 구성: 3. Plotly 시각화 (두 회귀선 비교)
# ---------------------------------------------------------
fig = go.Figure()

# 1. 실제 관측 데이터
fig.add_trace(
    go.Scatter(
        x=df["연도"],
        y=df["평균기온"],
        mode="markers",
        name="실제 연평균 기온",
        marker=dict(color="gray", size=6, opacity=0.6),
    )
)

# X축 전체 범위 지정
years_range = np.arange(1900, 2101)

# 2. 전체 기간 회귀선
trend_y_all = slope_all * (years_range - 1908) + intercept_all
fig.add_trace(
    go.Scatter(
        x=years_range,
        y=trend_y_all,
        mode="lines",
        name=f"전체 기간 추세 (+{slope_100y_all:.2f}°C/100년)",
        line=dict(color="#1f77b4", width=2.5),
    )
)

# 3. 최근 20년 회귀선
trend_y_recent = slope_recent * (years_range - 1908) + intercept_recent
fig.add_trace(
    go.Scatter(
        x=years_range,
        y=trend_y_recent,
        mode="lines",
        name=f"최근 20년 추세 (+{slope_100y_recent:.2f}°C/100년)",
        line=dict(color="#d62728", width=2.5, dash="dash"),
    )
)

# 4. 선택 연도 점 강조
fig.add_trace(
    go.Scatter(
        x=[selected_year, selected_year],
        y=[predicted_temp_all, predicted_temp_recent],
        mode="markers+text",
        name="선택 연도 예측점",
        marker=dict(color=["#1f77b4", "#d62728"], size=12, symbol="star"),
        text=[f"{predicted_temp_all:.2f}°C", f"{predicted_temp_recent:.2f}°C"],
        textposition="top center",
    )
)

# 레이아웃 설정
fig.update_layout(
    title="서울 연평균 기온 변화 및 추세선 비교",
    xaxis_title="연도",
    yaxis_title="평균 기온 (°C)",
    hovermode="x unified",
    template="plotly_white",
    xaxis=dict(range=[1895, 2105]),
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
)

st.plotly_chart(fig, use_container_width=True)
