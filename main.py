import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 페이지 기본 설정
st.set_page_config(page_title="서울 기온 다항회귀 분석기", layout="wide")
st.title("🌡️ 서울 연평균 기온 다항 회귀(1차, 3차, 9차 곡선) 분석 및 예측")

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

    return filtered_df


df = load_and_process_data()

# ---------------------------------------------------------
# 데이터 분할 (훈련용: ~2005년, 테스트용: 2006년~)
# ---------------------------------------------------------
df_train = df[df["연도"] <= 2005].copy()
df_test = df[df["연도"] >= 2006].copy()

# 데이터 개수 정보 출력
col1, col2, col3, col4 = st.columns(4)
col1.metric("훈련용 데이터 수 (~2005)", f"{len(df_train)}개 해")
col2.metric(
    "훈련 기간",
    f"{int(df_train['연도'].min())}년 ~ {int(df_train['연도'].max())}년",
)
col3.metric("테스트용 데이터 수 (2006~)", f"{len(df_test)}개 해")
col4.metric(
    "테스트 기간",
    f"{int(df_test['연도'].min())}년 ~ {int(df_test['연도'].max())}년",
)

st.markdown("---")

# ---------------------------------------------------------
# 고차 다항식 오버플로우 방지를 위한 스케일링 (StandardScaler)
# ---------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(df_train[["연도"]].values).flatten()
X_test_scaled = scaler.transform(df_test[["연도"]].values).flatten()

y_train = df_train["평균기온"].values
y_test = df_test["평균기온"].values

# 2050년 예측을 위한 스케일링 변환
year_2050_scaled = scaler.transform([[2050]])[0, 0]

# ---------------------------------------------------------
# 다항 회귀 모델 학습 및 테스트 데이터 평가 (1차, 3차, 9차)
# ---------------------------------------------------------
degrees = [1, 3, 9]
results = []
models = {}

for deg in degrees:
    # 스케일링된 X 데이터로 다항식 피팅
    coeffs = np.polyfit(X_train_scaled, y_train, deg)
    poly_func = np.poly1d(coeffs)
    models[deg] = poly_func

    # 학습에 사용되지 않은 테스트 데이터로 평가
    y_pred_test = poly_func(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred_test)

    # 2050년 예측값 계산
    pred_2050 = poly_func(year_2050_scaled)

    results.append(
        {
            "차수": f"{deg}차 {"곡선" if deg > 1 else "직선"}",
            "테스트 평균 오차 (MAE)": f"±{mae:.3f} °C",
            "2050년 예상 기온": f"{pred_2050:.2f} °C",
            "MAE_num": mae,
            "pred_2050_num": pred_2050,
        }
    )

# ---------------------------------------------------------
# UI 구성: 1. 평가 및 예측 결과 표
# ---------------------------------------------------------
st.subheader("📋 훈련에 사용되지 않은 테스트 데이터(2006~2025) 채점 결과")

res_df = pd.DataFrame(results)[
    ["차수", "테스트 평균 오차 (MAE)", "2050년 예상 기온"]
]
st.dataframe(res_df, use_container_width=True)

st.info(
    "💡 **참고:** 고차 다항식(9차)일수록 훈련 데이터에는 지나치게 맞춰지지만(과적합), "
    "학습에 포함되지 않은 미래 구간(2050년 등) 예측 시 폭발적으로 발산하는 경향을 볼 수 있습니다."
)

st.markdown("---")

# ---------------------------------------------------------
# UI 구성: 2. 예측 슬라이더 및 시각화
# ---------------------------------------------------------
selected_year = st.slider(
    "예측하고 싶은 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2050,
    step=1,
)

selected_scaled = scaler.transform([[selected_year]])[0, 0]

p_col1, p_col2, p_col3 = st.columns(3)
p_col1.metric("1차 모델 예측값", f"{models[1](selected_scaled):.2f} °C")
p_col2.metric("3차 모델 예측값", f"{models[3](selected_scaled):.2f} °C")
p_col3.metric("9차 모델 예측값", f"{models[9](selected_scaled):.2f} °C")

# Plotly 시각화
fig = go.Figure()

# 1. 훈련 데이터
fig.add_trace(
    go.Scatter(
        x=df_train["연도"],
        y=df_train["평균기온"],
        mode="markers",
        name="훈련용 데이터 (~2005)",
        marker=dict(color="blue", size=6, opacity=0.6),
    )
)

# 2. 테스트 데이터
fig.add_trace(
    go.Scatter(
        x=df_test["연도"],
        y=df_test["평균기온"],
        mode="markers",
        name="테스트용 데이터 (2006~2025)",
        marker=dict(color="red", size=8, symbol="diamond"),
    )
)

# 곡선을 부드럽게 표현하기 위한 X축 범위 생성 (1900~2100)
years_dense = np.linspace(1900, 2100, 400)
years_dense_scaled = scaler.transform(years_dense.reshape(-1, 1)).flatten()

# 각 차수별 곡선 그리기
colors = {1: "green", 3: "orange", 9: "purple"}
dashes = {1: "solid", 3: "dash", 9: "dot"}

for deg in degrees:
    y_curve = models[deg](years_dense_scaled)
    fig.add_trace(
        go.Scatter(
            x=years_dense,
            y=y_curve,
            mode="lines",
            name=f"{deg}차 회귀 곡선",
            line=dict(color=colors[deg], width=2.5, dash=dashes[deg]),
        )
    )

# 선택 연도 예측 점 표시
for deg in degrees:
    pred_val = models[deg](selected_scaled)
    fig.add_trace(
        go.Scatter(
            x=[selected_year],
            y=[pred_val],
            mode="markers",
            name=f"{selected_year}년 {deg}차 예측점",
            marker=dict(color=colors[deg], size=10, symbol="star"),
            showlegend=False,
        )
    )

fig.update_layout(
    title="1차, 3차, 9차 다항 회귀 곡선 비교 및 테스트 데이터 검증",
    xaxis_title="연도",
    yaxis_title="평균 기온 (°C)",
    hovermode="x unified",
    template="plotly_white",
    xaxis=dict(range=[1895, 2105]),
    yaxis=dict(range=[df["평균기온"].min() - 3, df["평균기온"].max() + 10]),
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
)

st.plotly_chart(fig, use_container_width=True)
