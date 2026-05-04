#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ============================================================
# KOSPI200 변동성 체제 분석 및 동태적 헤지 전략
# KOSPI200 Volatility Regime Analysis & Dynamic Hedging
#
# 분석 기간: 2007.01 ~ 2025.12
# 작성일: 2026.04
# 도구: Python 3.x
# 라이브러리: yfinance, pandas, numpy, statsmodels,
#             ruptures, arch, matplotlib, scipy
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
from scipy import stats

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 0. 데이터 수집
# ============================================================

import yfinance as yf

# KOSPI200 일별 데이터
kospi200 = yf.download("^KS200", start="2007-01-02", end="2025-12-31")
kospi = kospi200[['Close']].copy()
kospi.columns = ['KOSPI200']
kospi.index.name = 'Date'

# VIX (FRED에서 직접 다운로드 후 vix_raw.csv로 저장 필요)
# https://fred.stlouisfed.org/series/VIXCLS
vix = pd.read_csv("vix_raw.csv", parse_dates=['observation_date'],
                  index_col='observation_date')
vix.columns = ['VIX']
vix = vix.replace('.', np.nan).astype(float)

# 연방기금금리 (FRED에서 직접 다운로드 후 ffr_raw.csv로 저장 필요)
# https://fred.stlouisfed.org/series/DFF
ffr = pd.read_csv("ffr_raw.csv", parse_dates=['observation_date'],
                  index_col='observation_date')
ffr.columns = ['FFR']
ffr = ffr.replace('.', np.nan).astype(float)

# 병합 (KOSPI200 영업일 기준)
df = kospi.copy()
df = df.join(vix, how='left')
df = df.join(ffr, how='left')
df['VIX'] = df['VIX'].ffill()
df['FFR'] = df['FFR'].ffill()
df['ret'] = np.log(df['KOSPI200'] / df['KOSPI200'].shift(1))
df = df.dropna()

df.to_csv("merged_data.csv", encoding="utf-8-sig")
print(f"데이터 수집 완료: {df.index[0].date()} ~ {df.index[-1].date()}, {len(df)}행")


# ============================================================
# 1. 기초 통계 및 시각화
# ============================================================

print("\n=== 기술통계 ===")
print(df.describe().round(4))
print("\n=== 왜도 ===")
print(df.skew().round(4))
print("\n=== 첨도 ===")
print(df.kurt().round(4))

# 4개 변수 시계열
fig, axes = plt.subplots(4, 1, figsize=(14, 12))
fig.suptitle("Data Overview: 2007.01 ~ 2025.12", fontsize=13)

axes[0].plot(df.index, df['KOSPI200'], color='steelblue', linewidth=0.8)
axes[0].set_title("KOSPI200 Index")
axes[0].set_ylabel("Index")

axes[1].plot(df.index, df['ret'], color='gray', linewidth=0.5)
axes[1].set_title("KOSPI200 Log Return")
axes[1].set_ylabel("Return")
axes[1].axhline(0, color='black', linewidth=0.5)

axes[2].plot(df.index, df['VIX'], color='crimson', linewidth=0.8)
axes[2].set_title("VIX")
axes[2].set_ylabel("VIX")

axes[3].plot(df.index, df['FFR'], color='darkorange', linewidth=0.8)
axes[3].set_title("Federal Funds Rate")
axes[3].set_ylabel("%")

for ax in axes:
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("data_overview.png", dpi=150, bbox_inches='tight')
plt.show()

# 수익률 분포
df['RV'] = df['ret'].rolling(22).std() * np.sqrt(252)
df_clean = df.dropna()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("KOSPI200 일별 로그수익률 분포 특성", fontsize=13)

ret = df_clean['ret']
x = np.linspace(ret.min(), ret.max(), 300)
mu, std = ret.mean(), ret.std()

axes[0].hist(ret, bins=100, density=True, color='steelblue', alpha=0.6, label='실제 분포')
axes[0].plot(x, stats.norm.pdf(x, mu, std), color='crimson',
             linewidth=2, label=f'정규분포 (μ={mu:.4f}, σ={std:.4f})')
axes[0].set_title("수익률 분포 vs 정규분포")
axes[0].set_xlabel("로그수익률")
axes[0].set_ylabel("밀도")
axes[0].legend()
axes[0].grid(alpha=0.3)

textstr = f'평균: {mu:.4f}\n표준편차: {std:.4f}\n왜도: {ret.skew():.4f}\n첨도: {ret.kurt():.4f}'
axes[0].text(0.97, 0.97, textstr, transform=axes[0].transAxes,
             fontsize=9, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

(osm, osr), (slope, intercept, r) = stats.probplot(ret, dist="norm")
axes[1].scatter(osm, osr, color='steelblue', alpha=0.3, s=3, label='관측값')
axes[1].plot(osm, slope * np.array(osm) + intercept,
             color='crimson', linewidth=2, label='정규분포 기준선')
axes[1].set_title("Q-Q Plot (정규분포 대비)")
axes[1].set_xlabel("이론적 분위수")
axes[1].set_ylabel("표본 분위수")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("return_distribution.png", dpi=150, bbox_inches='tight')
plt.show()


# ============================================================
# 2. 단계 1: 그랜저 인과성 검정
# ============================================================

from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR

df_analysis = df_clean.copy()
df_analysis['ΔFFR'] = df_analysis['FFR'].diff()
df_analysis = df_analysis.dropna()

# ADF 단위근 검정
variables = {
    'KOSPI200 실현변동성 (RV)': df_analysis['RV'],
    'VIX': df_analysis['VIX'],
    'FFR': df_analysis['FFR'],
    'FFR 변화량 (ΔFFR)': df_analysis['ΔFFR'],
}

print("\n=== ADF 단위근 검정 결과 ===")
print(f"{'변수':<30} {'ADF 통계량':>12} {'p-value':>10} {'판정':>10}")
print("-" * 65)
for name, series in variables.items():
    series = series.dropna()
    result = adfuller(series, autolag='AIC')
    판정 = "정상 ✅" if result[1] < 0.05 else "단위근 ⚠️"
    print(f"{name:<30} {result[0]:>12.4f} {result[1]:>10.4f} {판정:>10}")

# VAR 최적 시차 선택
var_data = df_analysis[['RV', 'VIX', 'ΔFFR']].dropna()
model = VAR(var_data)
lag_result = model.select_order(maxlags=10)
print("\n=== VAR 최적 시차 선택 ===")
print(lag_result.summary())

# 그랜저 인과성 검정 (BIC 기준 lag=4)
print("\n=== 그랜저 인과성 검정 (lag=4, BIC 기준) ===")

print("\n── VIX → RV ──")
result1 = grangercausalitytests(df_analysis[['RV', 'VIX']], maxlag=4, verbose=False)
for lag in range(1, 5):
    f_stat = result1[lag][0]['ssr_ftest'][0]
    p_val = result1[lag][0]['ssr_ftest'][1]
    print(f"  lag={lag}: F={f_stat:.4f}, p={p_val:.4f}", "✅" if p_val < 0.05 else "")

print("\n── ΔFFR → RV ──")
result2 = grangercausalitytests(df_analysis[['RV', 'ΔFFR']], maxlag=4, verbose=False)
for lag in range(1, 5):
    f_stat = result2[lag][0]['ssr_ftest'][0]
    p_val = result2[lag][0]['ssr_ftest'][1]
    print(f"  lag={lag}: F={f_stat:.4f}, p={p_val:.4f}", "✅" if p_val < 0.05 else "")

print("\n── RV → VIX (역방향) ──")
result3 = grangercausalitytests(df_analysis[['VIX', 'RV']], maxlag=4, verbose=False)
for lag in range(1, 5):
    f_stat = result3[lag][0]['ssr_ftest'][0]
    p_val = result3[lag][0]['ssr_ftest'][1]
    print(f"  lag={lag}: F={f_stat:.4f}, p={p_val:.4f}", "✅" if p_val < 0.05 else "")


# ============================================================
# 3. 단계 2: PELT 변화점 탐지
# ============================================================

import ruptures as rpt

rv = df_analysis['RV'].values.reshape(-1, 1)
dates = df_analysis.index

# BIC 기준 (주 분석)
algo = rpt.Pelt(model="l2").fit(rv)
bps_main = algo.predict(pen=3.0)
bp_dates_main = [dates[i-1] for i in bps_main[:-1]]

# AIC 기준 (강건성 검증)
bps_robust = algo.predict(pen=2.0)
bp_dates_robust = [dates[i-1] for i in bps_robust[:-1]]

print("\n=== BIC 기준 변화점 (pen=3.0) ===")
for i, d in enumerate(bp_dates_main):
    print(f"  변화점 {i+1}: {d.strftime('%Y-%m-%d')}")

print("\n=== AIC 기준 변화점 (pen=2.0) ===")
for i, d in enumerate(bp_dates_robust):
    print(f"  변화점 {i+1}: {d.strftime('%Y-%m-%d')}")

# PELT 시각화
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
for ax, bp_dates, title in zip(
    axes,
    [bp_dates_main, bp_dates_robust],
    ["PELT Breakpoints - Main (pen=3.0)",
     "PELT Breakpoints - Robust (pen=2.0)"]
):
    ax.plot(dates, df_analysis['RV'], color='steelblue', linewidth=0.8)
    for d in bp_dates:
        ax.axvline(d, color='crimson', linewidth=1.2, linestyle='--', alpha=0.8)
    ax.set_title(title)
    ax.set_ylabel("Annualized Volatility")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("pelt_results.png", dpi=150, bbox_inches='tight')
plt.show()


# ============================================================
# 4. 단계 3: 체제별 GARCH/EGARCH 추정
# ============================================================

from arch import arch_model

regimes = {
    'Regime 1': ('2007-02-01', '2008-09-18'),
    'Regime 2': ('2008-09-19', '2009-02-27'),
    'Regime 3': ('2009-02-28', '2012-10-05'),
    'Regime 4': ('2012-10-06', '2020-03-17'),
    'Regime 5': ('2020-03-18', '2020-04-22'),
    'Regime 6': ('2020-04-23', '2025-12-30'),
}

garch_results = {}

print("\n=== 체제별 GARCH/EGARCH 추정 결과 ===")
print("=" * 75)

for regime, (start, end) in regimes.items():
    ret = df_analysis.loc[start:end, 'ret'] * 100
    n = len(ret)

    garch = arch_model(ret, vol='Garch', p=1, q=1, dist='normal')
    garch_fit = garch.fit(disp='off')

    egarch = arch_model(ret, vol='EGarch', p=1, o=1, q=1, dist='normal')
    egarch_fit = egarch.fit(disp='off')

    g_params = garch_fit.params
    e_params = egarch_fit.params
    persistence = g_params['alpha[1]'] + g_params['beta[1]']
    leverage = e_params['gamma[1]']

    garch_results[regime] = {
        'n': n,
        'alpha': g_params['alpha[1]'],
        'beta': g_params['beta[1]'],
        'persistence': persistence,
        'gamma': leverage,
        'GARCH_AIC': garch_fit.aic,
        'EGARCH_AIC': egarch_fit.aic,
    }

    print(f"\n{regime} ({start} ~ {end}, n={n})")
    print(f"  GARCH(1,1): α={g_params['alpha[1]']:.4f}, β={g_params['beta[1]']:.4f}, 지속성={persistence:.4f}")
    print(f"  EGARCH(1,1): γ(레버리지)={leverage:.4f}")
    print(f"  AIC — GARCH: {garch_fit.aic:.2f} / EGARCH: {egarch_fit.aic:.2f}")

print("\n" + "=" * 75)


# ============================================================
# 5. 단계 4: 체제별 최소분산 헤지비율 및 HE
# ============================================================

# KOSPI200 선물 데이터 (Investing.com에서 다운로드 후 저장 필요)
# 파일명: kospi200_futures_raw.csv
futures = pd.read_csv("kospi200_futures_raw.csv")
futures['날짜'] = futures['날짜'].str.replace(' ', '')
futures['날짜'] = pd.to_datetime(futures['날짜'], format='%Y-%m-%d')
futures = futures.set_index('날짜').sort_index()
futures['종가'] = futures['종가'].astype(str).str.replace(',', '').astype(float)
futures['ret_f'] = np.log(futures['종가'] / futures['종가'].shift(1))
futures = futures.dropna(subset=['ret_f'])

df_hedge = df_analysis.join(futures[['ret_f']], how='left')

regimes_hedge = {
    'Regime 4': ('2013-08-07', '2020-03-17'),
    'Regime 5': ('2020-03-18', '2020-04-22'),
    'Regime 6': ('2020-04-23', '2025-12-30'),
}

print("\n=== 체제별 현물-선물 상관계수 ===")
for regime, (start, end) in regimes_hedge.items():
    subset = df_hedge.loc[start:end].dropna(subset=['ret', 'ret_f'])
    corr = np.corrcoef(subset['ret'], subset['ret_f'])[0, 1]
    print(f"  {regime}: {corr:.4f} (n={len(subset)})")

print(f"\n{'체제':<12} {'n':>6} {'헤지비율(h*)':>14} {'HE (%)':>10} {'현물분산':>12} {'헤지분산':>12}")
print("-" * 70)

he_results = {}

for regime, (start, end) in regimes_hedge.items():
    subset = df_hedge.loc[start:end].dropna(subset=['ret', 'ret_f'])
    ret_s = subset['ret']
    ret_f = subset['ret_f']
    n = len(ret_s)

    ret_s_pct = ret_s * 100
    ret_f_pct = ret_f * 100

    garch_s = arch_model(ret_s_pct, vol='Garch', p=1, q=1, dist='normal')
    fit_s = garch_s.fit(disp='off')
    var_s = fit_s.conditional_volatility ** 2

    garch_f = arch_model(ret_f_pct, vol='Garch', p=1, q=1, dist='normal')
    fit_f = garch_f.fit(disp='off')
    var_f = fit_f.conditional_volatility ** 2

    corr = np.corrcoef(ret_s_pct, ret_f_pct)[0, 1]
    cov_sf = corr * np.sqrt(var_s) * np.sqrt(var_f)
    h_star = np.mean(cov_sf / var_f)

    ret_hedged = ret_s - h_star * ret_f
    var_unhedged = np.var(ret_s)
    var_hedged = np.var(ret_hedged)
    HE = 1 - var_hedged / var_unhedged

    he_results[regime] = {
        'n': n, 'h_star': h_star, 'HE': HE,
        'var_unhedged': var_unhedged, 'var_hedged': var_hedged,
    }

    print(f"{regime:<12} {n:>6} {h_star:>14.4f} {HE*100:>10.2f} {var_unhedged:>12.6f} {var_hedged:>12.6f}")

# 단일 모형
print("\n=== 단일 모형 (2013-08-07 ~ 2025-12-30) ===")
subset_all = df_hedge.loc['2013-08-07':'2025-12-30'].dropna(subset=['ret', 'ret_f'])
ret_s_all = subset_all['ret'] * 100
ret_f_all = subset_all['ret_f'] * 100

garch_s_all = arch_model(ret_s_all, vol='Garch', p=1, q=1, dist='normal')
fit_s_all = garch_s_all.fit(disp='off')
var_s_all = fit_s_all.conditional_volatility ** 2

garch_f_all = arch_model(ret_f_all, vol='Garch', p=1, q=1, dist='normal')
fit_f_all = garch_f_all.fit(disp='off')
var_f_all = fit_f_all.conditional_volatility ** 2

corr_all = np.corrcoef(ret_s_all, ret_f_all)[0, 1]
cov_all = corr_all * np.sqrt(var_s_all) * np.sqrt(var_f_all)
h_star_all = np.mean(cov_all / var_f_all)

ret_hedged_all = subset_all['ret'] - h_star_all * subset_all['ret_f']
HE_all = 1 - np.var(ret_hedged_all) / np.var(subset_all['ret'])

print(f"단일 헤지비율: {h_star_all:.4f}")
print(f"단일 HE: {HE_all*100:.2f}%")


# ============================================================
# 6. 종합 시각화
# ============================================================

colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#937860']
bp_dates = [pd.Timestamp(d) for d in [
    '2008-09-18', '2009-02-27', '2012-10-05', '2020-03-17', '2020-04-22'
]]
starts = [pd.Timestamp('2007-02-01')] + bp_dates
ends = bp_dates + [pd.Timestamp('2025-12-30')]

persistence = [v['persistence'] for v in garch_results.values()]
leverage = [v['gamma'] for v in garch_results.values()]
he_values = [92.30, 97.83, 94.86]
he_single = HE_all * 100

fig = plt.figure(figsize=(16, 22))
fig.suptitle("KOSPI200 Volatility Regime Analysis", fontsize=14, y=0.995)
fig.subplots_adjust(top=0.97, hspace=0.35)

# 패널 1: PELT 변화점
ax1 = fig.add_subplot(4, 1, 1)
ax1.plot(df_analysis.index, df_analysis['RV'], color='steelblue', linewidth=0.8, zorder=3)
for i, (s, e) in enumerate(zip(starts, ends)):
    ax1.axvspan(s, e, alpha=0.15, color=colors[i])
    mid = s + (e - s) / 2
    ax1.text(mid, 0.92, f'R{i+1}', ha='center', fontsize=8,
             color=colors[i], transform=ax1.get_xaxis_transform())
for d in bp_dates:
    ax1.axvline(d, color='crimson', linewidth=1.2, linestyle='--', alpha=0.8)
ax1.set_title("KOSPI200 실현변동성 및 PELT 변화점 (BIC, pen=3.0)", pad=10)
ax1.set_ylabel("연율화 변동성")
ax1.xaxis.set_major_locator(mdates.YearLocator(2))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.grid(alpha=0.3)

# 패널 2: 변동성 지속성
ax2 = fig.add_subplot(4, 1, 2)
bars2 = ax2.bar(range(1, 7), persistence, color=colors, alpha=0.85, edgecolor='white')
ax2.axhline(y=np.mean(persistence), color='black', linewidth=1.0,
            linestyle='--', label=f'전체 평균: {np.mean(persistence):.4f}')
ax2.set_xticks(range(1, 7))
ax2.set_xticklabels([f'Regime {i}' for i in range(1, 7)])
ax2.set_ylim(0.85, 1.01)
ax2.set_title("체제별 변동성 지속성 (α+β)", pad=10)
ax2.set_ylabel("α + β")
ax2.legend()
ax2.grid(alpha=0.3, axis='y')
for bar, val in zip(bars2, persistence):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.001,
             f'{val:.4f}', ha='center', va='bottom', fontsize=9)

# 패널 3: 레버리지 효과
ax3 = fig.add_subplot(4, 1, 3)
bars3 = ax3.bar(range(1, 7), leverage, color=colors, alpha=0.85, edgecolor='white')
ax3.axhline(y=0, color='black', linewidth=0.8)
ax3.axhline(y=np.mean(leverage), color='black', linewidth=1.0,
            linestyle='--', label=f'전체 평균: {np.mean(leverage):.4f}')
ax3.set_xticks(range(1, 7))
ax3.set_xticklabels([f'Regime {i}' for i in range(1, 7)])
ax3.set_title("체제별 레버리지 효과 (γ)", pad=10)
ax3.set_ylabel("γ")
ax3.legend()
ax3.grid(alpha=0.3, axis='y')
for bar, val in zip(bars3, leverage):
    ax3.text(bar.get_x() + bar.get_width()/2, val - 0.02,
             f'{val:.4f}', ha='center', va='top', fontsize=9)

# 패널 4: HE 비교
ax4 = fig.add_subplot(4, 1, 4)
he_colors = [colors[3], colors[4], colors[5]]
bars4 = ax4.bar(range(3), he_values, color=he_colors, alpha=0.85,
                edgecolor='white', label='체제별 HE')
ax4.axhline(y=he_single, color='crimson', linewidth=1.5,
            linestyle='--', label=f'단일 모형 HE: {he_single:.2f}%')
ax4.set_xticks(range(3))
ax4.set_xticklabels(['Regime 4', 'Regime 5', 'Regime 6'])
ax4.set_ylim(88, 100)
ax4.set_title("체제별 헤지 효율성 (HE) vs 단일 모형", pad=10)
ax4.set_ylabel("HE (%)")
ax4.legend()
ax4.grid(alpha=0.3, axis='y')
for bar, val in zip(bars4, he_values):
    ax4.text(bar.get_x() + bar.get_width()/2, val + 0.1,
             f'{val:.2f}%', ha='center', va='bottom', fontsize=9)

plt.savefig("regime_analysis.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n모든 분석 완료.")

