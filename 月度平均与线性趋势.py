import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from matplotlib.dates import DateFormatter
from datetime import datetime
from scipy import stats

# 设置图像美观
plt.rcParams.update({
    'font.size': 11,
    'font.sans-serif': ['SimHei'],
    'axes.unicode_minus': False,
    'figure.dpi': 120
})

# 季节映射
SEASON_MAP = {
    1:'冬季', 2:'冬季', 12:'冬季',
    3:'春季', 4:'春季', 5:'春季',
    6:'夏季', 7:'夏季', 8:'夏季',
    9:'秋季', 10:'秋季', 11:'秋季'
}

# -----------------------
# 文件选择
# -----------------------
Tk().withdraw()
print("选择 输入 (潮汐校正后) 数据文件 (.pkl 或 .csv)")
infile = filedialog.askopenfilename(
    title="选择潮汐校正后的文件 (.pkl 或 .csv)",
    filetypes=[("Pickle/CSV","*.pkl *.csv")]
)
if not infile:
    raise SystemExit("未选择输入文件，退出")

print("选择 输出 文件夹")
outdir = filedialog.askdirectory(title="选择输出文件夹")
if not outdir:
    raise SystemExit("未选择输出文件夹，退出")
os.makedirs(outdir, exist_ok=True)


# -----------------------
# 读取数据
# -----------------------
if infile.lower().endswith('.pkl'):
    with open(infile, 'rb') as f:
        obj = pickle.load(f)
    dates = pd.to_datetime(obj['dates'])
    transects = {
        k: np.array(v['corrected'] if isinstance(v, dict) else v, dtype=float)
        for k,v in obj['transects'].items()
    }
else:
    df = pd.read_csv(infile, parse_dates=[0], index_col=0)
    dates = df.index
    transects = {c: df[c].values for c in df.columns}

print(f"载入 {len(dates)} 个日期，{len(transects)} 条 transects")

summary = []

for name, series in transects.items():
    print(f"处理 {name} ...")

    s_clean = pd.Series(series, index=dates)

    monthly = s_clean.resample('ME').mean()

    df_clean = pd.DataFrame({'val': s_clean})
    df_clean['month'] = df_clean.index.month
    df_clean['season'] = df_clean['month'].map(SEASON_MAP)

    fig, ax = plt.subplots(figsize=(9,4))
    ax.plot(dates, s_clean, 'o-', ms=3, lw=1, color='steelblue', label='原始数据')
    ax.plot(monthly.index, monthly.values, 's-', ms=5, lw=2, color='darkorange', label='月均')

    slope = np.nan
    r2 = np.nan
    monthly_dropna = monthly.dropna()
    if len(monthly_dropna) > 5:
        x = np.array([d.year + (d.month-1)/12 for d in monthly_dropna.index])
        y = monthly_dropna.values
        slope, intercept, r, p, stderr = stats.linregress(x, y)
        yfit = intercept + slope * x
        ax.plot(monthly_dropna.index, yfit, 'r--', lw=1.5,
                label=f"趋势: {slope:.2f} m/yr")
        r2 = r**2

    ax.set_ylabel("岸线位置 (m)")
    ax.set_title(f"{name} - 月度平均与趋势")
    ax.legend()
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{name}_monthly_trend.png"), dpi=200)
    plt.close()

    summary.append({
        "transect": name,
        "slope_m_per_year": slope,
        "R2": r2
    })

    df_clean['year'] = df_clean.index.year
    df_clean['season_year'] = df_clean['year'].astype(str) + "_" + df_clean['season']
    seasonal_ts = df_clean.groupby('season_year')['val'].mean()

    season_dates = []
    season_labels = []
    for key in seasonal_ts.index:
        year, season = key.split("_")
        year = int(year)
        if season == "冬季":
            dt = datetime(year, 1, 15)
        elif season == "春季":
            dt = datetime(year, 4, 15)
        elif season == "夏季":
            dt = datetime(year, 7, 15)
        else:
            dt = datetime(year, 10, 15)
        season_dates.append(dt)
        season_labels.append(season)

    seasonal_ts.index = pd.to_datetime(season_dates)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = {'冬季': '#4C72B0', '春季': '#55A868', '夏季': '#C44E52', '秋季': '#DD8452'}

    sc = ax.scatter(seasonal_ts.index, seasonal_ts.values,
                    c=[colors[s] for s in season_labels],
                    s=70, edgecolor='k', zorder=3)

    slope_s = np.nan
    r2_s = np.nan
    if len(seasonal_ts.dropna()) > 3:
        x = np.array([d.year + (d.month - 1) / 12 for d in seasonal_ts.index])
        y = seasonal_ts.values
        slope_s, intercept_s, r, p, stderr = stats.linregress(x, y)
        yfit = intercept_s + slope_s * x
        ax.plot(seasonal_ts.index, yfit, 'r--', lw=1.5,
                label=f"趋势: {slope_s:.2f} m/yr")
        r2_s = r ** 2

    handles = [
        plt.Line2D([], [], marker='o', color='w', markerfacecolor=colors['冬季'], markersize=8, label='冬季'),
        plt.Line2D([], [], marker='o', color='w', markerfacecolor=colors['春季'], markersize=8, label='春季'),
        plt.Line2D([], [], marker='o', color='w', markerfacecolor=colors['夏季'], markersize=8, label='夏季'),
        plt.Line2D([], [], marker='o', color='w', markerfacecolor=colors['秋季'], markersize=8, label='秋季')
    ]
    ax.legend(handles=handles, title="季节", loc="best")

    ax.set_ylabel("岸线位置 (m)")
    ax.set_title(f"{name} - 季节平均与趋势")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{name}_seasonal_trend.png"), dpi=200)
    plt.close()


df_summary = pd.DataFrame(summary)
df_summary.to_csv(os.path.join(outdir, "trend_summary.csv"), index=False, encoding="utf-8-sig")

print("\n全部完成！月度 + 趋势图、季节平均图和趋势汇总 CSV 已输出到：", outdir)
