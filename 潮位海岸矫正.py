import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tkinter import Tk, filedialog
from scipy import stats


rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# -------------------------------
# 参数设置（保持 CoastSat 的思路）
# -------------------------------
min_points = 4       # 至少需要多少点才拟合
min_tide_range = 0.3 # 潮差至少多少米
tide_ref_mode = "median"  # 参考潮位: "median" 或固定数值

# -------------------------------
# 选择输入/输出
# -------------------------------
Tk().withdraw()

print("请选择岸线数据 shorelines.pkl")
pkl_file = filedialog.askopenfilename(title="选择岸线 .pkl 文件", filetypes=[("Pickle files","*.pkl")])
if not pkl_file:
    raise FileNotFoundError("未选择 pkl 文件")

print("请选择潮位 CSV 文件 (格式: datetime, tide_m)")
tide_file = filedialog.askopenfilename(title="选择潮位 CSV 文件", filetypes=[("CSV files","*.csv")])
if not tide_file:
    raise FileNotFoundError("未选择潮位文件")

print("请选择输出文件夹")
save_dir = filedialog.askdirectory(title="选择输出文件夹")
if not save_dir:
    raise FileNotFoundError("未选择输出文件夹")

os.makedirs(save_dir, exist_ok=True)

# -------------------------------
# 读取数据
# -------------------------------
with open(pkl_file, "rb") as f:
    shoreline_data = pickle.load(f)

dates = shoreline_data["dates"]
transects = shoreline_data.get("transects", None)
if transects is None:
    raise ValueError("shorelines.pkl 里没有 transects 数据，请确认输入正确")

tide_df = pd.read_csv(tide_file, parse_dates=["dates"])
tide_df = tide_df.rename(columns={"dates": "datetime", "tide": "tide_m"})
tide_df["datetime"] = tide_df["datetime"].dt.tz_localize(None)
tide_df = tide_df.set_index("datetime").sort_index()

print(f"✅ 载入 {len(dates)} 个日期的岸线数据")
print(f"✅ 载入 {len(tide_df)} 条潮位记录")

# -------------------------------
# 定义函数
# -------------------------------
def get_tide_at_dates(dates, tide_df):
    # 转换 dates 为 DatetimeIndex
    date_index = pd.DatetimeIndex(dates)
    # 最近邻插值 (允许最大 1 小时差)
    tide_series = tide_df["tide_m"].reindex(date_index, method="nearest", tolerance=pd.Timedelta("1h"))
    # 如果有找不到的值，用线性插值补齐
    tide_series = tide_series.interpolate(limit_direction="both")
    return tide_series.values


def estimate_beach_slope(R, tide, min_points=8, min_tide_range=0.3):
    ok = ~np.isnan(R) & ~np.isnan(tide)
    if ok.sum() < min_points:
        return None
    tide_range = tide[ok].max() - tide[ok].min()
    if tide_range < min_tide_range:
        return None
    slope, intercept, r_value, p_value, std_err = stats.linregress(tide[ok], R[ok])
    return {
        "a": intercept,
        "b": slope,
        "r2": r_value**2,
        "p": p_value,
        "stderr_b": std_err,
        "n": ok.sum(),
        "tide_range": tide_range,
    }

def apply_tidal_correction(R, tide, b, tide_ref):
    return R - b * (tide - tide_ref)

# -------------------------------
# 主循环：逐 transect
# -------------------------------
summary = []
corrected = {}

tide_all = get_tide_at_dates(dates, tide_df)
if tide_ref_mode == "median":
    tide_ref = np.nanmedian(tide_all)
else:
    tide_ref = float(tide_ref_mode)

for name, R in transects.items():
    R = np.array(R, dtype=float)
    tide = tide_all

    fit = estimate_beach_slope(R, tide, min_points, min_tide_range)
    if fit is None:
        print(f"⚠️ {name}: 样本不足或潮差太小，跳过拟合")
        corrected[name] = {"raw": R, "corrected": R, "tide": tide}
        continue

    R_corr = apply_tidal_correction(R, tide, fit["b"], tide_ref)
    corrected[name] = {"raw": R, "corrected": R_corr, "tide": tide}

    summary.append({
        "transect": name,
        "a": fit["a"],
        "b": fit["b"],
        "r2": fit["r2"],
        "stderr_b": fit["stderr_b"],
        "n": fit["n"],
        "tide_range": fit["tide_range"],
    })

    # (1) 潮位 vs 岸线位置
    plt.figure(figsize=(6,5))
    plt.scatter(tide, R, c="blue", s=20, label="原始点")
    if fit is not None:
        xfit = np.linspace(tide.min(), tide.max(), 50)
        yfit = fit["a"] + fit["b"]*xfit
        plt.plot(xfit, yfit, "r-", label=f"拟合: b={fit['b']:.2f}, R²={fit['r2']:.2f}")
    plt.xlabel("潮位 (m)")
    plt.ylabel("岸线位置 (m)")
    plt.title(f"{name}: 潮位-岸线拟合")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{name}_fit.png"), dpi=200)
    plt.close()

    # (2) 时间序列：原始 vs 校正
    plt.figure(figsize=(10,5))
    plt.plot(dates, R, "bo-", label="原始")
    plt.plot(dates, R_corr, "ro-", label="校正后")
    plt.ylabel("岸线位置 (m)")
    plt.xlabel("日期")
    plt.title(f"{name}: 时序对比")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{name}_timeseries.png"), dpi=200)
    plt.close()

# -------------------------------
# 保存结果
# -------------------------------
df_summary = pd.DataFrame(summary)
df_summary.to_csv(os.path.join(save_dir, "transect_summary.csv"), index=False, encoding="utf-8-sig")

out_pkl = os.path.join(save_dir, "shorelines_tide_corrected.pkl")
with open(out_pkl, "wb") as f:
    pickle.dump({"dates": dates, "transects": corrected}, f)

df_corr = pd.DataFrame(
    {name: np.array(corrected[name]["corrected"], dtype=float) for name in corrected},
    index=pd.to_datetime(dates)
)
df_corr.index.name = "date"
out_csv = os.path.join(save_dir, "shorelines_tide_corrected.csv")
df_corr.to_csv(out_csv, encoding="utf-8-sig", float_format="%.4f")

print(f"\n🎉 全部完成！结果已保存到: {save_dir}")
