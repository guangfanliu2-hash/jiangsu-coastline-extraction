import pandas as pd
import numpy as np
from scipy.stats import linregress
from tkinter import Tk, filedialog
import os


def calculate_stats(values, label_prefix):
    """
    辅助函数：计算一组数据的侵蚀/淤积统计特征
    values: 包含正负值的数组
    """
    values = np.array(values)
    values = values[~np.isnan(values)]  # 去除空值

    if len(values) == 0:
        print(f"  {label_prefix}: 无有效数据")
        return {}

    # 淤积 (Accretion): 正值
    accretion = values[values > 0]
    # 侵蚀 (Erosion): 负值
    erosion = values[values < 0]

    stats = {}

    # --- 淤积统计 ---
    if len(accretion) > 0:
        stats[f'{label_prefix}_淤积_最大值'] = np.max(accretion)
        stats[f'{label_prefix}_淤积_最小值'] = np.min(accretion)
        stats[f'{label_prefix}_淤积_平均值'] = np.mean(accretion)
    else:
        stats[f'{label_prefix}_淤积_最大值'] = 0
        stats[f'{label_prefix}_淤积_最小值'] = 0
        stats[f'{label_prefix}_淤积_平均值'] = 0

    # --- 侵蚀统计 (保留负号) ---
    if len(erosion) > 0:
        # 最大侵蚀：通常指流失最严重，即数值最小（如 -10 比 -1 更严重）
        stats[f'{label_prefix}_侵蚀_最大值(最严重)'] = np.min(erosion)
        # 最小侵蚀：指流失最少，即数值最大（最接近0）
        stats[f'{label_prefix}_侵蚀_最小值(最轻微)'] = np.max(erosion)
        stats[f'{label_prefix}_侵蚀_平均值'] = np.mean(erosion)
    else:
        stats[f'{label_prefix}_侵蚀_最大值(最严重)'] = 0
        stats[f'{label_prefix}_侵蚀_最小值(最轻微)'] = 0
        stats[f'{label_prefix}_侵蚀_平均值'] = 0

    return stats


def main():
    # 1. 选择文件
    Tk().withdraw()
    csv_path = filedialog.askopenfilename(
        title="请选择 transects_distances.csv 文件",
        filetypes=[("CSV Files", "*.csv")]
    )
    if not csv_path:
        print("未选择文件。")
        return

    print("读取数据中...")
    df = pd.read_csv(csv_path)

    # 确保时间格式正确
    df['date_dt'] = pd.to_datetime(df['date'])

    # 将时间转换为小数年用于拟合 (例如 2018.5)
    def to_year_fraction(date):
        start_of_year = pd.Timestamp(year=date.year, month=1, day=1)
        days_in_year = 366 if date.is_leap_year else 365
        return date.year + (date - start_of_year).days / days_in_year

    df['year_frac'] = df['date_dt'].apply(to_year_fraction)
    df['year_int'] = df['date_dt'].dt.year

    # ==========================================
    # 1. 计算【长期拟合趋势】 (Trend / Slope)
    # ==========================================
    print("正在计算长期拟合趋势...")
    transect_slopes = []

    # 按断面分组计算斜率
    for tid, grp in df.groupby('transect_id'):
        if len(grp) < 2:
            continue

        # 使用 distance_m_used (插值后的距离)
        y = grp['distance_m_used'].values
        x = grp['year_frac'].values

        # 线性回归
        slope, intercept, r_val, p_val, std_err = linregress(x, y)
        transect_slopes.append(slope)

    trend_stats = calculate_stats(transect_slopes, "长期趋势(米/年)")

    # ==========================================
    # 2. 计算【逐年变化量】 (Annual Change)
    # ==========================================
    print("正在计算逐年变化量...")

    # 透视表：行=断面，列=年份，值=距离
    # 如果一年有多个数据，取平均值代表该年位置
    pivot_year = df.pivot_table(index='transect_id', columns='year_int', values='distance_m_used', aggfunc='mean')

    # 计算相邻年份的差值 (Year_N - Year_N-1)
    diff_df = pivot_year.diff(axis=1)

    # 将所有断面的所有年份变化展平成一个长列表
    all_annual_changes = diff_df.values.flatten()

    annual_stats = calculate_stats(all_annual_changes, "逐年变化(米)")

    # ==========================================
    # 3. 输出结果
    # ==========================================
    print("\n" + "=" * 50)
    print("           岸线侵蚀淤积统计结果")
    print("=" * 50)

    print(f"\n【1. 长期拟合趋势 (Linear Regression Rate)】")
    print(f"   (基于所有年份数据的线性回归斜率)")
    print(f"   - 淤积断面的最大速率: {trend_stats.get('长期趋势(米/年)_淤积_最大值', 0):.4f} m/a")
    print(f"   - 淤积断面的最小速率: {trend_stats.get('长期趋势(米/年)_淤积_最小值', 0):.4f} m/a")
    print(f"   - 淤积断面的平均速率: {trend_stats.get('长期趋势(米/年)_淤积_平均值', 0):.4f} m/a")
    print("-" * 30)
    print(f"   - 侵蚀断面的最大速率(最严重): {trend_stats.get('长期趋势(米/年)_侵蚀_最大值(最严重)', 0):.4f} m/a")
    print(f"   - 侵蚀断面的最小速率(最轻微): {trend_stats.get('长期趋势(米/年)_侵蚀_最小值(最轻微)', 0):.4f} m/a")
    print(f"   - 侵蚀断面的平均速率: {trend_stats.get('长期趋势(米/年)_侵蚀_平均值', 0):.4f} m/a")

    print(f"\n【2. 逐年变化幅度 (Year-to-Year Change)】")
    print(f"   (基于相邻年份的距离差值)")
    print(f"   - 单年最大淤积量: {annual_stats.get('逐年变化(米)_淤积_最大值', 0):.4f} m")
    print(f"   - 单年最小淤积量: {annual_stats.get('逐年变化(米)_淤积_最小值', 0):.4f} m")
    print(f"   - 单年平均淤积量: {annual_stats.get('逐年变化(米)_淤积_平均值', 0):.4f} m")
    print("-" * 30)
    print(f"   - 单年最大侵蚀量(最严重): {annual_stats.get('逐年变化(米)_侵蚀_最大值(最严重)', 0):.4f} m")
    print(f"   - 单年最小侵蚀量(最轻微): {annual_stats.get('逐年变化(米)_侵蚀_最小值(最轻微)', 0):.4f} m")
    print(f"   - 单年平均侵蚀量: {annual_stats.get('逐年变化(米)_侵蚀_平均值', 0):.4f} m")

    print("\n" + "=" * 50)

    # 保存统计结果到文本文件
    out_dir = os.path.dirname(csv_path)
    out_txt = os.path.join(out_dir, "Stats_Erosion_Accretion.txt")
    with open(out_txt, "w", encoding='utf-8') as f:
        f.write("岸线侵蚀淤积统计报告\n")
        f.write("========================\n\n")
        for k, v in trend_stats.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("\n")
        for k, v in annual_stats.items():
            f.write(f"{k}: {v:.4f}\n")

    print(f"统计报告已保存至: {out_txt}")
    print("输入任意键退出...")
    input()


if __name__ == "__main__":
    main()