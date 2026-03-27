import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import geopandas as gpd
from tkinter import Tk, filedialog
from shapely.geometry import LineString, MultiLineString
from matplotlib import rcParams


# 设置中文字体
def set_chinese_font():
    candidates = ["Microsoft YaHei", "SimHei", "Heiti SC", "STHeiti", "WenQuanYi Zen Hei"]
    for c in candidates:
        try:
            rcParams['font.sans-serif'] = [c]
            rcParams['axes.unicode_minus'] = False
            return
        except Exception:
            continue
    rcParams['font.sans-serif'] = ['DejaVu Sans']
    rcParams['axes.unicode_minus'] = False


set_chinese_font()


def densify_geometry(geom, step_size):
    """
    对几何体进行插值加密，确保两点间距不大于 step_size。
    """
    if isinstance(geom, MultiLineString):
        lines = []
        for g in geom.geoms:
            lines.extend(densify_geometry(g, step_size))
        return lines
    elif isinstance(geom, LineString):
        length = geom.length
        if length == 0:
            return []
        num_segments = int(math.ceil(length / step_size))
        points = []
        for i in range(num_segments + 1):
            point = geom.interpolate(i * length / num_segments)
            points.append([point.x, point.y])
        return points
    return []


def fractal_dimension_realworld(points, box_sizes_m):
    Ns = []
    points_np = np.array(points)
    x, y = points_np[:, 0], points_np[:, 1]

    x_min, y_min = x.min(), y.min()

    for eps in box_sizes_m:
        if eps <= 0:
            Ns.append(0)
            continue

        # 计算网格坐标
        xi = np.floor((x - x_min) / eps).astype(int)
        yi = np.floor((y - y_min) / eps).astype(int)

        if xi.size == 0:
            Ns.append(0)
            continue

        # 【核心逻辑】：np.unique 保证了同一个网格无论有多少个点，只会被计数一次
        # 这就解决了“重复覆盖”的疑虑
        unique_boxes = np.unique(np.stack([xi, yi], axis=1), axis=0)
        Ns.append(len(unique_boxes))

    Ns = np.array(Ns, dtype=float)

    # 拟合计算 D
    valid = (Ns > 0)
    if np.sum(valid) < 2:
        return np.nan, np.nan, Ns

    log_eps = np.log(np.array(box_sizes_m)[valid])
    log_N = np.log(Ns[valid])

    slope, intercept, r_value, p_value, std_err = linregress(log_eps, log_N)
    D = -slope
    R2 = r_value ** 2

    return D, R2, Ns


def project_to_metric(gdf):
    """确保数据转换为投影坐标系（米）"""
    if gdf.crs is None:
        raise ValueError("输入 shapefile 没有定义 CRS。")
    results = []
    is_geo = gdf.crs.is_geographic
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            results.append((None, None))
            continue
        if is_geo:
            lon, lat = geom.centroid.x, geom.centroid.y
            zone = int((lon + 180) / 6) + 1
            epsg = 32600 + zone if lat >= 0 else 32700 + zone
            geom_proj = gpd.GeoSeries([geom], crs=gdf.crs).to_crs(f"EPSG:{epsg}").iloc[0]
            results.append((geom_proj, epsg))
        else:
            epsg = gdf.crs.to_epsg() if gdf.crs.to_epsg() else None
            results.append((geom, epsg))
    return results


def extract_year(date_val):
    """尝试从各种格式中提取年份"""
    s = str(date_val).strip()
    # 尝试解析前4位数字
    import re
    match = re.search(r'(\d{4})', s)
    if match:
        return match.group(1)
    return "Unknown_Year"


def main():
    Tk().withdraw()
    shp_path = filedialog.askopenfilename(title="请选择岸线 Shapefile", filetypes=[("Shapefile", "*.shp")])
    if not shp_path: return
    out_dir = filedialog.askdirectory(title="请选择输出文件夹")
    if not out_dir: return

    # 定义盒子尺寸
    box_sizes = [10, 20, 30, 60, 120, 240, 360, 500, 998]

    # 加密步长
    densify_step = min(box_sizes) / 2.0
    print(f"插值加密步长: {densify_step} 米")

    gdf = gpd.read_file(shp_path)

    # 寻找时间字段
    time_field = None
    possible_names = ['date', 'Date', 'DATE', 'time', 'Time', 'year', 'Year']
    for f in possible_names:
        if f in gdf.columns:
            time_field = f
            break

    if time_field is None:
        print("未找到时间字段，所有数据将作为一个整体（Unknown_Year）处理。")
        gdf['__year_extracted__'] = "Unknown_Year"
    else:
        # 提取年份
        gdf['__year_extracted__'] = gdf[time_field].apply(extract_year)

    print(f"正在进行投影转换...")
    proj_results = project_to_metric(gdf)

    # 【关键修改】：按年份分组，而不是按具体的日期字符串分组
    # 这样会将 2018-01, 2018-02 等所有属于 2018 的片段合并
    grouped_by_year = {}

    for i, row in gdf.iterrows():
        geom_proj, epsg = proj_results[i]
        if geom_proj is None: continue

        year = row['__year_extracted__']

        if year not in grouped_by_year:
            grouped_by_year[year] = {"geoms": [], "epsg": epsg}
        grouped_by_year[year]["geoms"].append(geom_proj)

    rows_detail = []
    rows_summary = []

    print(f"开始计算分形维数（共 {len(grouped_by_year)} 个年份组）...")

    for year, info in grouped_by_year.items():
        geoms = info["geoms"]
        epsg = info["epsg"]

        # 1. 计算该年份下的【真实几何总长度】
        # 这里把该年内所有片段的长度直接相加
        total_len_real = sum(g.length for g in geoms)

        # 2. 生成点集（合并该年所有片段的点）
        all_dense_points = []
        for g in geoms:
            dense_pts = densify_geometry(g, densify_step)
            all_dense_points.extend(dense_pts)

        if not all_dense_points:
            continue

        points_for_calc = np.array(all_dense_points)

        # 3. 盒子计数
        D, R2, Ns = fractal_dimension_realworld(points_for_calc, box_sizes)

        # 4. 计算不同盒子下的近似长度
        length_box_array = Ns * np.array(box_sizes)

        # 写入详细数据
        for bs, n, l_box in zip(box_sizes, Ns, length_box_array):
            rows_detail.append({
                "年份": year,
                "盒子分辨率(米)": bs,
                "盒子数量(N)": int(n),
                "盒计数长度(米)": float(l_box),  # N * epsilon
                "真实岸线总长度(米)": total_len_real,  # 该年份所有片段的几何长度之和
                "分形维数(D)": D,
                "拟合R²": R2,
                "EPSG": epsg
            })

        # 写入汇总数据
        rows_summary.append({
            "年份": year,
            "分形维数(D)": D,
            "拟合R²": R2,
            "真实岸线总长度(米)": total_len_real,
            "EPSG": epsg
        })

        # 绘图
        valid = Ns > 0
        if np.sum(valid) >= 2:
            log_eps = np.log(box_sizes)[valid]
            log_N = np.log(Ns[valid])
            slope, intercept, r, _, _ = linregress(log_eps, log_N)

            plt.figure(figsize=(6, 5))
            plt.scatter(log_eps, log_N, c='blue', label='观测值')
            plt.plot(log_eps, slope * log_eps + intercept, 'r--', label=f'D={-slope:.3f}')
            plt.title(f"年份: {year} (真实总长: {total_len_real:.0f}m)")
            plt.xlabel("ln(ε)")
            plt.ylabel("ln(N)")
            plt.legend()
            plt.grid(True, ls=':')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"Fractal_{year}.png"), dpi=150)
            plt.close()

    # 保存 CSV
    df_detail = pd.DataFrame(rows_detail)
    # 按照年份排序
    df_detail = df_detail.sort_values(['年份', '盒子分辨率(米)'])
    df_detail.to_csv(os.path.join(out_dir, "岸线分形_详细盒计数.csv"), index=False, encoding='utf-8-sig')

    df_sum = pd.DataFrame(rows_summary)
    df_sum = df_sum.sort_values('年份')
    df_sum.to_csv(os.path.join(out_dir, "岸线分形_年度汇总.csv"), index=False, encoding='utf-8-sig')

    print("✅ 处理完成！")
    print(f"结果已按年份合并，并计算了每一年的岸线总长度。")


if __name__ == "__main__":
    main()