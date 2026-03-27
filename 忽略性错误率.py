# shoreline_omission_error_assessment.py
import os
import math
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString
from tkinter import Tk, filedialog, simpledialog

# 忽略警告
warnings.filterwarnings("ignore")


# ---------------- 工具函数 ----------------

def get_utm_crs(gdf):
    """根据数据中心点自动获取 UTM 投影 CRS"""
    if not gdf.crs.is_geographic:
        return gdf.crs

    lon_center = gdf.geometry.centroid.x.mean()
    lat_center = gdf.geometry.centroid.y.mean()
    zone = int((lon_center + 180) / 6) + 1

    if lat_center >= 0:
        epsg_code = 32600 + zone
    else:
        epsg_code = 32700 + zone

    return f"EPSG:{epsg_code}"


def extract_vertices(geometry):
    """将几何体打散为折点坐标列表"""
    points = []
    if isinstance(geometry, LineString):
        points.extend(list(geometry.coords))
    elif isinstance(geometry, MultiLineString):
        for geom in geometry.geoms:
            points.extend(list(geom.coords))
    return points


# ---------------- 主程序 ----------------

def main():
    root = Tk()
    root.withdraw()

    print("--- 岸线提取忽略性错误率（Omission Error）计算工具 ---")

    # 1. 选择文件
    shp_path = filedialog.askopenfilename(
        title="请选择包含参考线和提取岸线的 Shapefile",
        filetypes=[("Shapefile", "*.shp")]
    )
    if not shp_path:
        print("未选择文件，退出。")
        return

    # 2. 输入参数 (基于提取线建立缓冲区)
    limit_dist = simpledialog.askfloat("参数设置", "请输入评定范围限制半径 (米, 默认100):", initialvalue=100.0)
    valid_dist = simpledialog.askfloat("参数设置", "请输入有效提取判定半径 (米, 默认10):", initialvalue=10.0)

    if limit_dist is None or valid_dist is None:
        return

    print(f"读取文件: {shp_path}")
    gdf = gpd.read_file(shp_path)

    if 'date' not in gdf.columns:
        print("错误：缺少 'date' 字段。")
        return
    gdf['date_str'] = gdf['date'].astype(str)

    # 3. 坐标投影转换
    target_crs = get_utm_crs(gdf)
    print(f"转换坐标系至: {target_crs}")
    gdf_proj = gdf.to_crs(target_crs)

    # 4. 数据分离
    # 参考线 (Truth)
    ref_gdf = gdf_proj[gdf_proj['date_str'] == '评定']
    # 提取线 (Result)
    target_gdfs = gdf_proj[gdf_proj['date_str'] != '评定']

    if ref_gdf.empty:
        print("错误：未找到 date='评定' 的参考岸线。")
        return
    if target_gdfs.empty:
        print("错误：未找到提取岸线数据。")
        return

    # 5. 预处理参考线折点
    # 因为我们要计算的是“参考线的折点”有多少被遗漏了
    print("正在提取参考岸线折点...")
    ref_coords = []
    for geom in ref_gdf.geometry:
        ref_coords.extend(extract_vertices(geom))

    # 将参考线的所有折点转为 GeoDataFrame
    ref_points_gdf = gpd.GeoDataFrame(
        geometry=[Point(xy) for xy in ref_coords],
        crs=gdf_proj.crs
    )
    total_ref_points_raw = len(ref_points_gdf)
    print(f"参考岸线原始总折点数: {total_ref_points_raw}")

    results = []

    print("\n开始逐个日期计算忽略性错误率...")
    print("原理：以提取岸线为中心生成缓冲区，检查参考线折点是否落入。")
    print("-" * 80)
    print(f"{'日期':<15} | {'评定区参考点数':<14} | {'被提取点数':<12} | {'遗漏点数':<10} | {'忽略性错误率(%)'}")
    print("-" * 80)

    # 6. 逐日期计算
    for date_val, group in target_gdfs.groupby('date_str'):
        # 合并该日期下提取的所有线段，处理 MultiLineString
        extracted_geom_union = group.geometry.unary_union

        # --- 生成双重缓冲区 (基于提取岸线) ---

        # 1. 限制缓冲区 (Limit Buffer)：
        # 用于定义分母。即：我们只关心提取线周围 100m 内的参考线是否被提取。
        # 如果参考线在 5km 以外，说明那个区域压根没进行提取，不计入本次精度的分母。
        limit_buffer = extracted_geom_union.buffer(limit_dist)

        # 2. 有效缓冲区 (Valid Buffer)：
        # 用于定义分子（捕获成功）。即：提取线 10m 范围内的参考点算作“正确提取”。
        valid_buffer = extracted_geom_union.buffer(valid_dist)

        # --- 空间筛选 ---

        # 筛选1：确定分母 (参与评定的参考点)
        ref_in_limit = ref_points_gdf[ref_points_gdf.geometry.within(limit_buffer)]
        denominator = len(ref_in_limit)

        if denominator == 0:
            print(f"{date_val:<15} | {'0':<14} | {'-':<12} | {'-':<10} | 无参考点落入评定范围")
            continue

        # 筛选2：确定命中数 (被有效覆盖的参考点)
        # 注意：是在 ref_in_limit 的基础上筛选，还是直接筛选？结果是一样的，但这样更快。
        ref_hit = ref_in_limit[ref_in_limit.geometry.within(valid_buffer)]
        hit_count = len(ref_hit)

        # 计算遗漏
        miss_count = denominator - hit_count

        # 计算错误率
        omission_rate = (miss_count / denominator) * 100.0

        print(f"{date_val:<15} | {denominator:<14} | {hit_count:<12} | {miss_count:<10} | {omission_rate:.2f}%")

        results.append({
            "日期": date_val,
            "评定范围内参考点总数": denominator,
            "被成功覆盖的参考点数": hit_count,
            "遗漏的参考点数": miss_count,
            "忽略性错误率(%)": round(omission_rate, 2)
        })

    print("-" * 80)

    # 7. 保存结果
    if results:
        res_df = pd.DataFrame(results)
        avg_error = res_df["忽略性错误率(%)"].mean()
        print(f"\n平均忽略性错误率: {avg_error:.2f}%")

        input_dir = os.path.dirname(shp_path)
        out_csv = os.path.join(input_dir, "岸线提取忽略性错误率结果.csv")
        res_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存至: {out_csv}")
    else:
        print("\n未生成有效结果。")

    input("按回车键退出...")


if __name__ == "__main__":
    main()