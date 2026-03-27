# shoreline_accuracy_assessment.py
import os
import math
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString
from tkinter import Tk, filedialog, simpledialog

# 忽略一些且版本警告
warnings.filterwarnings("ignore")


# ---------------- 工具函数 ----------------

def get_utm_crs(gdf):
    """根据数据中心点自动获取 UTM 投影 CRS"""
    # 如果本身就是投影坐标系，直接返回
    if not gdf.crs.is_geographic:
        return gdf.crs

    # 获取中心点
    lon_center = gdf.geometry.centroid.x.mean()
    lat_center = gdf.geometry.centroid.y.mean()

    # 计算 UTM 带号
    zone = int((lon_center + 180) / 6) + 1

    # 判断北/南半球
    if lat_center >= 0:
        epsg_code = 32600 + zone  # 北半球
    else:
        epsg_code = 32700 + zone  # 南半球

    return f"EPSG:{epsg_code}"


def extract_vertices(geometry):
    """
    将 LineString 或 MultiLineString 打散为所有折点坐标的列表
    """
    points = []
    if isinstance(geometry, LineString):
        points.extend(list(geometry.coords))
    elif isinstance(geometry, MultiLineString):
        for geom in geometry.geoms:
            points.extend(list(geom.coords))
    return points


# ---------------- 主程序 ----------------

def main():
    # 1.不仅初始化 Tkinter，还隐藏主窗口
    root = Tk()
    root.withdraw()

    print("--- 岸线提取类别性错误率计算工具 ---")

    # 2. 选择文件
    shp_path = filedialog.askopenfilename(
        title="请选择包含参考线和提取岸线的 Shapefile",
        filetypes=[("Shapefile", "*.shp")]
    )
    if not shp_path:
        print("未选择文件，程序退出。")
        return

    # 3. 输入参数
    limit_dist = simpledialog.askfloat("参数设置", "请输入评定范围缓冲区半径 (米):", initialvalue=100.0)
    valid_dist = simpledialog.askfloat("参数设置", "请输入正确性判定缓冲区半径 (米):", initialvalue=10.0)

    if limit_dist is None or valid_dist is None:
        print("参数输入取消，退出。")
        return

    print(f"读取文件: {shp_path}")
    gdf = gpd.read_file(shp_path)

    # 4. 检查字段
    # 统一将 date 字段转为字符串处理，防止格式问题
    if 'date' not in gdf.columns:
        print("错误：Shapefile 中缺少 'date' 字段。")
        return
    gdf['date_str'] = gdf['date'].astype(str)

    # 5. 坐标系投影转换 (关键步骤)
    # 计算缓冲区必须使用投影坐标系(米)，不能用经纬度
    target_crs = get_utm_crs(gdf)
    print(f"转换坐标系至: {target_crs} 以进行距离计算...")
    gdf_proj = gdf.to_crs(target_crs)

    # 6. 分离参考线和提取线
    ref_gdf = gdf_proj[gdf_proj['date_str'] == '评定']
    target_gdfs = gdf_proj[gdf_proj['date_str'] != '评定']

    if ref_gdf.empty:
        print("错误：未找到 date='评定' 的参考岸线。")
        return

    if target_gdfs.empty:
        print("错误：未找到提取岸线数据。")
        return

    # 合并所有参考线（防止有多条评定线段）为一个几何体
    ref_geom = ref_gdf.geometry.unary_union

    # 7. 生成双重缓冲区
    print(f"生成缓冲区: 评定范围={limit_dist}m, 正确范围={valid_dist}m")
    # 宽缓冲区：限制评定区域
    limit_buffer = ref_geom.buffer(limit_dist)
    # 窄缓冲区：判定正确区域
    valid_buffer = ref_geom.buffer(valid_dist)

    results = []

    print("\n开始逐个日期计算错误率...")
    print("-" * 60)
    print(f"{'日期':<15} | {'总点数(评定区)':<10} | {'正确点数':<10} | {'错误点数':<10} | {'错误率(%)'}")
    print("-" * 60)

    # 按日期分组处理
    for date_val, group in target_gdfs.groupby('date_str'):
        # 提取当前日期所有线要素的折点
        all_coords = []
        for geom in group.geometry:
            all_coords.extend(extract_vertices(geom))

        if not all_coords:
            continue

        # 将点转换为 GeoDataFrame 以进行快速空间查询
        # 这里创建一个点的集合
        points_gdf = gpd.GeoDataFrame(
            geometry=[Point(xy) for xy in all_coords],
            crs=gdf_proj.crs
        )

        # ---------------------------------------------------------
        # 核心算法：两级筛选
        # ---------------------------------------------------------

        # 第一级筛选：只保留落在 [评定范围(100m)] 内的点
        # 使用 within 检查点是否在 limit_buffer 内
        points_in_limit = points_gdf[points_gdf.geometry.within(limit_buffer)]

        total_eval_count = len(points_in_limit)

        if total_eval_count == 0:
            print(f"{date_val:<15} | {'0':<10} | {'-':<10} | {'-':<10} | 此日期无点落入评定范围")
            continue

        # 第二级筛选：在第一级的基础上，检查点是否在 [正确范围(10m)] 内
        points_correct = points_in_limit[points_in_limit.geometry.within(valid_buffer)]

        correct_count = len(points_correct)

        # 计算
        error_count = total_eval_count - correct_count
        error_rate = (error_count / total_eval_count) * 100.0

        # 打印控制台
        print(f"{date_val:<15} | {total_eval_count:<13} | {correct_count:<13} | {error_count:<13} | {error_rate:.2f}%")

        results.append({
            "日期": date_val,
            "评定范围内总点数": total_eval_count,
            "正确点数": correct_count,
            "错误点数": error_count,
            "类别性错误率(%)": round(error_rate, 2)
        })

    print("-" * 60)

    # 8. 保存结果
    if results:
        res_df = pd.DataFrame(results)

        # 计算平均错误率
        avg_error = res_df["类别性错误率(%)"].mean()
        print(f"\n所有日期平均错误率: {avg_error:.2f}%")

        # 构造输出路径
        input_dir = os.path.dirname(shp_path)
        out_csv = os.path.join(input_dir, "岸线提取精度评定结果.csv")
        res_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
        print(f"\n详细结果已保存至: {out_csv}")
    else:
        print("\n未生成有效计算结果。")

    # 暂停一下以便用户看到结果
    input("按回车键退出...")


if __name__ == "__main__":
    main()