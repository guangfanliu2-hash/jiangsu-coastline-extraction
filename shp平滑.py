import os
import math
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from scipy.signal import savgol_filter
from tkinter import Tk, filedialog, simpledialog

def ask_parameters():
    root = Tk()
    root.withdraw()

    shp_path = filedialog.askopenfilename(
        title="请选择输入岸线 Shapefile (*.shp)",
        filetypes=[("Shapefile", "*.shp")]
    )
    if not shp_path:
        raise SystemExit("❌ 未选择 shapefile，程序终止。")

    out_dir = filedialog.askdirectory(title="请选择输出目录（保存平滑后的 shp）")
    if not out_dir:
        raise SystemExit("❌ 未选择输出目录，程序终止。")

    smoothing_m = simpledialog.askfloat(
        "平滑尺度 (m)",
        "请输入平滑尺度（米），如 10 / 20 / 30：",
        initialvalue=20.0
    )
    if smoothing_m is None:
        raise SystemExit("❌ 未输入平滑尺度，程序终止。")

    spacing_m = simpledialog.askfloat(
        "重采样间距 (m)",
        "请输入重采样间距（米），推荐 1-5：",
        initialvalue=1.0
    )
    if spacing_m is None:
        spacing_m = 1.0

    polyorder = simpledialog.askinteger(
        "Savitzky 多项式阶数",
        "请输入多项式阶数（推荐 2 或 3）：",
        initialvalue=2
    )
    if polyorder is None:
        polyorder = 2

    root.destroy()
    return shp_path, out_dir, float(smoothing_m), float(spacing_m), int(polyorder)

def utm_epsg_for_lonlat(lon, lat):
    zone = int((lon + 180) / 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone

def ensure_metric_crs(gdf):
    if gdf.crs is None:
        raise ValueError("❌ 输入 shapefile 没有 CRS！请先在 GIS 中定义坐标系。")

    if gdf.crs.is_geographic:
        centroid = gdf.unary_union.centroid
        lon, lat = centroid.x, centroid.y
        epsg = utm_epsg_for_lonlat(lon, lat)
        gdf_proj = gdf.to_crs(f"EPSG:{epsg}")
        print(f"✅ 自动从经纬度投影到 UTM EPSG:{epsg}")
        return gdf_proj, epsg
    else:
        epsg = gdf.crs.to_epsg()
        print(f"✅ 输入数据已是投影坐标 EPSG:{epsg}")
        return gdf.copy(), epsg

# ============================
# 等距重采样
# ============================
def resample_linestring(ls, spacing):
    if ls.length == 0:
        return np.array(ls.coords)

    n_pts = max(2, int(math.ceil(ls.length / spacing)) + 1)
    distances = np.linspace(0, ls.length, n_pts)
    pts = [ls.interpolate(d) for d in distances]
    coords = np.array([[p.x, p.y] for p in pts])
    return coords

# ============================
# Savitzky-Golay 平滑
# ============================
def savgol_smooth_coords(coords, smoothing_m, spacing_m, polyorder=2):
    if coords.shape[0] < 3:
        return coords.copy()

    window = max(3, int(round(smoothing_m / spacing_m)))
    if window % 2 == 0:
        window += 1

    if window >= coords.shape[0]:
        window = coords.shape[0] - 1 if (coords.shape[0]-1) % 2 == 1 else coords.shape[0] - 2
        if window < 3:
            return coords.copy()

    if polyorder >= window:
        polyorder = max(1, window - 1)

    xs = coords[:, 0]
    ys = coords[:, 1]
    xs_s = savgol_filter(xs, window, polyorder, mode='interp')
    ys_s = savgol_filter(ys, window, polyorder, mode='interp')

    return np.vstack([xs_s, ys_s]).T

# ============================
# 主程序
# ============================
def main():
    shp_path, out_dir, smoothing_m, spacing_m, polyorder = ask_parameters()
    os.makedirs(out_dir, exist_ok=True)

    gdf = gpd.read_file(shp_path)
    print(f"✅ 读取岸线文件：{len(gdf)} 条")

    gdf_metric, used_epsg = ensure_metric_crs(gdf)

    out_geoms = []

    for geom in gdf_metric.geometry:
        if geom is None:
            out_geoms.append(None)
            continue

        parts = geom.geoms if isinstance(geom, MultiLineString) else [geom]
        smoothed_parts = []

        for part in parts:
            coords_rs = resample_linestring(part, spacing_m)
            smoothed_coords = savgol_smooth_coords(coords_rs, smoothing_m, spacing_m, polyorder)
            smoothed_parts.append(LineString(smoothed_coords))

        if len(smoothed_parts) == 1:
            out_geoms.append(smoothed_parts[0])
        else:
            out_geoms.append(MultiLineString(smoothed_parts))

    gdf_out = gdf_metric.copy()
    gdf_out["geometry"] = out_geoms

    base = os.path.splitext(os.path.basename(shp_path))[0]
    out_shp = os.path.join(out_dir, f"{base}_smoothed_{int(smoothing_m)}m.shp")
    gdf_out.to_file(out_shp)

    print("\n================ 处理完成 ✅ ================ ")
    print(f"📁 输出文件：{out_shp}")
    print(f"📏 平滑尺度：{smoothing_m} m")
    print(f"📐 重采样间距：{spacing_m} m")
    print(f"📊 Savitzky 多项式阶数：{polyorder}")
    print("===========================================\n")

if __name__ == "__main__":
    main()
