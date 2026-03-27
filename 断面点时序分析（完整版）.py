import os
import math
import pickle
import warnings
import re

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point, MultiLineString
from shapely.ops import nearest_points, linemerge
from pyproj import Transformer, CRS
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import linregress
from tkinter import Tk, filedialog, simpledialog
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# ----------------------- 可调参数 -----------------------
DEFAULT_TRANSECT_SPACING = 30.0  # 断面间距
DEFAULT_TRANSECT_HALF_LENGTH = 1000.0  # 断面半长
DEFAULT_PANEL_WIDTH = 500.0  # 展开图每张图显示长度
UNFOLD_SAMPLING_STEP = 5.0  # 展开图采样间距
# -------------------------------------------------------

from matplotlib import rcParams


def set_chinese_font():
    cand = ["Microsoft YaHei", "SimHei", "Heiti SC", "STHeiti", "WenQuanYi Zen Hei"]
    for f in cand:
        try:
            rcParams['font.sans-serif'] = [f]
            rcParams['axes.unicode_minus'] = False
            return
        except Exception:
            continue
    rcParams['font.sans-serif'] = ['DejaVu Sans']
    rcParams['axes.unicode_minus'] = False


set_chinese_font()


# ---------------- helper functions ----------------
def utm_epsg_for_lonlat(lon, lat):
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        return 32600 + zone
    else:
        return 32700 + zone


def densify_line(line: LineString, spacing_m: float):
    if line is None:
        return []
    if line.is_empty:
        return []
    if line.length == 0:
        return []

    n = max(int(math.ceil(line.length / spacing_m)), 1)
    pts = [line.interpolate(float(i) / n, normalized=True) for i in range(n + 1)]
    return np.array([[p.x, p.y] for p in pts])


def unit_normal_at_point(line: LineString, t=0.5):
    coords = np.array(line.coords)
    dx = coords[-1, 0] - coords[0, 0]
    dy = coords[-1, 1] - coords[0, 1]
    L = math.hypot(dx, dy)
    if L == 0: return np.array([0.0, 0.0])
    tx, ty = dx / L, dy / L
    return np.array([ty, -tx])


def project_point_onto_line(line: LineString, pt: Point):
    proj_dist = line.project(pt)
    proj_pt = line.interpolate(proj_dist)
    return proj_pt, proj_dist


# ---------------- main ----------------
def main():
    Tk().withdraw()
    shp_path = filedialog.askopenfilename(title="请选择平滑后的岸线 Shapefile", filetypes=[("Shapefile", "*.shp")])
    if not shp_path: return
    out_dir = filedialog.askdirectory(title="请选择输出目录")
    if not out_dir: return

    transect_spacing = simpledialog.askfloat("参数", "断面间距（米）:", initialvalue=DEFAULT_TRANSECT_SPACING)
    transect_half_length = simpledialog.askfloat("参数", "断面半长（米）:", initialvalue=DEFAULT_TRANSECT_HALF_LENGTH)
    panel_width = simpledialog.askfloat("参数", "展开图每张图显示长度（米）:", initialvalue=DEFAULT_PANEL_WIDTH)
    interp_choice = simpledialog.askstring("补全", "是否线性插值补全缺失值？(Y/N)", initialvalue='Y')
    interp = (str(interp_choice).strip().upper() == 'Y')
    dir_choice = simpledialog.askinteger("方向", "法线方向(1或-1):", initialvalue=1)
    if dir_choice not in (1, -1): dir_choice = 1

    # ================= 修改：允许输入年份，匹配该年份所有数据 =================
    ref_year_input = simpledialog.askstring("基准选择",
                                            "请输入作为相对基准的年份（如 2018）。\n"
                                            "程序将合并该年份下的所有岸线段作为 0 基准。\n"
                                            "若留空，则使用原始几何基线。",
                                            initialvalue="")
    # ====================================================================

    print("读取数据...")
    gdf = gpd.read_file(shp_path)

    base_gdf = gdf[gdf['date'].astype(str) == '基线']
    if base_gdf.empty: raise ValueError("未找到 '基线' 要素")
    baseline = base_gdf.iloc[0].geometry

    shore_gdf = gdf[gdf['date'].astype(str) != '基线'].copy()
    if shore_gdf.empty: raise ValueError("未找到岸线要素")

    orig_crs = gdf.crs
    if orig_crs and orig_crs.is_geographic:
        c = baseline.centroid
        epsg = utm_epsg_for_lonlat(c.x, c.y)
        print(f"转换投影至 EPSG:{epsg}")
        target_crs = CRS.from_epsg(epsg)
        gdf_utm = gdf.to_crs(target_crs.to_string())
    else:
        gdf_utm = gdf.copy()

    baseline_utm = gdf_utm[gdf_utm['date'].astype(str) == '基线'].iloc[0].geometry
    shore_gdf_utm = gdf_utm[gdf_utm['date'].astype(str) != '基线'].copy()
    shore_gdf_utm['date_str'] = shore_gdf_utm['date'].astype(str)
    unique_dates = sorted(shore_gdf_utm['date_str'].unique())

    # ================= 修改：识别属于该年份的所有日期列表 =================
    ref_target_dates = []  # 这是一个列表，存放所有匹配的日期
    if ref_year_input and ref_year_input.strip():
        # 只要日期字符串包含输入的年份，就加入列表
        ref_target_dates = [d for d in unique_dates if ref_year_input in d]
        if ref_target_dates:
            print(f"✅ 已选择相对基准年份: {ref_year_input}")
            print(f"   包含的日期段: {ref_target_dates}")
        else:
            print(f"⚠️ 未找到包含 '{ref_year_input}' 的数据，将使用原始几何基线模式。")

    # 1) 生成断面
    bl_len = baseline_utm.length
    n_trans = max(int(math.floor(bl_len / transect_spacing)), 1)
    t_pos = np.linspace(0, bl_len, n_trans + 1)
    t_pts = [baseline_utm.interpolate(d) for d in t_pos]
    normal = unit_normal_at_point(baseline_utm) * dir_choice

    # 3) 计算交点
    results = {}
    per_transect_records = {}

    print("正在计算断面交点...")
    for ti, (pos, pt) in enumerate(zip(t_pos, t_pts)):
        tid = f"transect_{ti + 1}"
        bx, by = pt.x, pt.y
        p1 = (bx + normal[0] * transect_half_length, by + normal[1] * transect_half_length)
        p2 = (bx - normal[0] * transect_half_length, by - normal[1] * transect_half_length)
        t_line = LineString([p1, p2])

        per_transect_records[tid] = {'pos': pos, 'base_xy': (bx, by)}

        date_dists = {}
        for date in unique_dates:
            # 筛选日期
            subset = shore_gdf_utm[shore_gdf_utm['date_str'] == date]
            best_dist = np.nan
            best_pt = None

            # 可能有多个fragment，遍历
            for _, row in subset.iterrows():
                inter = t_line.intersection(row.geometry)
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue

                try:
                    inter = t_line.intersection(geom)
                except Exception:
                    continue

                if inter is None or inter.is_empty:
                    continue

                if inter.geom_type == 'Point':
                    cands = [inter]
                elif inter.geom_type == 'MultiPoint':
                    cands = list(inter.geoms)
                elif 'LineString' in inter.geom_type:
                    cands = [Point(c) for g in (inter.geoms if hasattr(inter, 'geoms') else [inter]) for c in g.coords]
                else:
                    cands = []

                for p in cands:
                    vx, vy = p.x - bx, p.y - by
                    proj_len = vx * normal[0] + vy * normal[1]
                    if abs(proj_len) > transect_half_length + 1.0: continue
                    proj_p, _ = project_point_onto_line(baseline_utm, p)
                    d = p.distance(proj_p)
                    if math.isnan(best_dist) or d < best_dist:
                        best_dist = d
                        best_pt = p
            date_dists[date] = float(best_dist) if best_pt else np.nan
        results[tid] = date_dists

    # 4) 转 DataFrame
    rows = []
    for tid, dists in results.items():
        pos = per_transect_records[tid]['pos']
        for date, d in dists.items():
            rows.append({'transect_id': tid, 'baseline_pos_m': pos, 'date': date, 'distance_m': d})
    df = pd.DataFrame(rows)
    df['date_dt'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values(['transect_id', 'date_dt'])

    # 5) 插值
    if interp:
        print("执行插值补全...")
        df_out = []
        for tid, grp in df.groupby('transect_id'):
            s = grp.set_index('date_dt').sort_index()
            v = s['distance_m'].interpolate(method='time', limit_direction='both')
            s = s.assign(distance_m_used=v.values).reset_index()
            df_out.append(s)
        df = pd.concat(df_out, ignore_index=True)
    else:
        df['distance_m_used'] = df['distance_m']

    # ================= 修改：计算相对距离（支持多日期基准） =================
    df['plot_distance'] = df['distance_m_used']

    if ref_target_dates:
        print(f"正在转换数据：以 {ref_year_input} 年的数据（共{len(ref_target_dates)}个日期）为混合零点...")

        # 1. 提取所有属于该年份的记录
        ref_df = df[df['date'].isin(ref_target_dates)]

        # 2. 构建字典：{transect_id : distance}
        # 如果同一个断面在该年份对应了多个日期（理论上不应发生，除非数据重叠），取平均值
        ref_map_series = ref_df.groupby('transect_id')['distance_m_used'].mean()
        ref_map = ref_map_series.to_dict()

        def get_rel_dist(row):
            base_val = ref_map.get(row['transect_id'])
            # 如果基准年该断面没数据，或者当前断面没数据，则结果为NaN
            if base_val is None or pd.isna(base_val) or pd.isna(row['distance_m_used']):
                return np.nan
            return row['distance_m_used'] - base_val

        df['plot_distance'] = df.apply(get_rel_dist, axis=1)
        y_axis_label = f"相对 {ref_year_input} 距离 (m)"
    else:
        y_axis_label = "距离基线距离 (m)"

    # 保存
    csv_out = os.path.join(out_dir, "transects_distances.csv")
    df.to_csv(csv_out, index=False, encoding='utf-8-sig')
    pkl_out = os.path.join(out_dir, "transects_distances.pkl")
    with open(pkl_out, 'wb') as f:
        pickle.dump({'results': results, 'df': df}, f)
    print("数据已保存。")

    # ================= 8) 断面时序图 =================
    print("生成断面时序回归图...")
    transect_slopes = {}
    for tid, grp in df.groupby('transect_id'):
        g = grp.sort_values('date_dt')
        y = g['plot_distance'].astype(float).values
        valid = ~np.isnan(y)
        slope_val = np.nan

        if valid.sum() >= 2:
            times = g['date_dt']
            x_years = np.array([t.year + (t.timetuple().tm_yday / 365.25) for t in times[valid]])
            slope, intercept, r_value, _, _ = linregress(x_years, y[valid])
            slope_val = slope

            plt.figure(figsize=(7, 4))
            plt.scatter(times, y, c='tab:blue', label='观测值')
            tmin, tmax = times[valid].min(), times[valid].max()
            y1 = slope * (tmin.year + tmin.timetuple().tm_yday / 365.25) + intercept
            y2 = slope * (tmax.year + tmax.timetuple().tm_yday / 365.25) + intercept
            plt.plot([tmin, tmax], [y1, y2], 'r--', label=f'趋势: {slope:.3f} m/a')

            if ref_target_dates:
                plt.axhline(0, color='gray', linestyle=':', alpha=0.8, label=f'{ref_year_input}基准')

            plt.title(f"{tid} 时序变化 (R²={r_value ** 2:.2f})")
            plt.xlabel("时间")
            plt.ylabel(y_axis_label)
            plt.grid(True, ls='--', alpha=0.6)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{tid}_timeseries.png"), dpi=150)
            plt.close()
        transect_slopes[tid] = slope_val

    # ================= 9) 逐年变化量 (逻辑不变) =================
    print("生成逐年变化量图...")
    id_pattern = re.compile(r'(\d+)')

    def extract_id(s):
        m = id_pattern.search(s)
        return int(m.group(1)) if m else 0

    df['transect_num'] = df['transect_id'].apply(extract_id)
    df['year'] = df['date_dt'].dt.year
    pivot_year = df.pivot_table(index='transect_num', columns='year', values='distance_m_used', aggfunc='mean')

    full_index = np.arange(1, n_trans + 2)
    pivot_year = pivot_year.reindex(full_index)
    years_sorted = sorted([int(y) for y in pivot_year.columns if not pd.isna(y)])

    if len(years_sorted) >= 2:
        for i in range(len(years_sorted) - 1):
            y1 = years_sorted[i]
            y2 = years_sorted[i + 1]
            change = pivot_year[y2] - pivot_year[y1]
            plot_vals = change.fillna(0.0).values

            if change.dropna().size == 0:
                limit_val = 10.0
            else:
                limit_val = np.nanmax(np.abs(change.dropna().values))
                if limit_val == 0: limit_val = 10.0
            ylim_val = limit_val * 1.1

            plt.figure(figsize=(12, 5))
            ax = plt.gca()
            colors = np.where(plot_vals >= 0, '#1f77b4', '#d62728')
            ax.bar(full_index, plot_vals, color=colors, width=1.0, align='center')
            ax.axhline(0, color='black', linewidth=0.8)
            plt.title(f"逐年岸线变化量: {y1}-{y2} (正值:淤积 / 负值:侵蚀)", fontsize=14)
            plt.xlabel("断面点序号")
            plt.ylabel("年变化量 / m")
            plt.xlim(1, full_index.max())
            ax.set_xticks(np.arange(1, full_index.max() + 1, max(1, int(full_index.max() / 10))))
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
            plt.ylim(-ylim_val, ylim_val)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"Annual_Change_{y1}-{y2}.png"), dpi=300)
            plt.close()

    # ================= 10) 基线展开图 (修改逻辑) =================
    print(f"生成基线展开图（采样间距: {UNFOLD_SAMPLING_STEP}m）...")
    n_panels = int(math.ceil(bl_len / panel_width))
    date_list = sorted(unique_dates)
    norm = mcolors.Normalize(vmin=0, vmax=len(date_list) - 1)
    cmap = cm.get_cmap('viridis')

    shoreings_by_date = {}

    # 10.1 提取所有数据
    for date in unique_dates:
        subset = shore_gdf_utm[shore_gdf_utm['date_str'] == date]
        pts_along = []
        for _, row in subset.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            coords = densify_line(geom, spacing_m=UNFOLD_SAMPLING_STEP)
            if len(coords) == 0: continue
            for (x, y) in coords:
                pt = Point(x, y)
                proj_pt, along = project_point_onto_line(baseline_utm, pt)
                perp = pt.distance(proj_pt)
                if perp <= transect_half_length:
                    pts_along.append((along, perp))
        shoreings_by_date[date] = np.array(pts_along) if len(pts_along) > 0 else np.empty((0, 2))

    # 10.2 构建混合基准插值器
    ref_interpolator = None
    if ref_target_dates:
        # 将所有属于该年份的日期的数据点全部收集起来
        all_ref_pts_list = []
        for d in ref_target_dates:
            arr = shoreings_by_date.get(d)
            if arr is not None and arr.shape[0] > 0:
                all_ref_pts_list.append(arr)

        if all_ref_pts_list:
            # 合并为一个大的数组
            combined_ref = np.vstack(all_ref_pts_list)
            # 按 along 排序
            combined_ref = combined_ref[combined_ref[:, 0].argsort()]

            # 去重和插值
            ref_x = combined_ref[:, 0]
            ref_y = combined_ref[:, 1]

            # 简单的去重处理（如果有完全相同的x，保留第一个）
            _, u_idx = np.unique(ref_x, return_index=True)
            ref_x = ref_x[u_idx]
            ref_y = ref_y[u_idx]

            def get_ref_perp(x_vals):
                return np.interp(x_vals, ref_x, ref_y, left=np.nan, right=np.nan)

            ref_interpolator = get_ref_perp
            print("✅ 基准年份展开曲面构建完成。")

    for p in range(n_panels):
        a0 = p * panel_width
        a1 = min((p + 1) * panel_width, bl_len)
        plt.figure(figsize=(10, 4))

        if ref_target_dates:
            plt.autoscale(axis='y')
        else:
            plt.ylim(0, transect_half_length * 1.05)

        for i, date in enumerate(date_list):
            arr = shoreings_by_date.get(date)
            if arr is None or arr.shape[0] == 0: continue
            mask = (arr[:, 0] >= a0) & (arr[:, 0] <= a1)
            sel = arr[mask]
            if sel.shape[0] == 0: continue

            bins = np.round(sel[:, 0]).astype(int)
            unique_bins = np.unique(bins)
            xs, ys = [], []
            for b in unique_bins:
                group = sel[bins == b]
                best = group[np.argmin(group[:, 1])]

                curr_x = best[0]
                curr_y = best[1]
                final_y = curr_y

                # 如果启用相对基准，减去该位置混合基准面的距离
                if ref_interpolator:
                    ref_y_val = ref_interpolator(curr_x)
                    if not np.isnan(ref_y_val):
                        final_y = curr_y - ref_y_val
                    else:
                        continue

                xs.append(curr_x - a0)
                ys.append(final_y)

            plt.plot(xs, ys, color=cmap(norm(i)), linewidth=0.8)

        if ref_target_dates:
            plt.title(f"展开图 (相对于{ref_year_input})：{a0:.0f}m – {a1:.0f}m")
            plt.ylabel(f"相对 {ref_year_input} 距离 (m)")
            plt.axhline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        else:
            plt.title(f"基线展开图：{a0:.0f}m – {a1:.0f}m")
            plt.ylabel("垂向距离 (m)")

        plt.xlabel("沿基线距离 (m)")
        plt.grid(ls='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"baseline_unfold_panel_{p + 1}.png"), dpi=300)
        plt.close()

    # Legend (不变)
    fig, ax = plt.subplots(figsize=(6, 2))
    for i, date in enumerate(date_list):
        ax.plot([i, i + 1], [0, 0], color=cmap(norm(i)), linewidth=8)
    ax.set_xlim(0, len(date_list))
    ax.set_yticks([])
    ax.set_xlabel("时间演变", fontsize=10)
    step = max(1, len(date_list) // 5)
    ax.set_xticks(np.arange(0, len(date_list), step) + 0.5)
    ax.set_xticklabels([date_list[i] for i in range(0, len(date_list), step)], rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Legend_Time.png"), dpi=300)
    plt.close()

    # ================= 11) 侵蚀/淤积 Shapefile (不变) =================
    print("生成基于拟合趋势的侵蚀/淤积岸线 Shapefile...")
    transect_meta = df[['transect_id', 'baseline_pos_m']].drop_duplicates().sort_values('baseline_pos_m')
    t_ids = transect_meta['transect_id'].values
    t_pos = transect_meta['baseline_pos_m'].values
    erosion_segs = []
    accretion_segs = []

    for i in range(len(t_ids) - 1):
        tid = t_ids[i]
        pos_start = t_pos[i]
        pos_end = t_pos[i + 1]
        slope = transect_slopes.get(tid, np.nan)
        if np.isnan(slope) or slope == 0: continue
        p1 = baseline_utm.interpolate(pos_start)
        p2 = baseline_utm.interpolate(pos_end)
        segment = LineString([p1, p2])
        if slope < 0:
            erosion_segs.append(segment)
        elif slope > 0:
            accretion_segs.append(segment)

    out_data = []
    if erosion_segs:
        merged = linemerge(erosion_segs)
        if isinstance(merged, LineString): merged = MultiLineString([merged])
        out_data.append({'Type': 'Erosion', 'Trend': 'Negative', 'geometry': merged})
    if accretion_segs:
        merged = linemerge(accretion_segs)
        if isinstance(merged, LineString): merged = MultiLineString([merged])
        out_data.append({'Type': 'Accretion', 'Trend': 'Positive', 'geometry': merged})
    if out_data:
        gdf_res = gpd.GeoDataFrame(out_data, crs=gdf_utm.crs)
        save_path = os.path.join(out_dir, "Shoreline_Trend_Segments.shp")
        gdf_res.to_file(save_path, encoding='utf-8')
        print(f"✅ 基于拟合趋势的 Shapefile 已保存: {save_path}")

    print("✅ 全部完成。")


if __name__ == "__main__":
    main()