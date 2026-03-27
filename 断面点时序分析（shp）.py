import os, sys, json
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point, MultiLineString
from shapely.ops import split
from tkinter import Tk, filedialog
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from pyproj import Transformer, CRS
from scipy.stats import linregress
from osgeo import gdal
import pickle
from datetime import datetime


from matplotlib import rcParams
def set_chinese_font():
    candidates = ["Microsoft YaHei", "SimHei", "WenQuanYi Zen Hei", "Heiti SC", "STHeiti"]
    for f in candidates:
        try:
            rcParams['font.sans-serif'] = [f]
            rcParams['axes.unicode_minus'] = False
            # 测试绘一个中文
            plt.figure()
            plt.text(0.1, 0.1, "中文", fontsize=6)
            plt.close()
            return
        except Exception:
            continue
    rcParams['font.sans-serif'] = ['DejaVu Sans']
    rcParams['axes.unicode_minus'] = False
    print("警告：未找到常见中文字体，图中中文可能显示为方块。")

set_chinese_font()

# -------------------------
# 弹窗选择文件 / 目录
# -------------------------
Tk().withdraw()
shp_path = filedialog.askopenfilename(title="请选择岸线 Shapefile（.shp）", filetypes=[("Shapefile", "*.shp")])
if not shp_path:
    print("未选择 shapefile，退出。"); sys.exit(1)

tif_path = filedialog.askopenfilename(title="请选择底图 GeoTIFF（用于绘图）", filetypes=[("GeoTIFF", "*.tif"), ("GeoTIFF", "*.tiff")]
)
if not tif_path:
    print("未选择底图，退出。"); sys.exit(1)

out_dir = filedialog.askdirectory(title="请选择结果保存目录")
if not out_dir:
    print("未选择输出目录，退出。"); sys.exit(1)

os.makedirs(out_dir, exist_ok=True)

# -------------------------
# 读取 shapefile
# -------------------------
gdf = gpd.read_file(shp_path)
print(f"已读取 shapefile：{shp_path}，要素数：{len(gdf)}，CRS：{gdf.crs}")
# 时间字段自动识别（若字段名非 'date'，可在此处手动修改）
time_field = None
for cand in ['date','Date','DATE','time','Time','timestamp']:
    if cand in gdf.columns:
        time_field = cand
        break
if time_field is None:
    # 如果没有 date 字段，用索引作为时间标识（不推荐）
    time_field = '__time_index__'
    gdf[time_field] = [f"feature_{i}" for i in range(len(gdf))]
    print("未检测到常用时间字段，已用索引作为时间标识。")
else:
    # 统一把 time_field 转为 datetime（若可能）
    try:
        gdf['__date_dt__'] = pd.to_datetime(gdf[time_field])
    except Exception:
        gdf['__date_dt__'] = gdf[time_field]  # 非日期字符串也保留
    print(f"时间字段：{time_field}")

# -------------------------
# 读取底图 GeoTIFF（按 3,2,1 -> R,G,B）
# -------------------------
ds = gdal.Open(tif_path, gdal.GA_ReadOnly)
if ds is None:
    raise FileNotFoundError(f"无法打开底图：{tif_path}")

# 读取波段（注意索引：gdal 波段从 1 开始）
def read_band_safe(ds, idx):
    if idx <= ds.RasterCount:
        arr = ds.GetRasterBand(idx).ReadAsArray().astype(np.float32)
        return arr
    else:
        return None

# 按你指定的顺序 3,2,1 （即文件中第3波段是 R）
R = read_band_safe(ds, 3)
G = read_band_safe(ds, 2)
B = read_band_safe(ds, 1)
if R is None or G is None or B is None:
    # 若文件通道不足，退回到前 3 波段
    print("警告：影像波段少于3，退回使用前3个波段")
    R = read_band_safe(ds, 1); G = read_band_safe(ds, 2); B = read_band_safe(ds, 3)

# 2% - 98% 线性拉伸用于显示
def stretch_2_98(arr):
    a = arr[~np.isnan(arr)]
    if a.size == 0:
        return np.zeros_like(arr)
    p2, p98 = np.percentile(a, (2,98))
    if p98 == p2:
        p2, p98 = a.min(), a.max()
    out = (arr - p2) / (p98 - p2 + 1e-12)
    out = np.clip(out, 0.0, 1.0)
    return out

R_s = stretch_2_98(R)
G_s = stretch_2_98(G)
B_s = stretch_2_98(B)
rgb = np.dstack([R_s, G_s, B_s])

# 影像变换参数（用于 imshow extent）
gt = ds.GetGeoTransform()
x_min = gt[0]; x_res = gt[1]; y_res = gt[5]; y_max = gt[3]
x_max = x_min + ds.RasterXSize * x_res
y_min = y_max + ds.RasterYSize * y_res
origin = 'upper' if y_res < 0 else 'lower'
extent = (x_min, x_max, y_min, y_max)

# 影像 EPSG（若无则为 None）
def ds_epsg(dataset):
    proj = dataset.GetProjection()
    if not proj:
        return None
    srs = CRS.from_wkt(proj)
    try:
        return srs.to_epsg()
    except:
        return None

img_epsg = ds_epsg(ds)
print(f"底图 EPSG：{img_epsg}")

# -------------------------
# 显示底图并叠加岸线（转换到图像 CRS 用于显示）
# -------------------------
# 转换岸线到图像 CRS（若两者 CRS 不同）
if gdf.crs is None:
    raise ValueError("输入 shapefile 没有定义 CRS，请先赋予正确 CRS（比如 EPSG:4326 或 UTM）。")

# 用于显示的 gdf_img = 岸线（显示 CRS 为 img_epsg）
if img_epsg is None:
    gdf_img = gdf.copy()
else:
    try:
        gdf_img = gdf.to_crs(epsg=img_epsg)
    except Exception as e:
        print("无法转换岸线到影像 CRS（用于显示），将尝试直接使用原始坐标绘图：", e)
        gdf_img = gdf.copy()

# 交互绘制 transects（在图像 CRS 上绘制）
print("\n请在弹出的窗口上用鼠标点击：每两点定义一条断面（从陆地→海洋方向），按 Enter 完成。")
fig, ax = plt.subplots(figsize=(10,8))
ax.imshow(rgb, extent=extent, origin=origin)
# 叠加岸线（白色）
for geom in gdf_img.geometry:
    try:
        if geom is None or geom.is_empty:
            continue
        if isinstance(geom, (LineString, MultiLineString)):
            x,y = geom.xy if isinstance(geom, LineString) else ([],[])
            if isinstance(geom, LineString):
                ax.plot(x, y, color='white', linewidth=0.8)
            else:
                for part in geom.geoms:
                    xx, yy = part.xy
                    ax.plot(xx, yy, color='white', linewidth=0.8)
    except Exception:
        continue

pts = []
transect_lines_display = []
def onclick(event):
    if event.inaxes != ax: return
    if event.xdata is None or event.ydata is None: return
    x,y = event.xdata, event.ydata
    pts.append((x,y))
    ax.plot(x,y,'ro',markersize=4)
    if len(pts) % 2 == 0:
        p0 = pts[-2]; p1 = pts[-1]
        ln, = ax.plot([p0[0],p1[0]],[p0[1],p1[1]], color='yellow', linewidth=1.6)
        transect_lines_display.append(ln)
    fig.canvas.draw()

def onkey(event):
    if event.key in ('enter','return'):
        plt.close(fig)

cid1 = fig.canvas.mpl_connect('button_press_event', onclick)
cid2 = fig.canvas.mpl_connect('key_press_event', onkey)
plt.title("交互绘制断面：每两点定义一条断面（陆地→海洋）。按 Enter 完成。", fontsize=12)
plt.show()

if len(pts) < 2:
    print("未绘制断面，退出。"); sys.exit(1)

# 组成 transects（图像 CRS 坐标）
transects_img = {}
for i in range(0, len(pts)-1, 2):
    a = np.array(pts[i]); b = np.array(pts[i+1])
    name = f"断面_{i//2+1}"
    transects_img[name] = np.vstack([a,b])

print(f"共绘制 {len(transects_img)} 条断面（图像 CRS）")

# -------------------------
# 为距离计算准备：统一投影到米（UTM）
# 逻辑：
# - 如果 shapefile 已是投影坐标（非地理 CRS），优先使用 shapefile 的 CRS（假定单位为米）
# - 否则，自动为每个要素选择合适 UTM 带并把该要素投影到对应 UTM（确保米单位）
# 为简单起见：我们为**全部要素**选用一个共同的 UTM 带（基于 shapefile 全体质心）
# -------------------------
# 先得到一个统一的计算投影 EPSG（UTM）
def choose_utm_epsg_from_gdf(gdf_in):
    # 若已有投影且不是地理（单位为米），直接返回
    try:
        crs = gdf_in.crs
        if crs is not None and not CRS(crs).is_geographic:
            epsg = CRS(crs).to_epsg()
            if epsg is not None:
                return epsg
    except Exception:
        pass
    # 否则计算质心（经纬度）并选 UTM 带
    # 将 gdf 投到 EPSG:4326 以确保 lon/lat
    gdf4326 = gdf_in.to_crs(epsg=4326)
    centroid = gdf4326.unary_union.centroid
    lon, lat = centroid.x, centroid.y
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        epsg_utm = 32600 + zone
    else:
        epsg_utm = 32700 + zone
    return epsg_utm

calc_epsg = choose_utm_epsg_from_gdf(gdf)
print(f"用于距离计算的投影 EPSG（米制）：{calc_epsg}")

# 创建 transformer：
# - img -> calc_epsg （用于将用户在图像上绘制的 transects 投影到米）
# - gdf -> calc_epsg （用于将岸线投影到米）
img_to_calc = None
if img_epsg is None:
    # 若影像无 CRS，不能直接转换；警告并假设影像与 shapefile 相同 CRS
    print("警告：影像没有 CRS 信息；将假设影像坐标系与 shapefile 相同以便计算（请确认）。")
    img_to_calc = None
else:
    img_to_calc = Transformer.from_crs(img_epsg, calc_epsg, always_xy=True)

gdf_calc = gdf.to_crs(epsg=calc_epsg)  # 所有岸线在 calc_epsg（米）

# 把 transects_img（图像 CRS）投影到 calc_epsg（米）
transects_calc = {}
for name, arr in transects_img.items():
    if img_epsg is None:
        # 假设图像 CRS 与 shapefile 相同，先将图像坐标按 gdf.crs->calc_epsg 做转换
        if gdf.crs is None:
            raise ValueError("影像无 CRS，且 shapefile 也未定义 CRS，无法投影。")
        # 构造 transformer from display-coords (image assumed to be shapefile CRS) to calc_epsg
        transformer = Transformer.from_crs(gdf.crs, calc_epsg, always_xy=True)
        x_t, y_t = transformer.transform(arr[:,0], arr[:,1])
    else:
        # 一般情况：有 img_epsg
        x_t, y_t = img_to_calc.transform(arr[:,0], arr[:,1])
    transects_calc[name] = np.c_[x_t, y_t]

# -------------------------
# 为每个日期、每条断面求交：并计算沿断面起点到交点的距离（米）
# 算法：
# - 对每个日期，取该日期下的所有岸线（原始每条 LineString）
# - 将这些岸线投影到 calc_epsg（已完成 gdf_calc）
# - 对单个断面（LineString），计算与每条岸线的交点（可能有多个）
# - 对于得到的交点集合，依据用户绘制方向（断面从第一个点到第二个点），计算交点在断面上的投影位置 t（0..L）
# - 选择第一个正向交点（t>0 且最小）作为该日期的断面位置；如果没有正交点则记录 NaN
# - 同时将所有找到的交点保存用于可视化（但时序只保留最先出现的那个）
# -------------------------
# 组织岸线按日期：注意 shapefile 可能同一日期多条
# 将日期字段转换为 datetime.date（方便排序与分组）
def normalize_date_val(val):
    try:
        return pd.to_datetime(val).date()
    except Exception:
        # 如果字段已经是 datetime.date 或字符串非法，尽量直接返回
        if isinstance(val, (pd.Timestamp, datetime)):
            return pd.to_datetime(val).date()
        return val

gdf['__date_norm__'] = gdf[time_field].apply(normalize_date_val)
# 建立字典：date -> list of geometries (原始 CRS)
dates_unique = sorted(list(pd.unique(gdf['__date_norm__'])))
print(f"检测到 {len(dates_unique)} 个日期（可能包含非时间字符串）")

# mapping from date -> GeoSeries in calc_epsg
shorelines_by_date = {}
for d in dates_unique:
    subset = gdf[gdf['__date_norm__'] == d]
    if subset.empty: continue
    subset_proj = subset.to_crs(epsg=calc_epsg)
    shorelines_by_date[d] = list(subset_proj.geometry)

# 准备结果结构
# rows: index 时间（date），columns 为每个断面名，值为距离（米）
dates_sorted = sorted(shorelines_by_date.keys())
transect_names = list(transects_calc.keys())
results = {name: [] for name in transect_names}

# 保存所有交点（用于可视化/诊断）： dict[transect][date] = list of (x,y) points (米)
all_intersections = {name: {} for name in transect_names}

# 工具：计算点在断面上的沿线距离（从起点到投影点），以及投影参数 t（0..L）
def point_along_line_distance(pt, line_pts):
    # pt: (x,y), line_pts: Nx2 array, order from start->end
    # 计算 point 在线上的最近投影和沿线距离
    # parametric project: loop segments
    x, y = pt
    segs = np.diff(line_pts, axis=0)
    seg_starts = line_pts[:-1]
    total = 0.0
    best_t = None
    best_dist = None
    acc = 0.0
    for i, (s, v) in enumerate(zip(seg_starts, segs)):
        vx, vy = v
        wx = x - s[0]; wy = y - s[1]
        seg_len2 = vx*vx + vy*vy
        if seg_len2 == 0:
            proj_t_seg = 0.0
        else:
            proj_t_seg = (wx*vx + wy*vy) / seg_len2
        if proj_t_seg < 0:
            proj_x, proj_y = s[0], s[1]
        elif proj_t_seg > 1:
            proj_x, proj_y = s[0] + vx, s[1] + vy
        else:
            proj_x = s[0] + proj_t_seg * vx
            proj_y = s[1] + proj_t_seg * vy
        dist = np.hypot(x - proj_x, y - proj_y)
        # compute distance along line to projection point
        dist_along = acc + np.hypot(proj_x - s[0], proj_y - s[1])
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_t = dist_along
        acc += np.hypot(vx, vy)
    return best_t, best_dist

# 主循环：按日期计算交点并记录每个断面最先的正向交点（若没有则 NaN）
for d in dates_sorted:
    shore_geoms = shorelines_by_date.get(d, [])
    for tname in transect_names:
        tr_pts = transects_calc[tname]  # Nx2 with N=2
        # treat as line segment
        tr_line = LineString(tr_pts)
        tr_length = tr_line.length
        # collect all intersections
        ints = []
        for g in shore_geoms:
            try:
                inter = tr_line.intersection(g)
            except Exception:
                inter = None
            if inter is None or inter.is_empty:
                continue
            # intersection may be point, multi-point, line (overlap)
            if isinstance(inter, Point):
                ints.append((inter.x, inter.y))
            else:
                # MultiPoint or LineString intersection - extract representative points
                if hasattr(inter, 'geoms'):
                    for comp in inter.geoms:
                        if isinstance(comp, Point):
                            ints.append((comp.x, comp.y))
                        elif isinstance(comp, LineString):
                            # add line endpoints as intersection proxies
                            coords = list(comp.coords)
                            if coords:
                                ints.append(coords[0]); ints.append(coords[-1])
                elif isinstance(inter, LineString):
                    coords = list(inter.coords)
                    if coords:
                        ints.append(coords[0]); ints.append(coords[-1])
        # compute along-transect distances for all found intersection points
        pts_along = []
        for p in ints:
            t_along, perp_dist = point_along_line_distance(p, tr_pts)
            if t_along is None:
                continue
            # ensure within slight margin of transect extents
            if t_along >= -1e-6 and t_along <= tr_length + 1e-6:
                pts_along.append((t_along, p))
        # save all intersections (unsorted)
        all_intersections[tname][d] = [p for _,p in pts_along]
        # choose first positive (从起点到海洋方向，选择最小正 t)
        positive = [t for t,_ in [(ta,pp) for ta,pp in pts_along] if t >= 0]
        if len(positive) == 0:
            results[tname].append(np.nan)
        else:
            minpos = min(positive)
            results[tname].append(float(minpos))

# 结果 DataFrame：index 为日期, columns 为断面名
df = pd.DataFrame({k: results[k] for k in transect_names}, index=pd.to_datetime(dates_sorted))
df.index.name = "date"
csv_out = os.path.join(out_dir, "transects_timeseries.csv")
df.to_csv(csv_out, encoding='utf-8-sig')
print(f"已保存时序 CSV：{csv_out}")

# 保存 pkl（结构化保存）
pkl_out = os.path.join(out_dir, "transects_results.pkl")
with open(pkl_out, "wb") as f:
    pickle.dump({"dates": dates_sorted, "transects": transects_calc, "time_series": df, "all_intersections": all_intersections, "calc_epsg": calc_epsg}, f)
print(f"已保存 PKL：{pkl_out}")

# 保存 transects_geojson（在 calc_epsg 坐标系，单位米）
features = []
for name, arr in transects_calc.items():
    coords = [[float(x), float(y)] for x,y in arr]
    features.append({"type":"Feature", "properties":{"name":name}, "geometry":{"type":"LineString","coordinates":coords}})
geo = {"type":"FeatureCollection", "features":features}
with open(os.path.join(out_dir, "transects.geojson"), "w", encoding="utf-8") as f:
    json.dump(geo, f, ensure_ascii=False, indent=2)
print("已保存 transects.geojson")

# -------------------------
# 绘图与统计：每一断面单独输出图表
# - 主图：原始时序点（每个日期） + 线性拟合（年均速率）
# - 月度均值（若数据跨月）画一张图
# - 季度均值画一张图
# -------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

plots_dir = os.path.join(out_dir, "plots")
ensure_dir(plots_dir)

for tname in transect_names:
    series = df[tname].copy()
    # 转为 pandas.Series indexed by DatetimeIndex
    series.index = pd.to_datetime(series.index)
    # 描述性统计
    # 年均变化速率：用时间（年）作为自变量做线性回归（忽略 NaN）
    valid_mask = series.notna()
    if valid_mask.sum() >= 2:
        xyears = (series.index[valid_mask].to_series().map(lambda d: d.year + d.dayofyear/365.25) - (series.index[valid_mask][0].year + series.index[valid_mask][0].dayofyear/365.25)).values
        yvals = series[valid_mask].values
        slope, intercept, r_value, p_value, std_err = linregress(xyears, yvals)
        annual_rate = slope  # 单位：m / year
        R2 = r_value**2
    else:
        slope = intercept = r_value = p_value = std_err = np.nan
        annual_rate = np.nan
        R2 = np.nan

    # 主时序图（点 + 线性拟合）
    fig, ax = plt.subplots(figsize=(8,5), dpi=200)
    ax.plot(series.index, series.values, 'o-', label="断面位置（m）")
    if not np.isnan(slope):
        x_all = (series.index.map(lambda d: d.year + d.dayofyear/365.25) - (series.index[valid_mask][0].year + series.index[valid_mask][0].dayofyear/365.25)).values
        fit_y = intercept + slope * ( (series.index.map(lambda d: d.year + d.dayofyear/365.25) - (series.index[valid_mask][0].year + series.index[valid_mask][0].dayofyear/365.25)) )
        ax.plot(series.index, fit_y, 'r--', label=f"线性拟合: 速率={annual_rate:.3f} m/年, R²={R2:.3f}")
    ax.set_title(f"{tname} — 断面时序（单位：米）", fontsize=12)
    ax.set_xlabel("时间", fontsize=11)
    ax.set_ylabel("沿断面距离（m）", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fpath = os.path.join(plots_dir, f"{tname}_timeseries.png")
    plt.savefig(fpath, dpi=300)
    plt.close()

    # 月度均值图
    ser_month = series.resample('M').mean()
    fig, ax = plt.subplots(figsize=(8,5), dpi=200)
    ax.plot(ser_month.index, ser_month.values, 'o-', label="月度均值")
    ax.set_title(f"{tname} — 月度均值时序", fontsize=12)
    ax.set_xlabel("时间（按月）", fontsize=11)
    ax.set_ylabel("位置（m）", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fpath = os.path.join(plots_dir, f"{tname}_monthly_mean.png")
    plt.savefig(fpath, dpi=300)
    plt.close()

    # 季度均值图
    ser_q = series.resample('Q').mean()
    fig, ax = plt.subplots(figsize=(8,5), dpi=200)
    ax.plot(ser_q.index, ser_q.values, 'o-', label="季度均值")
    ax.set_title(f"{tname} — 季度均值时序", fontsize=12)
    ax.set_xlabel("时间（按季度）", fontsize=11)
    ax.set_ylabel("位置（m）", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fpath = os.path.join(plots_dir, f"{tname}_quarterly_mean.png")
    plt.savefig(fpath, dpi=300)
    plt.close()

    print(f"已生成 {tname} 的图像：时序 + 月度 + 季度（保存在 {plots_dir}）; 年均速率={annual_rate if not np.isnan(annual_rate) else 'nan'} m/年")

# 保存综合统计表（包含每断面的年均速率）
summary_rows = []
for tname in transect_names:
    series = df[tname].copy()
    valid_mask = series.notna()
    if valid_mask.sum() >= 2:
        xyears = (series.index[valid_mask].to_series().map(lambda d: d.year + d.dayofyear/365.25) - (series.index[valid_mask][0].year + series.index[valid_mask][0].dayofyear/365.25)).values
        yvals = series[valid_mask].values
        slope, intercept, r_value, p_value, std_err = linregress(xyears, yvals)
        annual_rate = slope
        R2 = r_value**2
    else:
        annual_rate = np.nan; R2 = np.nan
    summary_rows.append({"断面": tname, "年均变化速率(m/年)": annual_rate, "拟合R2": R2})

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(os.path.join(out_dir, "transects_summary_rates.csv"), encoding='utf-8-sig', index=False)
print("已保存断面年均速率汇总表（transects_summary_rates.csv）")

print("\n全部处理完成 ✅")
print(f"输出目录：{out_dir}")
