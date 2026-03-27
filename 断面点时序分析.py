import os
import pickle
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from tkinter import Tk, filedialog
from osgeo import gdal, osr
from pyproj import Transformer
import pandas as pd

from coastsat import SDS_transects

def get_epsg_from_dataset(ds):
    proj = ds.GetProjection()
    if not proj:
        return None
    srs = osr.SpatialReference(wkt=proj)
    try:
        epsg = int(srs.GetAttrValue("AUTHORITY",1))
        return epsg
    except:
        return None

def infer_epsg_from_coords(coords):
    # coords: Nx2
    xs = coords[:,0]
    ys = coords[:,1]
    # if within lon/lat bounds -> 4326
    if (xs.min() >= -180 and xs.max() <= 180 and ys.min() >= -90 and ys.max() <= 90):
        return 4326
    # otherwise assume projected (ask user)
    return None

def normalize_rgb_for_plot(rgb):
    # rgb as HxWx3
    arr = rgb.astype(np.float64)
    p2 = np.percentile(arr, 2)
    p98 = np.percentile(arr, 98)
    if p98 == p2:
        p98 = arr.max()
        p2 = arr.min()
    arr = (arr - p2) / (p98 - p2 + 1e-12)
    arr = np.clip(arr, 0.0, 1.0)
    return arr

Tk().withdraw()
print("请选择岸线数据 .pkl 文件")
pkl_file = filedialog.askopenfilename(title="选择岸线 .pkl 文件", filetypes=[("Pickle files","*.pkl")])
if not pkl_file:
    raise FileNotFoundError("没有选择岸线文件")

print("请选择底图影像 (RGB GeoTIFF)")
rgb_file = filedialog.askopenfilename(title="选择 RGB 影像", filetypes=[("GeoTIFF","*.tif *.tiff")])
if not rgb_file:
    raise FileNotFoundError("没有选择底图影像")

print("请选择结果保存目录")
save_dir = filedialog.askdirectory(title="选择结果保存目录")
if not save_dir:
    raise FileNotFoundError("没有选择保存路径")

# ----------------------------
# Load shorelines pkl
# ----------------------------
with open(pkl_file, "rb") as f:
    output = pickle.load(f)

# expected structure:
# output['dates'] -> list of datetimes
# output['shorelines'] -> list of numpy arrays (N x 2) in some EPSG
# sometimes keys differ; try few options:
if 'dates' not in output or 'shorelines' not in output:
    raise KeyError("pkl 内容不包含 'dates' 或 'shorelines' 键，无法继续")

dates = output['dates']
shorelines = output['shorelines']
print(f"✅ 已加载岸线数据，共 {len(dates)} 个日期")

# ----------------------------
# Load RGB image and its CRS/extent
# ----------------------------
ds = gdal.Open(rgb_file)
if ds is None:
    raise FileNotFoundError(f"无法打开影像: {rgb_file}")

# read first 3 bands (or available)
bands_avail = min(3, ds.RasterCount)
rgb_arr = np.dstack([ds.GetRasterBand(i+1).ReadAsArray() for i in range(bands_avail)])
rgb_arr = normalize_rgb_for_plot(rgb_arr)

gt = ds.GetGeoTransform()
x_min = gt[0]
x_res = gt[1]
y_res = gt[5]
y_max = gt[3]
x_max = x_min + ds.RasterXSize * x_res
y_min = y_max + ds.RasterYSize * y_res
# origin determination for imshow
origin = "upper" if y_res < 0 else "lower"
extent = (x_min, x_max, y_min, y_max)

rgb_epsg = get_epsg_from_dataset(ds)
print(f"✅ 底图 EPSG: {rgb_epsg}")

# ----------------------------
# Determine shorelines EPSG (try from pkl metadata or infer)
# ----------------------------
shore_epsg = None
# If pkl contains 'inputs' with 'output_epsg', use it
if isinstance(output.get('inputs', None), dict):
    shore_epsg = output['inputs'].get('output_epsg', None)

if shore_epsg is None:
    first_valid = None
    for sl in shorelines:
        if sl is not None and len(sl) > 0:
            first_valid = np.array(sl)
            break
    if first_valid is None:
        raise ValueError("没有在 pkl 中找到有效岸线数据")
    inferred = infer_epsg_from_coords(first_valid)
    if inferred is not None:
        shore_epsg = inferred
        print(f"✅ 推断岸线 EPSG 为: {shore_epsg} (经纬度)")
    else:
        # ask user to input EPSG (fallback)
        try:
            val = input("无法自动推断岸线 EPSG，请输入岸线 EPSG(例如 28356) 然后回车，或直接回车使用默认 28356: ").strip()
        except:
            val = ""
        shore_epsg = int(val) if val != "" else 28356
        print(f"使用岸线 EPSG: {shore_epsg}")

print(f"✅ 岸线 EPSG: {shore_epsg}")

# ----------------------------
# Project shorelines into image CRS for display
# ----------------------------
shorelines_imgcrs = []
if shore_epsg == rgb_epsg or rgb_epsg is None:
    # If rgb_epsg is None, assume shorelines are in same units as image (user provided)
    for sl in shorelines:
        shorelines_imgcrs.append(np.array(sl) if sl is not None else None)
else:
    transformer_to_img = Transformer.from_crs(shore_epsg, rgb_epsg, always_xy=True)
    for sl in shorelines:
        if sl is None or len(sl) == 0:
            shorelines_imgcrs.append(None)
        else:
            x_t, y_t = transformer_to_img.transform(sl[:,0], sl[:,1])
            shorelines_imgcrs.append(np.c_[x_t, y_t])

# ----------------------------
# Show image + shorelines and allow interactive drawing of transects
# ----------------------------
print("\n请在弹出的窗口上绘制 transects（每条由两点定义：陆地→海），绘制时会实时显示点与线；按 Enter/Return 完成并关闭窗口。")
fig, ax = plt.subplots(figsize=(10,8))
ax.imshow(rgb_arr, extent=extent, origin=origin)
# overlay shorelines (in image CRS)
for sl in shorelines_imgcrs:
    if sl is None or len(sl) == 0:
        continue
    ax.plot(sl[:,0], sl[:,1], color='white', linewidth=0.8)

pts = []  # store clicked points
transect_lines = []  # list of plotted line artists (for display)

# event handlers for live plotting
def onclick(event):
    if event.inaxes != ax:
        return
    if event.xdata is None or event.ydata is None:
        return
    x, y = event.xdata, event.ydata
    pts.append((x,y))
    ax.plot(x, y, 'ro', markersize=4)
    # if this completes a pair, draw a transect line
    if len(pts) % 2 == 0:
        p0 = pts[-2]; p1 = pts[-1]
        ln, = ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color='yellow', linewidth=1.6)
        transect_lines.append(ln)
    fig.canvas.draw()

def onkey(event):
    # finish drawing when Enter/Return pressed
    if event.key in ('enter','return'):
        plt.close(fig)

cid_click = fig.canvas.mpl_connect('button_press_event', onclick)
cid_key = fig.canvas.mpl_connect('key_press_event', onkey)

plt.title("交互绘制 transects：点击两点定义一条，按 Enter 完成")
plt.show()  # blocks until closed via onkey

# after window closed, pts contains sequence of clicked points
if len(pts) < 2:
    raise ValueError("没有绘制 transects，程序退出")

# build transects dict in IMAGE CRS coordinates
transects_img = {}
for i in range(0, len(pts)-1, 2):
    a = np.array(pts[i])
    b = np.array(pts[i+1])
    name = f"Transect_{i//2+1}"
    transects_img[name] = np.vstack([a,b])

print(f"✅ 共绘制 {len(transects_img)} 条 transects（image CRS）")

# ----------------------------
# Convert transects from image CRS -> shorelines CRS (for compute_intersection_QC)
# ----------------------------
if rgb_epsg is None:
    # assume same as shore_epsg
    transects_proj = transects_img.copy()
else:
    transformer_to_shore = Transformer.from_crs(rgb_epsg, shore_epsg, always_xy=True)
    transects_proj = {}
    for k,v in transects_img.items():
        x_t, y_t = transformer_to_shore.transform(v[:,0], v[:,1])
        transects_proj[k] = np.c_[x_t, y_t]

# ----------------------------
# Call CoastSat QC intersection function
# ----------------------------
settings_transects = {
    'along_dist': 25,
    'min_points': 3,
    'max_std': 15,
    'max_range': 30,
    'min_chainage': -100,
    'multiple_inter': 'auto',
    'auto_prc': 0.1,
}

print("\n正在计算 transect 交点（质量控制模式）...")
# compute_intersection_QC expects: output (shoreline dict like coastSat), transects (dict of 2pt arrays), settings_transects
# Our 'output' pkl should match coastSat format (dates + shorelines in shore_epsg)
cross_distance = SDS_transects.compute_intersection_QC(output, transects_proj, settings_transects)
print("✅ 交点计算完成")

# ----------------------------
# Save CSV and time-series plot
# ----------------------------
df = pd.DataFrame(cross_distance, index=dates)
csv_out = os.path.join(save_dir, "shoreline_transects.csv")
df.to_csv(csv_out)
print(f"✅ 已保存交点 CSV: {csv_out}")

# plot one subplot per transect (vertical stack)
n_tr = len(cross_distance)
fig2, axs = plt.subplots(n_tr, 1, figsize=(10, 3*n_tr), sharex=True)
if n_tr == 1:
    axs = [axs]

for ax, (name, arr) in zip(axs, cross_distance.items()):
    ax.plot(dates, arr, marker='o', linestyle='-')
    ax.set_title(name)
    ax.set_ylabel("Shoreline position (m)")
    ax.grid(alpha=0.5, linestyle="--")

axs[-1].set_xlabel("Date")
plt.tight_layout()
png_out = os.path.join(save_dir, "transects_timeseries.png")
plt.savefig(png_out, dpi=300)
plt.close()
print(f"✅ 已保存时序图: {png_out}")

# also save transects as geojson (in shorelines CRS)
features = []
for name, seg in transects_proj.items():
    coords = [[float(x), float(y)] for x,y in seg]
    feat = {
        "type":"Feature",
        "properties":{"name":name},
        "geometry":{"type":"LineString","coordinates":coords}
    }
    features.append(feat)
geo = {"type":"FeatureCollection","features":features}
import json
geo_out = os.path.join(save_dir, "transects.geojson")
with open(geo_out, "w", encoding="utf-8") as f:
    json.dump(geo, f, ensure_ascii=False, indent=2)
print(f"✅ 已保存 transects GeoJSON: {geo_out}")

transects_output = {
    "dates": dates,                   # 时间序列
    "transects": cross_distance,      # transect 交点结果
    "settings": settings_transects    # transect QC 参数
}
pkl_out = os.path.join(save_dir, "shoreline_transects.pkl")
with open(pkl_out, "wb") as f:
    pickle.dump(transects_output, f)
print(f"✅ 已保存 transects PKL: {pkl_out}")

print("\n全部完成 ✔️")
