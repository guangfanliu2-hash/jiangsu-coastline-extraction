import os
import pickle
import shapefile
from tkinter import Tk, filedialog
from osgeo import osr
from pyproj import Transformer

# ------------------------------
# 交互式选择文件
# ------------------------------
Tk().withdraw()

print("请选择岸线 .pkl 文件")
pkl_file = filedialog.askopenfilename(
    title="选择岸线 .pkl 文件", filetypes=[("Pickle 文件","*.pkl")]
)
if not pkl_file:
    raise SystemExit("未选择文件，退出。")

print("请选择输出文件夹")
out_dir = filedialog.askdirectory(title="选择输出文件夹")
if not out_dir:
    raise SystemExit("未选择输出文件夹，退出。")

# ------------------------------
# 读取 pkl
# ------------------------------
with open(pkl_file, "rb") as f:
    data = pickle.load(f)

shorelines = data.get('shorelines', [])
dates = data.get('dates', [])
inputs = data.get('inputs', {})

# 获取原始 EPSG
if isinstance(inputs, dict) and 'output_epsg' in inputs:
    epsg_in = int(inputs['output_epsg'])
else:
    # 若未定义，默认假设为 UTM (28356)
    epsg_in = 28356

print(f"✅ 检测到原始坐标系 EPSG:{epsg_in}")
epsg_out = 4326  # 输出为WGS84经纬度
print(f"🌍 目标坐标系 EPSG:{epsg_out} (GCS_WGS_1984)")

# ------------------------------
# 坐标转换准备
# ------------------------------
transformer = Transformer.from_crs(epsg_in, epsg_out, always_xy=True)

# ------------------------------
# 写 shapefile
# ------------------------------
out_shp = os.path.join(out_dir, os.path.splitext(os.path.basename(pkl_file))[0] + "_WGS84.shp")

w = shapefile.Writer(out_shp)
w.autoBalance = 1
w.field("date", "C", 20)

for date, sl in zip(dates, shorelines):
    if sl is None or len(sl) == 0:
        continue

    # 投影转换
    x_t, y_t = transformer.transform(sl[:, 0], sl[:, 1])
    coords = [(float(x), float(y)) for x, y in zip(x_t, y_t)]

    # 写入线段
    w.line([coords])
    w.record(date.strftime("%Y-%m-%d"))

w.close()

# ------------------------------
# 写入投影文件（.prj）
# ------------------------------
srs = osr.SpatialReference()
srs.ImportFromEPSG(epsg_out)
srs.MorphToESRI()
prj_file = out_shp.replace(".shp", ".prj")

with open(prj_file, "w") as f:
    f.write(srs.ExportToWkt())

print(f"\n✅ 已导出 shapefile: {out_shp}")
print(f"📌 坐标系: EPSG:{epsg_out} (GCS_WGS_1984, 单位：度)")
print("✅ 可直接在 ArcGIS/QGIS 中与 GEE 下载的影像重叠显示。")
