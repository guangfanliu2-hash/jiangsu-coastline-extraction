import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal, osr
from pyproj import Transformer
import matplotlib.cm as cm

# -------------------------------
# 输入
# -------------------------------
image_path   = input("请输入底图影像路径（.tif）：").strip('"')
pkl_path     = input("请输入岸线.pkl路径：").strip('"')
save_dir     = input("请输入结果保存目录：").strip('"')

os.makedirs(save_dir, exist_ok=True)
out_file = os.path.join(save_dir, "shorelines_overlay.jpg")

# -------------------------------
# 读取底图
# -------------------------------
ds = gdal.Open(image_path)
if ds is None:
    raise FileNotFoundError(f"无法打开影像: {image_path}")

proj = osr.SpatialReference(wkt=ds.GetProjection())
epsg_img = int(proj.GetAttrValue("AUTHORITY",1)) if proj.GetAttrValue("AUTHORITY",1) else None
print(f"✅ 底图 EPSG: {epsg_img}")

# 波段映射（S2 用 B2/B3/B4 = 真彩色）
band_map = {}
for i in range(1, ds.RasterCount+1):
    desc = ds.GetRasterBand(i).GetDescription().upper()
    if "B2" in desc: band_map["B2"] = i
    elif "B3" in desc: band_map["B3"] = i
    elif "B4" in desc: band_map["B4"] = i

if not all(b in band_map for b in ["B2","B3","B4"]):
    raise RuntimeError("找不到 B2/B3/B4 波段，请检查数据")

B = ds.GetRasterBand(band_map["B2"]).ReadAsArray().astype(float)
G = ds.GetRasterBand(band_map["B3"]).ReadAsArray().astype(float)
R = ds.GetRasterBand(band_map["B4"]).ReadAsArray().astype(float)

RGB = np.dstack((R, G, B))
RGB = (RGB - np.percentile(RGB, 2)) / (np.percentile(RGB, 98) - np.percentile(RGB, 2))
RGB = np.clip(RGB, 0, 1)

gt = ds.GetGeoTransform()
extent = [
    gt[0],
    gt[0] + ds.RasterXSize * gt[1],
    gt[3] + ds.RasterYSize * gt[5],
    gt[3]
]

# -------------------------------
# 读取岸线数据
# -------------------------------
with open(pkl_path, "rb") as f:
    output = pickle.load(f)

dates = output["dates"]
shorelines = output["shorelines"]

# 假设岸线投影是 settings['output_epsg']（之前你用的是 28356）
epsg_sl = 28356
print(f"✅ 岸线 EPSG 假定为: {epsg_sl}")

# -------------------------------
# 绘制
# -------------------------------
fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(RGB, extent=extent, origin="upper")

cmap = cm.get_cmap("tab20", len(dates))

for i, (date, sl) in enumerate(zip(dates, shorelines)):
    if sl is None or len(sl) == 0:
        continue

    # 转换到底图 EPSG
    if epsg_sl is not None and epsg_img is not None and epsg_sl != epsg_img:
        transformer = Transformer.from_crs(epsg_sl, epsg_img, always_xy=True)
        x_t, y_t = transformer.transform(sl[:,0], sl[:,1])
        sl_plot = np.c_[x_t, y_t]
    else:
        sl_plot = sl

    ax.plot(sl_plot[:,0], sl_plot[:,1], color=cmap(i), linewidth=1.2, label=date.strftime("%Y-%m-%d"))

ax.axis("off")
ax.legend(loc="upper right", fontsize=8, frameon=True)

plt.tight_layout()
plt.savefig(out_file, dpi=300, bbox_inches="tight", pad_inches=0.1)
plt.close()
print(f"🎉 已生成结果图: {out_file}")
