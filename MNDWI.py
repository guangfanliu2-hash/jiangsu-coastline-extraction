import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os

# ===============================
# 用户设置
# ===============================
# 输入多波段 Sentinel-2 影像路径（从 GEE 下载的那个）
input_tif = r"E:\Sentional\MNDWI\NARRA_S2_20180208.tif"

# 输出路径
output_tif = os.path.splitext(input_tif)[0] + "_MNDWI.tif"

# Sentinel-2 波段对应：
# B2=蓝, B3=绿, B4=红, B8=近红外, B11=短波红外
GREEN_BAND = 2   # 注意：索引从 1 开始 (在 rasterio 中 band=2 表示第2波段)
SWIR1_BAND = 5   # B11 是第5个波段（GEE 导出时 ['B2','B3','B4','B8','B11']）

# ===============================
# 读取波段并计算
# ===============================
with rasterio.open(input_tif) as src:
    green = src.read(GREEN_BAND).astype('float32')
    swir = src.read(SWIR1_BAND).astype('float32')
    profile = src.profile

# 如果数值范围是 0–10000，可根据实际是否除以 10000
# 如果 ENVI 中不归一化，这里也不要除
# green /= 10000.0
# swir  /= 10000.0

# MNDWI 计算公式
mndwi = (green - swir) / (green + swir + 1e-6)

# 掩膜无效值
mndwi[(green + swir) == 0] = np.nan

# ===============================
# 结果显示
# ===============================
plt.figure(figsize=(8,6))
plt.imshow(mndwi, cmap='RdYlBu', vmin=-1, vmax=1)
plt.colorbar(label='MNDWI')
plt.title("MNDWI from Sentinel-2")
plt.axis('off')
plt.show()

# ===============================
# 保存为 GeoTIFF
# ===============================
profile.update(dtype=rasterio.float32, count=1)
with rasterio.open(output_tif, 'w', **profile) as dst:
    dst.write(mndwi.astype(rasterio.float32), 1)

print(f"✅ 已保存 MNDWI 结果到: {output_tif}")
