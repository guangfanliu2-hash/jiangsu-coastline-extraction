import os
import pickle
import warnings
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure, filters
from osgeo import gdal, osr
from pyproj import Transformer
from collections import defaultdict
import datetime

warnings.filterwarnings("ignore")
plt.ion()

from coastsat import SDS_tools

sitename = "NARRA"
base_path = r"E:/Sentional/CoastSat-master/results"
data_path = r"E:/Sentional/CoastSat-master/data/NARRA"
meta_path = os.path.join(data_path, "metadata", f"{sitename}_metadata.pkl")

if not os.path.exists(meta_path):
    raise FileNotFoundError(f"找不到 metadata 文件: {meta_path}")

with open(meta_path, "rb") as f:
    metadata = pickle.load(f)

total_imgs = sum(len(metadata[sat]['filenames']) for sat in metadata.keys())
print(f"已加载 metadata，共 {total_imgs} 幅影像")

# 设置参数
results_path = os.path.join(base_path, sitename)
figures_path = os.path.join(results_path, "figures")
os.makedirs(figures_path, exist_ok=True)
water_detection_path = os.path.join(results_path, "water_detection")
os.makedirs(water_detection_path, exist_ok=True)

settings = {
    'output_epsg': 28356,
    'max_dist_ref': 100,
    'inputs': {'sitename': sitename, 'filepath': results_path}
}


def robust_vmin_vmax(arr):
    a = arr[np.isfinite(arr)]
    if a.size == 0: return -1, 1
    vmin, vmax = np.percentile(a, [2, 98])
    if vmin == vmax: vmax = vmin + 1e-3
    return vmin, vmax


def normalize_reference_to_output_epsg(ref_obj, output_epsg):
    if isinstance(ref_obj, dict):
        if 'coords_out' in ref_obj and 'epsg_out' in ref_obj:
            coords_out = np.asarray(ref_obj['coords_out'], float)
            epsg_out = int(ref_obj['epsg_out'])
            if epsg_out != int(output_epsg):
                tr = Transformer.from_crs(epsg_out, output_epsg, always_xy=True)
                x, y = tr.transform(coords_out[:, 0], coords_out[:, 1])
                coords_out = np.c_[x, y]
                epsg_out = int(output_epsg)
            return coords_out, epsg_out
    return None, int(output_epsg)


def save_geotiff(filename, arr, geo_transform, projection):
    driver = gdal.GetDriverByName('GTiff')
    rows, cols = arr.shape
    dataset = driver.Create(filename, cols, rows, 1, gdal.GDT_Float32)
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)
    band = dataset.GetRasterBand(1)
    band.WriteArray(arr)
    band.SetNoDataValue(np.nan)
    dataset.FlushCache()
    dataset = None


# ========== 参考岸线 ==========
ref_pkl = os.path.join(results_path, f"{sitename}_reference_shoreline.pkl")
reference_sl_out = None

if os.path.exists(ref_pkl):
    with open(ref_pkl, "rb") as f:
        ref_obj = pickle.load(f)
    reference_sl_out, _ = normalize_reference_to_output_epsg(ref_obj, settings['output_epsg'])

# ========== 数据收集与预处理 ==========
# 用于按年份存储MNDWI数据和地理信息，以便后续合成
mndwi_by_year = defaultdict(list)
georef_by_year = {}  # 假设同一年份内影像地理参考一致或只需取其一
proj_by_year = {}
epsg_by_year = {}

# 用于保存所有单期岸线结果（原始pkl）
shorelines_all, dates_all, satnames_all = [], [], []

print("\n开始处理影像...")

for satname, sat_meta in metadata.items():
    filepath_sat = SDS_tools.get_filepath(settings['inputs'], satname)
    filenames, dates, epsg_list = sat_meta['filenames'], sat_meta['dates'], sat_meta['epsg']

    for i, (fname, date_i) in enumerate(zip(filenames, dates)):
        fn = SDS_tools.get_filenames(fname, filepath_sat, satname)
        if isinstance(fn, list):
            fn = fn[0]

        # --- 使用GDAL读取影像 ---
        ds = gdal.Open(fn)
        if ds is None:
            print(f"无法打开影像 {fn}")
            continue

        # 读取数据
        im_ms = np.stack([ds.GetRasterBand(b + 1).ReadAsArray().astype(np.float32)
                          for b in range(ds.RasterCount)], axis=-1)
        georef = ds.GetGeoTransform()
        projection = ds.GetProjection()
        ds = None

        # 记录地理参考信息（按年记录，假设同年份区域一致）
        year = date_i.year
        if year not in georef_by_year:
            georef_by_year[year] = georef
            proj_by_year[year] = projection
            epsg_img = int(epsg_list[i]) if not isinstance(epsg_list, int) else int(epsg_list)
            epsg_by_year[year] = epsg_img

        im_nodata = np.isnan(im_ms[:, :, 0])

        # --- MNDWI计算 ---
        G = im_ms[:, :, 1].astype(float)
        SWIR = im_ms[:, :, 4].astype(float)
        # 避免除以0
        denominator = G + SWIR
        mndwi = np.divide(G - SWIR, denominator, out=np.full_like(G, np.nan), where=denominator != 0)

        # 处理空值：原始空值处设为NaN
        mndwi[im_nodata] = np.nan

        # 收集MNDWI数据用于合成
        mndwi_by_year[year].append(mndwi)

        # --- 单期岸线提取（仅计算数据，不保存图） ---
        # 仍然使用0作为阈值或简单逻辑，因为需求只说合成影像用OTSU
        water_mask_single = (mndwi > 0) & (~np.isnan(mndwi))
        contours = measure.find_contours(water_mask_single.astype(np.uint8), 0.5)

        if contours:
            contours.sort(key=lambda c: c.shape[0], reverse=True)
            contours = contours[:10]

            epsg_out = int(settings['output_epsg'])
            epsg_img_single = int(epsg_list[i]) if not isinstance(epsg_list, int) else int(epsg_list)

            for c in contours:
                shoreline_pix_rc = np.c_[c[:, 0], c[:, 1]]
                shoreline_world = SDS_tools.convert_pix2world(shoreline_pix_rc, np.array(georef))

                if epsg_img_single != epsg_out:
                    tr = Transformer.from_crs(epsg_img_single, epsg_out, always_xy=True)
                    x_t, y_t = tr.transform(shoreline_world[:, 0], shoreline_world[:, 1])
                    shoreline_out = np.c_[x_t, y_t]
                else:
                    shoreline_out = shoreline_world.copy()

                if reference_sl_out is not None and reference_sl_out.shape[0] >= 2:
                    dmat = np.sqrt(((shoreline_out[:, None, :] - reference_sl_out[None, :, :]) ** 2).sum(axis=2))
                    keep = (np.min(dmat, axis=1) <= settings['max_dist_ref'])
                    shoreline_out = shoreline_out[keep]
                    if shoreline_out.shape[0] < 2:
                        continue

                shorelines_all.append(shoreline_out)
                dates_all.append(date_i)
                satnames_all.append(satname)

# --- 保存原始所有日期的岸线数据 (.pkl) ---
output_all = {'dates': dates_all, 'shorelines': shorelines_all,
              'satname': satnames_all, 'inputs': settings['inputs']}
pkl_file_all = os.path.join(results_path, f"{sitename}_shorelines.pkl")
with open(pkl_file_all, "wb") as f: pickle.dump(output_all, f)
print(f"已保存所有日期岸线数据到: {pkl_file_all}")

# ========== 最大光谱指数合成与处理 ==========
composite_shorelines = []
composite_dates = []
composite_satnames = []  # 合成影像没有单一卫星名，可存为'Composite'

print("\n开始进行最大光谱指数合成及岸线提取...")

for year, mndwi_list in mndwi_by_year.items():
    print(f"处理年份: {year}，包含 {len(mndwi_list)} 幅影像")

    # 堆叠所有时相
    stack = np.stack(mndwi_list, axis=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mndwi_max = np.nanmax(stack, axis=0)

    # 保存合成后的MNDWI为TIFF
    out_tif_path = os.path.join(water_detection_path, f"{year}.tif")
    if year in georef_by_year and year in proj_by_year:
        save_geotiff(out_tif_path, mndwi_max, georef_by_year[year], proj_by_year[year])

    # --- OTSU 阈值分割 ---
    # 移除NaN值计算阈值
    valid_pixels = mndwi_max[~np.isnan(mndwi_max)]
    if valid_pixels.size == 0:
        print(f"{year} 年无有效像素，跳过")
        continue

    try:
        otsu_thresh = filters.threshold_otsu(valid_pixels)
        print(f"  OTSU 阈值: {otsu_thresh:.4f}")
    except Exception as e:
        print(f"  OTSU 计算失败，使用默认阈值 0: {e}")
        otsu_thresh = 0

    # 生成水体掩膜
    water_mask = (mndwi_max > otsu_thresh)

    # --- 提取合成影像岸线 ---
    contours = measure.find_contours(water_mask.astype(np.uint8), 0.5)
    if not contours:
        continue

    contours.sort(key=lambda c: c.shape[0], reverse=True)
    contours = contours[:10]

    plot_segments_pix = []
    epsg_out = int(settings['output_epsg'])
    epsg_img = epsg_by_year[year]
    georef = georef_by_year[year]

    # 设置一个代表性的日期对象（比如该年的1月1日，仅用于格式兼容）
    rep_date = datetime.datetime(year, 1, 1)

    for c in contours:
        shoreline_pix_rc = np.c_[c[:, 0], c[:, 1]]
        shoreline_world = SDS_tools.convert_pix2world(shoreline_pix_rc, np.array(georef))

        if epsg_img != epsg_out:
            tr = Transformer.from_crs(epsg_img, epsg_out, always_xy=True)
            x_t, y_t = tr.transform(shoreline_world[:, 0], shoreline_world[:, 1])
            shoreline_out = np.c_[x_t, y_t]
        else:
            shoreline_out = shoreline_world.copy()

        if reference_sl_out is not None and reference_sl_out.shape[0] >= 2:
            dmat = np.sqrt(((shoreline_out[:, None, :] - reference_sl_out[None, :, :]) ** 2).sum(axis=2))
            keep = (np.min(dmat, axis=1) <= settings['max_dist_ref'])
            shoreline_out = shoreline_out[keep]
            shoreline_pix_rc = shoreline_pix_rc[keep]
            if shoreline_out.shape[0] < 2:
                continue

        composite_shorelines.append(shoreline_out)
        composite_dates.append(rep_date)
        composite_satnames.append('Composite')
        plot_segments_pix.append(shoreline_pix_rc)

    # --- 保存合成影像叠加岸线图 ---
    if plot_segments_pix:
        vmin, vmax = robust_vmin_vmax(mndwi_max)
        plt.figure(figsize=(7, 6))
        # 使用原来的配色 RdYlBu 或者 gray，这里保留用户原习惯可能更倾向于看MNDWI值分布
        plt.imshow(mndwi_max, cmap="RdYlBu", origin='upper', vmin=vmin, vmax=vmax)
        plt.colorbar(label="MNDWI (Max Composite)")
        for seg_pix in plot_segments_pix:
            plt.plot(seg_pix[:, 1], seg_pix[:, 0], 'r-', linewidth=1)  # 调整颜色k-为黑、r-为红、c-为青、m-为品红、y-为黄
        plt.title(f"{sitename} {year} Composite Shoreline (Thresh={otsu_thresh:.2f})")
        plt.axis('off')
        out_overlay = os.path.join(figures_path, f"{sitename}_{year}_composite_shoreline.jpg")
        plt.savefig(out_overlay, dpi=220, bbox_inches='tight', pad_inches=0)
        plt.close()

# --- 保存合成岸线数据 (.pkl) ---
output_composite = {'dates': composite_dates, 'shorelines': composite_shorelines,
                    'satname': composite_satnames, 'inputs': settings['inputs']}
pkl_file_composite = os.path.join(results_path, f"{sitename}_composite_shorelines.pkl")
with open(pkl_file_composite, "wb") as f: pickle.dump(output_composite, f)
print(f"已保存合成岸线数据到: {pkl_file_composite}")


# --- 保存合成岸线 GeoJSON ---
def save_geojson(output, output_path, epsg_src):
    src = osr.SpatialReference()
    src.ImportFromEPSG(epsg_src)
    dst = osr.SpatialReference()
    dst.ImportFromEPSG(4326)
    xform = osr.CoordinateTransformation(src, dst)
    geojson = {"type": "FeatureCollection", "features": []}
    for date, sl in zip(output['dates'], output['shorelines']):
        if sl is None or len(sl) == 0: continue
        coords = [[*xform.TransformPoint(float(x), float(y))[:2]] for x, y in sl]
        # 使用年份作为日期标识
        date_str = date.strftime("%Y")
        geojson['features'].append({
            "type": "Feature", "properties": {"year": date_str},
            "geometry": {"type": "LineString", "coordinates": coords}
        })
    import json
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)


geojson_file_comp = os.path.join(results_path, f"{sitename}_composite_shorelines.geojson")
save_geojson(output_composite, geojson_file_comp, settings['output_epsg'])
print(f"已保存合成岸线 GeoJSON: {geojson_file_comp}")

print("\n全部完成！")