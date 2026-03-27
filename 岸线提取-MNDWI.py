import os
import pickle
import warnings
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
from osgeo import gdal, osr
from pyproj import Transformer

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
debug_path = os.path.join(results_path, "water_detection")
os.makedirs(debug_path, exist_ok=True)

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
                coords_out = np.c_[x, y];
                epsg_out = int(output_epsg)
            return coords_out, epsg_out
    return None, int(output_epsg)


# ========== 参考岸线 ==========
ref_pkl = os.path.join(results_path, f"{sitename}_reference_shoreline.pkl")
ref_geojson = os.path.join(results_path, f"{sitename}_reference_shoreline.geojson")
reference_sl_out = None

if os.path.exists(ref_pkl):
    with open(ref_pkl, "rb") as f:
        ref_obj = pickle.load(f)
    reference_sl_out, _ = normalize_reference_to_output_epsg(ref_obj, settings['output_epsg'])

# ========== 岸线提取 ==========
shorelines_all, dates_all, satnames_all = [], [], []
print("\n开始提取岸线...")

for satname, sat_meta in metadata.items():
    filepath_sat = SDS_tools.get_filepath(settings['inputs'], satname)
    filenames, dates, epsg_list = sat_meta['filenames'], sat_meta['dates'], sat_meta['epsg']

    for i, (fname, date_i) in enumerate(zip(filenames, dates)):
        fn = SDS_tools.get_filenames(fname, filepath_sat, satname)
        if isinstance(fn, list):
            fn = fn[0]  # 多波段文件情况

        # --- 使用GDAL读取影像 ---
        ds = gdal.Open(fn)
        if ds is None:
            print(f"无法打开影像 {fn}")
            continue
        im_ms = np.stack([ds.GetRasterBand(b + 1).ReadAsArray().astype(np.float32)
                          for b in range(ds.RasterCount)], axis=-1)
        georef = np.array(ds.GetGeoTransform())
        ds = None

        im_nodata = np.isnan(im_ms[:, :, 0])
        epsg_img = int(epsg_list[i]) if not isinstance(epsg_list, int) else int(epsg_list)

        # --- MNDWI计算 ---
        G = im_ms[:, :, 1].astype(float)
        SWIR = im_ms[:, :, 4].astype(float)
        mndwi = (G - SWIR) / (G + SWIR + 1e-6)
        mndwi[im_nodata] = np.nan
        water_mask = (mndwi > 0) & (~im_nodata)

        # --- 保存MNDWI彩图 ---
        date_str = date_i.strftime("%Y%m%d")
        vmin, vmax = robust_vmin_vmax(mndwi)
        plt.figure(figsize=(6, 5))
        im = plt.imshow(mndwi, cmap="RdYlBu", origin='upper', vmin=vmin, vmax=vmax)
        plt.colorbar(im, label="MNDWI")
        plt.title(f"{sitename} {satname} {date_str} MNDWI")
        out_ndwi = os.path.join(debug_path, f"{sitename}_{satname}_{date_str}_MNDWI.jpg")
        plt.savefig(out_ndwi, dpi=180);
        plt.close()

        # --- 提取岸线（多段） ---
        contours = measure.find_contours(water_mask.astype(np.uint8), 0.5)
        if not contours: continue
        contours.sort(key=lambda c: c.shape[0], reverse=True)
        contours = contours[:10]

        plot_segments_pix = []
        epsg_out = int(settings['output_epsg'])

        for c in contours:
            shoreline_pix_rc = np.c_[c[:, 0], c[:, 1]]
            shoreline_world = SDS_tools.convert_pix2world(shoreline_pix_rc, georef)
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

            shorelines_all.append(shoreline_out)
            dates_all.append(date_i)
            satnames_all.append(satname)
            plot_segments_pix.append(shoreline_pix_rc)

        if len(plot_segments_pix) == 0:
            continue

        # --- 保存叠加岸线图 ---
        plt.figure(figsize=(7, 6))
        plt.imshow(mndwi, cmap="gray", origin='upper', vmin=vmin, vmax=vmax)
        for seg_pix in plot_segments_pix:
            plt.plot(seg_pix[:, 1], seg_pix[:, 0], 'r-', linewidth=0.9)
        plt.axis('off')
        out_overlay = os.path.join(figures_path, f"{sitename}_{satname}_{date_str}_shoreline.jpg")
        plt.savefig(out_overlay, dpi=220, bbox_inches='tight', pad_inches=0)
        plt.close()

# --- 保存结果 ---
output = {'dates': dates_all, 'shorelines': shorelines_all,
          'satname': satnames_all, 'inputs': settings['inputs']}

pkl_file = os.path.join(results_path, f"{sitename}_shorelines.pkl")
with open(pkl_file, "wb") as f: pickle.dump(output, f)
print(f"已保存岸线数据到: {pkl_file}")


# --- 保存为GeoJSON ---
def save_geojson(output, output_path, epsg_src):
    src = osr.SpatialReference();
    src.ImportFromEPSG(epsg_src)
    dst = osr.SpatialReference();
    dst.ImportFromEPSG(4326)
    xform = osr.CoordinateTransformation(src, dst)
    geojson = {"type": "FeatureCollection", "features": []}
    for date, sl in zip(output['dates'], output['shorelines']):
        if sl is None or len(sl) == 0: continue
        coords = [[*xform.TransformPoint(float(x), float(y))[:2]] for x, y in sl]
        geojson['features'].append({
            "type": "Feature", "properties": {"date": date.strftime("%Y-%m-%d")},
            "geometry": {"type": "LineString", "coordinates": coords}
        })
    import json
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)


geojson_file = os.path.join(results_path, f"{sitename}_shorelines.geojson")
save_geojson(output, geojson_file, settings['output_epsg'])
print(f"已保存岸线 GeoJSON: {geojson_file}")

print("\n全部完成！")
