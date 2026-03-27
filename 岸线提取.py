import os
import pickle
import warnings
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
from osgeo import osr
from pyproj import Transformer

warnings.filterwarnings("ignore")
plt.ion()

from coastsat import SDS_preprocess, SDS_tools

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
    'cloud_thresh': 0.5,
    'dist_clouds': 300,
    'output_epsg': 28356,
    'check_detection': True,
    'adjust_detection': True,
    'save_figure': True,
    'min_beach_area': 1000,
    'min_length_sl': 200,
    'cloud_mask_issue': False,
    'sand_color': 'default',
    'pan_off': False,
    's2cloudless_prob': 40,
    'max_dist_ref': 100,
    'inputs': {
        'sitename': sitename,
        'filepath': results_path,
    }
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
                x, y = tr.transform(coords_out[:,0], coords_out[:,1])
                coords_out = np.c_[x, y]; epsg_out = int(output_epsg)
            return coords_out, epsg_out
        elif 'coords' in ref_obj and 'epsg' in ref_obj:
            coords = np.asarray(ref_obj['coords'], float)
            epsg_in = int(ref_obj['epsg'])
            tr = Transformer.from_crs(epsg_in, output_epsg, always_xy=True)
            x, y = tr.transform(coords[:,0], coords[:,1])
            return np.c_[x, y], int(output_epsg)

    arr = np.asarray(ref_obj, float)
    looks_lonlat = (np.all((arr[:,0] >= -180) & (arr[:,0] <= 180)) and
                    np.all((arr[:,1] >= -90) & (arr[:,1] <= 90)))
    if looks_lonlat:
        tr = Transformer.from_crs(4326, output_epsg, always_xy=True)
        x, y = tr.transform(arr[:,0], arr[:,1])
        return np.c_[x, y], int(output_epsg)
    else:
        return arr, int(output_epsg)

# 参考岸线（手绘）
ref_pkl = os.path.join(results_path, f"{sitename}_reference_shoreline.pkl")
ref_geojson = os.path.join(results_path, f"{sitename}_reference_shoreline.geojson")

reference_sl_out = None

if os.path.exists(ref_pkl):
    print(f"已找到参考岸线: {ref_pkl}")
    with open(ref_pkl, "rb") as f:
        ref_obj = pickle.load(f)
    reference_sl_out, _ = normalize_reference_to_output_epsg(ref_obj, settings['output_epsg'])
else:
    print("\n没有找到参考岸线文件，现在开始手动数字化 (鼠标点击绘制, 回车结束)...")
    first_sat = list(metadata.keys())[0]
    filepath_sat = SDS_tools.get_filepath(settings['inputs'], first_sat)
    fn0 = SDS_tools.get_filenames(metadata[first_sat]['filenames'][0], filepath_sat, first_sat)
    im_ms0, georef0, *_ = SDS_preprocess.preprocess_single(
        fn0, first_sat, cloud_mask_issue=False, pan_off=False, s2cloudless_prob=settings['s2cloudless_prob']
    )

    plt.figure(figsize=(8,7))
    plt.imshow(im_ms0[:,:,2], cmap="gray", origin='upper')
    plt.title("点击绘制参考岸线 (回车结束)")
    pts = plt.ginput(n=-1, timeout=0)
    plt.close()

    if len(pts) >= 2:
        pts = np.array(pts, float)
        pts_rc = np.c_[pts[:,1], pts[:,0]]
        ref_world_first = SDS_tools.convert_pix2world(pts_rc, georef0)
        epsg_first = int(metadata[first_sat]['epsg'][0]) if isinstance(metadata[first_sat]['epsg'], (list, tuple)) \
                     else int(metadata[first_sat]['epsg'])
        tr_ref = Transformer.from_crs(epsg_first, settings['output_epsg'], always_xy=True)
        rx, ry = tr_ref.transform(ref_world_first[:,0], ref_world_first[:,1])
        reference_sl_out = np.c_[rx, ry]
        ref_obj = {'coords_out': reference_sl_out, 'epsg_out': int(settings['output_epsg'])}
        with open(ref_pkl, "wb") as f: pickle.dump(ref_obj, f)

        # GeoJSON (WGS84)
        src = osr.SpatialReference(); src.ImportFromEPSG(settings['output_epsg'])
        dst = osr.SpatialReference(); dst.ImportFromEPSG(4326)
        xform = osr.CoordinateTransformation(src, dst)
        coords = [[*xform.TransformPoint(float(x), float(y))[:2]] for x,y in reference_sl_out]
        import json
        geo = {"type":"FeatureCollection","features":[{
            "type":"Feature","properties":{"name":"reference_shoreline"},
            "geometry":{"type":"LineString","coordinates":coords}
        }]}
        with open(ref_geojson, "w", encoding="utf-8") as f:
            json.dump(geo, f, ensure_ascii=False, indent=2)
        print(f"参考岸线已保存: {ref_pkl}, {ref_geojson}")

# 提取岸线
shorelines_all, dates_all, satnames_all = [], [], []

print("\n开始提取岸线...")
for satname, sat_meta in metadata.items():
    filepath_sat = SDS_tools.get_filepath(settings['inputs'], satname)
    filenames, dates, epsg_list = sat_meta['filenames'], sat_meta['dates'], sat_meta['epsg']

    for i, (fname, date_i) in enumerate(zip(filenames, dates)):
        fn = SDS_tools.get_filenames(fname, filepath_sat, satname)
        try:
            im_ms, georef, cloud_mask, im_extra, im_QA, im_nodata = SDS_preprocess.preprocess_single(
                fn, satname, cloud_mask_issue=False, pan_off=False, s2cloudless_prob=settings['s2cloudless_prob']
            )
        except Exception as e:
            print(f"预处理失败 {date_i}: {e}"); continue
        if im_ms is None: continue

        # NDWI
        G, NIR = im_ms[:,:,1].astype(float), im_ms[:,:,3].astype(float)
        ndwi = (G - NIR) / (G + NIR + 1e-9)
        ndwi[im_nodata] = np.nan
        water_mask = (ndwi > 0) & (~im_nodata)

        # MNDWI
        # G, SWIR = im_ms[:,:,1].astype(float) * 10000, im_ms[:,:,4].astype(float) * 10000
        # mndwi = (G - SWIR) / (G + SWIR + 1e-6)
        # mndwi[im_nodata] = np.nan
        # water_mask = (mndwi > 0) & (~im_nodata)

        # 保存 NDWI 彩色图
        date_str = date_i.strftime("%Y%m%d")
        vmin, vmax = robust_vmin_vmax(ndwi)
        plt.figure(figsize=(6,5))
        im = plt.imshow(ndwi, cmap="RdYlBu", origin='upper', vmin=vmin, vmax=vmax)
        plt.colorbar(im, label="NDWI")
        plt.title(f"{sitename} {satname} {date_str} NDWI")
        out_ndvi = os.path.join(debug_path, f"{sitename}_{satname}_{date_str}_NDWI.jpg")
        plt.savefig(out_ndvi, dpi=180); plt.close()

        # 找岸线（提取最长的 N 条）
        contours = measure.find_contours(water_mask.astype(np.uint8), 0.5)
        if not contours:
            continue

        # 按长度排序，取前 N 条
        N = 10  # 可调节：提取最长的 2 或 3 段岸线
        contours.sort(key=lambda c: c.shape[0], reverse=True)
        contours = contours[:N]

        # 用于绘图：收集每段的像素坐标（用于在一张图中显示所有选中段）
        plot_segments_pix = []

        # 对每一段单独处理并单条保存（不合并）
        for c in contours:
            shoreline_pix_rc = np.c_[c[:,0], c[:,1]]
            shoreline_world = SDS_tools.convert_pix2world(shoreline_pix_rc, georef)

            # 转到 output_epsg
            epsg_img = int(epsg_list[i]) if not isinstance(epsg_list, int) else int(epsg_list)
            epsg_out = int(settings['output_epsg'])
            if epsg_img != epsg_out:
                tr = Transformer.from_crs(epsg_img, epsg_out, always_xy=True)
                x_t, y_t = tr.transform(shoreline_world[:,0], shoreline_world[:,1])
                shoreline_out = np.c_[x_t, y_t]
            else:
                shoreline_out = shoreline_world.copy()

            # 筛选（与参考岸线的距离）
            if reference_sl_out is not None and reference_sl_out.shape[0] >= 2:
                dmat = np.sqrt(((shoreline_out[:,None,:] - reference_sl_out[None,:,:])**2).sum(axis=2))
                keep = (np.min(dmat, axis=1) <= settings['max_dist_ref'])
                shoreline_out_filtered = shoreline_out[keep]
                shoreline_pix_rc_filtered = shoreline_pix_rc[keep]
                if shoreline_out_filtered.shape[0] < 2:
                    # 该段经过过滤后太短，跳过保存但不影响其它段
                    continue
                shoreline_out = shoreline_out_filtered
                shoreline_pix_rc = shoreline_pix_rc_filtered

            # 保存该段（单独作为一条记录）
            shorelines_all.append(shoreline_out)
            dates_all.append(date_i)
            satnames_all.append(satname)

            # 收集像素坐标以便画图（用原像素索引绘图）
            plot_segments_pix.append(shoreline_pix_rc)

        # 如果没有任何段被保存（例如都被参考线过滤掉），跳过绘图保存
        if len(plot_segments_pix) == 0:
            print(f"{date_str} 没有满足条件的岸线段（可能都被过滤），跳过绘图")
            continue

        # 黑白底图 + 红线（在一张图中画出该影像所有保留的段）
        plt.figure(figsize=(7,6))
        plt.imshow(ndwi, cmap="gray", origin='upper', vmin=vmin, vmax=vmax)
        for seg_pix in plot_segments_pix:
            plt.plot(seg_pix[:,1], seg_pix[:,0], 'r-', linewidth=0.9)
        plt.axis('off')
        out_overlay = os.path.join(figures_path, f"{sitename}_{satname}_{date_str}_shoreline.jpg")
        plt.savefig(out_overlay, dpi=220, bbox_inches='tight', pad_inches=0)
        plt.close()

# 保存结果
output = {'dates': dates_all,'shorelines': shorelines_all,'satname': satnames_all,'inputs': settings['inputs']}

pkl_file = os.path.join(results_path, f"{sitename}_shorelines.pkl")
with open(pkl_file, "wb") as f: pickle.dump(output, f)
print(f"已保存岸线数据到: {pkl_file}")

def save_geojson(output, output_path, epsg_src):
    src = osr.SpatialReference(); src.ImportFromEPSG(epsg_src)
    dst = osr.SpatialReference(); dst.ImportFromEPSG(4326)
    xform = osr.CoordinateTransformation(src, dst)
    geojson = {"type": "FeatureCollection", "features": []}
    for date, sl in zip(output['dates'], output['shorelines']):
        if sl is None or len(sl)==0: continue
        coords = [[*xform.TransformPoint(float(x), float(y))[:2]] for x,y in sl]
        geojson['features'].append({
            "type":"Feature","properties":{"date":date.strftime("%Y-%m-%d")},
            "geometry":{"type":"LineString","coordinates":coords}
        })
    import json
    with open(output_path,"w",encoding="utf-8") as f: json.dump(geojson,f,ensure_ascii=False,indent=2)

geojson_file = os.path.join(results_path, f"{sitename}_shorelines.geojson")
save_geojson(output, geojson_file, settings['output_epsg'])
print(f"已保存岸线 GeoJSON: {geojson_file}")

print("\n全部完成！")
