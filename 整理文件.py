import os
import re
import pickle
import shutil
import warnings
from datetime import datetime
from osgeo import gdal, osr

warnings.filterwarnings("ignore", category=FutureWarning)


GEE_DOWNLOAD_FOLDER = r"E:/Sentional/DATA/江苏/连云港1/CoastSat_data"
OUTPUT_FOLDER = r"E:/Sentional/CoastSat-master/data/NARRA"
SITENAME = "NARRA"


RES_MAPPING = {
    "S2": "10m",
    "L8": "30m",
    "L9": "30m"
}

def organise_data():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    # 初始化 metadata（CoastSat 格式）
    metadata = {
        "S2": {"filenames": [], "dates": [], "epsg": [], "acc_georef": [], "cloud_cover": []},
        "L8": {"filenames": [], "dates": [], "epsg": [], "acc_georef": [], "cloud_cover": []},
        "L9": {"filenames": [], "dates": [], "epsg": [], "acc_georef": [], "cloud_cover": []}
    }

    for fname in os.listdir(GEE_DOWNLOAD_FOLDER):
        if not fname.endswith(".tif"):
            continue

        match = re.match(rf"{SITENAME}_(S2|L8|L9)_(\d{{8}})\.tif", fname)
        if not match:
            print(f"跳过未识别文件: {fname}")
            continue

        sat, date_str = match.groups()
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        res_folder = RES_MAPPING[sat]

        out_dir = os.path.join(OUTPUT_FOLDER, date_str, res_folder)
        os.makedirs(out_dir, exist_ok=True)

        src_path = os.path.join(GEE_DOWNLOAD_FOLDER, fname)
        dst_path = os.path.join(out_dir, fname)
        shutil.copy(src_path, dst_path)
        print(f"已整理 {fname} -> {dst_path}")

        ds = gdal.Open(src_path)
        epsg = 4326
        if ds:
            proj = ds.GetProjection()
            if proj:
                srs = osr.SpatialReference(wkt=proj)
                epsg_str = srs.GetAttrValue("AUTHORITY", 1)
                if epsg_str:
                    epsg = int(epsg_str)
        ds = None

        metadata[sat]["filenames"].append(dst_path)
        metadata[sat]["dates"].append(date_obj)
        metadata[sat]["epsg"].append(epsg)
        metadata[sat]["acc_georef"].append(int(res_folder[:-1]))
        metadata[sat]["cloud_cover"].append(0.1)  # TODO: 这里目前填 0.1，可以以后替换为真实值

    meta_dir = os.path.join(OUTPUT_FOLDER, "metadata")
    os.makedirs(meta_dir, exist_ok=True)
    meta_path = os.path.join(meta_dir, f"{SITENAME}_metadata.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f"\n数据整理完成！metadata 已保存到: {meta_path}")


if __name__ == "__main__":
    organise_data()
