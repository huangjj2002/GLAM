import pandas as pd
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ================= 配置区域 =================
# [自动合并] 上游脚本生成的 CSV 文件名
IMPLANT_CSV = './list_for_conversion_implant_debug.csv'      # 由 generate_implant_csv.py 生成 (Test Set)
UNIMPLANT_CSV = './list_for_conversion_no_implant_debug.csv' # 由 generate_unimplant_csv.py 生成 (Train Set)

# 备用输入
#INPUT_LIST_CSV_FALLBACK = 'list_for_conversion_debug.csv' 

# 路径设置
EMBED_ROOT = r"./new_dataset"  
OUTPUT_DIR = "new_embed_data"
FINAL_CSV_NAME = 'embed_data.csv'

# 进程数
NUM_WORKERS = max(1, os.cpu_count() - 2) 

# [核心设置] 强制 Resize 尺寸 (Width, Height)
# 对应 main.py 中的 img_size=[1520, 912] (H, W)，这里设为 (912, 1520)
TARGET_SIZE = (912, 1520) 
# ===========================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def convert_dicom_to_jpg(dicom_path, output_path):
    """
    读取 DICOM -> 应用 VOI -> 归一化 -> Resize -> 保存为 JPG
    """
    try:
        dcm = pydicom.dcmread(dicom_path)
        
        try:
            image = apply_voi_lut(dcm.pixel_array, dcm)
        except:
            image = dcm.pixel_array.astype(float)
            
        # 处理 MONOCHROME1
        if hasattr(dcm, 'PhotometricInterpretation') and dcm.PhotometricInterpretation == "MONOCHROME1":
            image = np.amax(image) - image
        
        # 归一化到 0-255
        image = image.astype(np.float32)
        img_min = image.min()
        img_max = image.max()
        
        if img_max - img_min != 0:
            image = (image - img_min) / (img_max - img_min) * 255.0
        else:
            image = np.zeros_like(image)
            
        image = image.astype(np.uint8)

        # [Resize]
        if TARGET_SIZE is not None:
            image = cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)

        # 保存为 JPG，质量设为 95 (默认是95，可调)
        cv2.imwrite(output_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return True
    except Exception as e:
        print(f"Error converting {dicom_path}: {e}") 
        return False

def process_single_row(row_data, embed_root, images_out_root):
    """
    处理单行数据：路径拼接、转换检查、更新 Row 信息
    """
    patient_id = str(row_data['patient_id'])
    image_id = str(row_data['image_id'])
    dicom_rel_path = str(row_data['dicom_path'])
    
    # 清理路径
    clean_rel_path = dicom_rel_path.strip("/").strip("\\")
    dicom_full_path = os.path.join(embed_root, clean_rel_path)
    
    # 路径兼容
    if not os.path.exists(dicom_full_path):
        dicom_full_path = dicom_full_path.replace("\\", "/")
        
    # 创建输出目录
    patient_dir = os.path.join(images_out_root, patient_id)
    os.makedirs(patient_dir, exist_ok=True)
    
    # --- [修正点 1] 强制将 image_id 的后缀改为 .jpg ---
    # 原始 CSV 中可能是 .png (如果上游脚本生成的是png)，这里统一修正
    base_name = os.path.splitext(image_id)[0]
    new_image_id = base_name
    
    target_path = os.path.join(patient_dir, f"{new_image_id}_resized.jpg") 

    success = False
    
    # 检查目标文件是否存在且有效
    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
        success = True
    elif os.path.exists(dicom_full_path):
        success = convert_dicom_to_jpg(dicom_full_path, target_path)
    
    if success:
        if 'dicom_path' in row_data:
            del row_data['dicom_path']
            
        # --- [修正点 2] 更新 CSV 里的 image_id ---
        row_data['image_id'] = new_image_id
        
        return row_data
    else:
        return None

def main():
    # --- [Step 1] 自动合并逻辑 ---
    df_list = []
    
    if os.path.exists(IMPLANT_CSV):
        print(f">>> 发现 Implant 列表 (Test Set): {IMPLANT_CSV}")
        df_list.append(pd.read_csv(IMPLANT_CSV))
    
    if os.path.exists(UNIMPLANT_CSV):
        print(f">>> 发现 Unimplant 列表 (Train Set): {UNIMPLANT_CSV}")
        df_list.append(pd.read_csv(UNIMPLANT_CSV))
        
    if len(df_list) > 0:
        print(">>> 正在合并 CSV 文件...")
        df = pd.concat(df_list, ignore_index=True)
        print(f">>> 合并完成，共 {len(df)} 条数据。")
    else:
        print(f">>> 未找到生成的 CSV 文件，尝试读取备用文件: {INPUT_LIST_CSV_FALLBACK}")
        if not os.path.exists(INPUT_LIST_CSV_FALLBACK):
             print(f"错误：找不到任何输入文件。请先运行生成脚本。")
             return
        df = pd.read_csv(INPUT_LIST_CSV_FALLBACK)
    
    # --- [Step 2] 准备转换 ---
    # 输出文件夹改为 images_jpg 以防混淆
    images_out_root = os.path.join(OUTPUT_DIR, "images_jpg")
    ensure_dir(images_out_root)
    
    print(f">>> 开始转换 {len(df)} 张图像...")
    print(f"    目标格式: JPG")
    print(f"    目标目录: {images_out_root}")
    print(f"    目标尺寸: {TARGET_SIZE}")
    
    valid_records = []
    
    # --- [Step 3] 多进程并行处理 ---
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [
            executor.submit(process_single_row, row.to_dict(), EMBED_ROOT, images_out_root) 
            for _, row in df.iterrows()
        ]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = future.result()
            if result is not None:
                valid_records.append(result)
    
    print("\n" + "="*30)
    print(f"处理完成！成功转换: {len(valid_records)} / {len(df)}")
    
    # --- [Step 4] 保存最终 CSV ---
    if len(valid_records) > 0:
        final_df = pd.DataFrame(valid_records)
        final_csv_path = os.path.join(OUTPUT_DIR, FINAL_CSV_NAME)
        final_df.to_csv(final_csv_path, index=False)
        print(f">>> 最终 CSV 已保存: {final_csv_path}")
        
        if 'split' in final_df.columns:
             print("\n>>> 数据集分布统计 (split):")
             print(final_df['split'].value_counts())
    else:
        print(">>> 未生成任何有效数据，请检查 DICOM 路径设置。")

if __name__ == "__main__":
    multiprocessing.freeze_support() 
    main()