import pandas as pd
import numpy as np
import os

# ================= 配置区域 =================
CLINICAL_FILE = '../EMBED_OpenData_clinical.csv'
METADATA_FILE = '../EMBED_OpenData_metadata.csv'
OUTPUT_CSV_FOR_CONVERSION = 'no_implant_list_for_conversion.csv'
# ===========================================

def main():
    print(">>> Step 1: 读取数据...")
    df_clin = pd.read_csv(CLINICAL_FILE, low_memory=False)
    df_meta = pd.read_csv(METADATA_FILE, low_memory=False)

    # -------------------------------------------------------
    # 2. 识别所有“含植入物”的黑名单 ID (acc_anon)
    # -------------------------------------------------------
    # 只要有任何一处标记了 YES 或 1，就放入黑名单
    meta_implant_accs = df_meta[df_meta['BreastImplantPresent'] == 'YES']['acc_anon'].unique()
    clin_implant_accs = df_clin[df_clin['implanfind'] == 1]['acc_anon'].unique()
    
    # 取并集：得到所有需要排除的检查号
    implant_blacklist = set(meta_implant_accs) | set(clin_implant_accs)
    
    print(f"识别到不含植入物的检查号（黑名单）共: {len(implant_blacklist)} 个")

    # -------------------------------------------------------
    # 3. 筛选元数据：排除黑名单中的检查，且只保留 2D
    # -------------------------------------------------------
    # 使用 ~ 符号表示“取反”，即不在黑名单里的
    df_meta_clean = df_meta[~df_meta['acc_anon'].isin(implant_blacklist)].copy()
    
    if 'FinalImageType' in df_meta_clean.columns:
        df_meta_clean = df_meta_clean[df_meta_clean['FinalImageType'] == '2D']

    # -------------------------------------------------------
    # 4. 标签处理 (Clinical)
    # -------------------------------------------------------
    def determine_cancer_label(row):
        path = row['path_severity']
        asses = row['asses']
        
        # 1. 病理确诊 -> 优先级最高
        if path in [0, 1]: return 1
        elif path in [2, 3, 4, 5, 6]: return 0
        
        # 2. 无病理 -> 信任明确阴性的 BI-RADS
        elif pd.isna(path):
            if str(asses).strip().upper() in ['N', 'B', '1', '2']: return 0
            
        return -1 

    df_clin['Cancer'] = df_clin.apply(determine_cancer_label, axis=1)
    
    # 聚合标签：按病人、检查号、侧边取最大 Cancer 值
    # 同样排除掉黑名单里的临床记录
    df_clin_clean = df_clin[~df_clin['acc_anon'].isin(implant_blacklist)]
    clin_agg = df_clin_clean[df_clin_clean['Cancer'] != -1].groupby(['empi_anon', 'acc_anon', 'side'])['Cancer'].max().reset_index()
    clin_agg.rename(columns={'side': 'ImageLaterality'}, inplace=True)

    # -------------------------------------------------------
    # 5. 合并并输出
    # -------------------------------------------------------
    df_merged = pd.merge(df_meta_clean, clin_agg, 
                         on=['empi_anon', 'acc_anon', 'ImageLaterality'], 
                         how='inner')

    if df_merged.empty:
        print("!!! 筛选后没有剩余数据，请检查原始文件内容。")
        return

    output_df = pd.DataFrame()
    output_df['patient_id'] = df_merged['empi_anon'].astype(str)
    output_df['study_id'] = df_merged['acc_anon'].astype(str) 
    output_df['image_id'] = df_merged['anon_dicom_path'].apply(
        lambda x: os.path.basename(str(x)).replace('.dcm', '.png')
    )
    output_df['dicom_path'] = df_merged['anon_dicom_path']
    output_df['view_position'] = df_merged.get('ViewPosition', 'UNKNOWN')
    output_df['laterality'] = df_merged['ImageLaterality']
    output_df['cancer'] = df_merged['Cancer'].astype(int)
    output_df['split'] = 'training'

    print(f"筛选完成！最终提取到【无植入物】的影像数: {len(output_df)}")
    print(f"正样本(Cancer=1)数量: {output_df['Cancer'].sum()}")
    print(f"独立检查数(Studies): {output_df['study_id'].nunique()}")
    
    output_df.to_csv(OUTPUT_CSV_FOR_CONVERSION, index=False)
    print(f">>> 结果已保存至: {OUTPUT_CSV_FOR_CONVERSION}")

if __name__ == "__main__":
    main()