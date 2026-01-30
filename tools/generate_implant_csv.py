import pandas as pd
import numpy as np
import os

# ================= 配置区域 =================
CLINICAL_FILE = '../EMBED_OpenData_clinical.csv'
METADATA_FILE = '../EMBED_OpenData_metadata.csv'
OUTPUT_CSV_FOR_CONVERSION = 'implant_list_for_conversion.csv'
# ===========================================

def main():
    print(">>> Step 1: 读取数据...")
    df_clin = pd.read_csv(CLINICAL_FILE, low_memory=False)
    df_meta = pd.read_csv(METADATA_FILE, low_memory=False)

    # -------------------------------------------------------
    # 2. 识别“含植入物”的检查 ID (acc_anon)
    # -------------------------------------------------------
    # 逻辑：只要 Metadata 里标记了 YES，或者 Clinical 里标记了 1，就记录这个 acc_anon
    meta_implant_accs = df_meta[df_meta['BreastImplantPresent'] == 'YES']['acc_anon'].unique()
    clin_implant_accs = df_clin[df_clin['implanfind'] == 1]['acc_anon'].unique()
    
    # 取并集：得到所有确定含有植入物的检查号
    all_implant_accs = set(meta_implant_accs) | set(clin_implant_accs)
    
    print(f"找到含植入物的独立检查数 (Studies): {len(all_implant_accs)}")

    if len(all_implant_accs) == 0:
        print("!!! 未在数据集中发现任何植入物标记，请检查原始 CSV 列名或内容。")
        return

    # -------------------------------------------------------
    # 3. 筛选元数据：仅保留这些含植入物检查的 2D 影像
    # -------------------------------------------------------
    df_meta_implant = df_meta[df_meta['acc_anon'].isin(all_implant_accs)].copy()
    
    if 'FinalImageType' in df_meta_implant.columns:
        df_meta_implant = df_meta_implant[df_meta_implant['FinalImageType'] == '2D']

    # -------------------------------------------------------
    # 4. 标签处理 (Clinical)
    # -------------------------------------------------------
    def determine_cancer_label(row):
        path = row['path_severity']
        asses = row['asses']
        if path in [0, 1]: return 1
        elif path in [2, 3, 4, 5, 6]: return 0
        elif pd.isna(path):
            # 这里的 asses 可能是字符串或数字，统一转为字符串判断
            if str(asses).strip().upper() in ['N', 'B', '1', '2']: return 0
        return -1 

    df_clin['Cancer'] = df_clin.apply(determine_cancer_label, axis=1)
    
    # 聚合标签：按病人、检查号、侧边取最大 Cancer 值
    clin_agg = df_clin[df_clin['Cancer'] != -1].groupby(['empi_anon', 'acc_anon', 'side'])['Cancer'].max().reset_index()
    clin_agg.rename(columns={'side': 'ImageLaterality'}, inplace=True)

    # -------------------------------------------------------
    # 5. 合并并输出
    # -------------------------------------------------------
    df_merged = pd.merge(df_meta_implant, clin_agg, 
                         on=['empi_anon', 'acc_anon', 'ImageLaterality'], 
                         how='inner')

    if df_merged.empty:
        print("!!! 合并后数据为空，可能是合并键(side/laterality)不匹配，或这些检查缺乏有效病理标签。")
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
    output_df['split'] = 'test'

    print(f"筛选完成！最终提取到含植入物的影像数: {len(output_df)}")
    print(f"正样本(Cancer=1)数量: {output_df['Cancer'].sum()}")
    
    output_df.to_csv(OUTPUT_CSV_FOR_CONVERSION, index=False)
    print(f">>> 结果已保存至: {OUTPUT_CSV_FOR_CONVERSION}")

if __name__ == "__main__":
    main()