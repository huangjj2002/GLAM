import pandas as pd
import os

# ================= 配置区域 =================
# 输入：原始 CSV 文件的路径 (请修改为您实际的 CSV 路径)
# 例如 RSNA 的训练数据
INPUT_CSV_PATH = "/mnt/f/MammoCLIP/new_train_split.csv"

# 输出：生成的 1% 数据 CSV 保存路径
OUTPUT_CSV_PATH = "/mnt/f/MammoCLIP/rsna_mammo_train_1pct.csv"

# 采样比例 (0.01 代表 1%)
RATIO = 0.01
# ===========================================

def main():
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"错误：找不到文件 {INPUT_CSV_PATH}")
        return

    print(f"正在读取 {INPUT_CSV_PATH} ...")
    df = pd.read_csv(INPUT_CSV_PATH)
    
    total_rows = len(df)
    print(f"原始数据量：{total_rows} 行")

    # 执行随机采样
    # random_state=42 保证每次运行结果一致
    df_sampled = df.sample(frac=RATIO, random_state=42)
    
    sampled_rows = len(df_sampled)
    print(f"采样后数据量 ({RATIO*100}%): {sampled_rows} 行")

    # 保存为新文件
    df_sampled.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"已保存至：{OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()