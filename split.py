import pandas as pd
import os
from sklearn.model_selection import train_test_split

# 1. 设置文件路径
# 根据你之前的配置，你的数据应该在这里
data_dir = '/mnt/f/MammoCLIP/'
source_csv_path = os.path.join(data_dir, 'train.csv')

# 2. 读取原始训练数据
print(f"正在读取文件: {source_csv_path} ...")
try:
    df = pd.read_csv(source_csv_path)
    print(f"读取成功，原始数据共 {len(df)} 条。")
except FileNotFoundError:
    print(f"错误: 找不到文件 {source_csv_path}，请检查路径是否正确。")
    exit()

# 3. 检查是否有 'cancer' 列（这是我们划分的依据）
if 'cancer' not in df.columns:
    print("错误: CSV文件中缺少 'cancer' 标签列，无法进行分层划分。")
    exit()
df=df[df['implant']==0]
# 4. 进行划分 (90% 训练, 10% 测试)
# stratify=df['cancer'] 确保训练集和测试集中患癌样本的比例是一致的，这对医疗数据非常重要
train_df, test_df = train_test_split(
    df, 
    test_size=0.10, 
    random_state=42, 
    stratify=df['cancer']
)

# 5. 保存新的 CSV 文件
new_train_path = os.path.join(data_dir, 'new_train_split.csv')
new_test_path = os.path.join(data_dir, 'new_test_split.csv')

train_df.to_csv(new_train_path, index=False)
test_df.to_csv(new_test_path, index=False)

print("-" * 30)
print("划分完成！")
print(f"训练集保存为: {new_train_path} (数量: {len(train_df)})")
print(f"测试集保存为: {new_test_path}  (数量: {len(test_df)})")
print("-" * 30)