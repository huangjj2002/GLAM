import pandas as pd
import os
import shutil

# 配置你的 CSV 文件路径
# 根据你之前的描述，文件应该在这里
csv_files = [
    '/mnt/f/MammoCLIP/train.csv',
    '/mnt/f/MammoCLIP/test.csv',
    '/mnt/f/MammoCLIP/new_train_split.csv', # 如果你之前生成了这些文件
    '/mnt/f/MammoCLIP/new_test_split.csv'
]

def clean_image_id(img_id):
    """移除 image_id 中的后缀"""
    s = str(img_id).strip()
    # 移除常见的图片后缀
    for ext in ['.png', '.jpg', '.jpeg', '.dcm']:
        if s.lower().endswith(ext):
            s = s[:-len(ext)]
    return s

def process_csv(file_path):
    if not os.path.exists(file_path):
        print(f"跳过: 文件不存在 {file_path}")
        return

    print(f"正在处理: {file_path}")
    
    # 1. 备份原文件 (加上 .bak 后缀)
    backup_path = file_path + '.bak'
    if not os.path.exists(backup_path):
        shutil.copy(file_path, backup_path)
        print(f"  已备份至: {backup_path}")
    else:
        print(f"  备份已存在: {backup_path}")

    # 2. 读取 CSV
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"  读取失败: {e}")
        return

    # 3. 检查 image_id 列
    if 'image_id' not in df.columns:
        print("  错误: 找不到 'image_id' 列")
        return

    # 打印修改前的样子
    print(f"  修改前示例: {df['image_id'].head(3).tolist()}")

    # 4. 执行清洗
    # 强制转为字符串并去掉后缀
    df['image_id'] = df['image_id'].apply(clean_image_id)
    # 如果需要，也可以把该列转回整数类型 (可选，取决于你的图片文件名是否包含前导0)
    # df['image_id'] = pd.to_numeric(df['image_id'], errors='coerce').fillna(0).astype(int)

    # 打印修改后的样子
    print(f"  修改后示例: {df['image_id'].head(3).tolist()}")

    # 5. 保存覆盖原文件
    df.to_csv(file_path, index=False)
    print("  保存成功！\n")

if __name__ == "__main__":
    print("开始清洗 CSV 文件中的 image_id 后缀...\n")
    for csv_path in csv_files:
        process_csv(csv_path)
    print("所有文件处理完毕。")