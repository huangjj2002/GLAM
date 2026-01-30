# GLAM模型

## 安装 

### 使用虚拟环境安装环境和依赖

1. **安装环境**
   ```bash
    conda env create -f environment.yml
    pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
   ```
2. **代码运行**  
  模型需要在EMBED数据集上对VLM进行预训练，得到一个用于进行特征提取的骨干网络。[具体训练步骤参考原项目](https://github.com/XYPB/GLAM?tab=readme-ov-file#a-embed)。
  得到预训练模型后进行使用tools下的generate_implant以及generate_unimplant对EMBED的csv文件进行处理，然后使用embed脚本对embed数据进行处理

4. **运行命令**  
   对模型预训练参考原项目，使用EMBED数据集进行下游训练用以进行cancer分类参考run_5fold_train.sh，在训练前要修改dataset\constants_val.py中的路径信息。训练时参考test_5fold.sh。
