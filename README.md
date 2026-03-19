# GLAM模型

## 安装 

### 使用虚拟环境安装环境和依赖

1. **安装环境**
   ```bash
    conda env create -f environment.yml
    pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
   ```
2. **代码运行**  
   将预训练模型放入pretrain_model文件夹下，执行根目录下的run.sh脚本进行训练，在运行前修改run.sh脚本里面的变量名，例如csv文件路径以及png图片的根目录，使得模型能够正确的获取数据。
