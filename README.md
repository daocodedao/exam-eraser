
# 不好使

# 环境设置
## 环境 
```
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt 
```



# 使用方法

    -将输入图片（格式为 jpg）按顺序放入输入文件夹（src 目录下提供了一个文件夹）。
    -在 src 目录下，运行 "python main.py input-images"（将 `input-images` 替换为你的输入文件夹名称）。
    -等待处理。在 CUDA RTX 4060 上大约需要 1 分钟。
    -输出为一个名为 "processed.pdf" 的 PDF 文件，图片按标准排序顺序合并。

Tools 目录下有一些我用来制作训练数据的工具。


    - `bbox_add_to_folders`：运行边界框算法并将结果保存到文件夹。
    - `labeller`：一个支持 3 分类的图形界面标注工具（最终在 ViT 模型中使用 2 分类）。
    - `augment`：数据增强工具，将训练数据扩充 5 倍。
    - `dataset_loader`：将图片按 80/20 的比例划分为测试集和训练集。

ViT 参数：

    Patches = 16x16
    Model/embedding/hidden dim = 16
    2 Transformer-Encoder Blocks
    4 Heads of attention
    Output dim = 2

ViT 架构:

    Linear embedding 线性嵌入
    Classification token 分类标记
    Positional encoding (sin/cos) 位置编码（正弦/余弦）

    Transformer-Encoder Blocks:
        LayerNorm 层归一化
        Multi Headed Self attention 多头自注意力机制
        LayerNorm 层归一化
        MLP 多层感知机
            Linear to 4*model_dim 线性变换至 4 倍模型维度
            GELU GELU 激活函数
            Linear back to model_dim  线性变换回模型维度

    Classification MLP 分类多层感知机:
        model_dim 模型维度 -> 2
        softmax

5890 Total Trainable Parameters 可训练参数总数
在约 3 万张训练图像上的准确率：97.8%
