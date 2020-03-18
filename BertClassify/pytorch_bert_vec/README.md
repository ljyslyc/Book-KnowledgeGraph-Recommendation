# pytorch版Bert、Albert、Robert等模型的句向量生成
 基于[Transformers](https://github.com/huggingface/transformers)，[albert_pytorch](https://github.com/lonePatient/albert_pytorch),用来简化生成句向量的代码，方便后续进行文本相似度计算等。

## 依赖
见requirements.txt

## 使用
1. 下载中文[bert](https://github.com/ymcui/Chinese-BERT-wwm)、[albert](https://github.com/brightmart/albert_zh)的预训练模型。
2. 使用flask_api.py。

## 效果
直接运行flask_api.py，用cosine距离判断句子的相似度，可以看到相似句子的bert和albert直接输出的向量相余弦距离比较小。

## 参考
句向量输出代码主要参考了[Bert提取句子特征（pytorch_transformers）](https://blog.csdn.net/weixin_41519463/article/details/100863313)。

