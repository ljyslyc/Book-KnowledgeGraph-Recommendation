import numpy as np
import torch
from transformers import BertModel, BertConfig, BertTokenizer
from scipy.spatial.distance import cosine

from albert.albert_total import get_albert_total
from torch import nn

config_path = 'model/bert/bert_config.json'
model_path = 'model/bert/pytorch_model.bin'
vocab_path = 'model/bert/vocab.txt'

al_config_path = 'model/albert_tiny/bert_config.json'
al_model_path = 'model/albert_tiny/pytorch_model.bin'
al_vocab_path = 'model/albert_tiny/vocab.txt'


class BertTextNet(nn.Module):
    def __init__(self):
        """
        bert模型。
        """
        super(BertTextNet, self).__init__()
        modelConfig = BertConfig.from_pretrained(config_path)
        self.textExtractor = BertModel.from_pretrained(
            model_path, config=modelConfig)
        self.tokenizer = BertTokenizer.from_pretrained(vocab_path)

    def forward(self, tokens, segments, input_masks):
        output = self.textExtractor(tokens, token_type_ids=segments,
                                    attention_mask=input_masks)
        text_embeddings = output[0][:, 0, :]
        return text_embeddings


class BertSeqVec(object):
    def __init__(self, text_net):
        """
        接收一个bert或albert模型，对文本进行向量化。
        :param text_net: bert或albert模型实例。
        """
        self.text_net = text_net
        self.tokenizer = text_net.tokenizer

    def seq2vec(self, text):
        """
        对文本向量化。
        :param text:str，未分词的文本。
        :return:
        """
        text = "[CLS] {} [SEP]".format(text)
        tokens, segments, input_masks = [], [], []

        tokenized_text = self.tokenizer.tokenize(text)  # 用tokenizer对句子分词
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)  # 索引列表
        tokens.append(indexed_tokens)
        segments.append([0] * len(indexed_tokens))
        input_masks.append([1] * len(indexed_tokens))

        max_len = max([len(single) for single in tokens])  # 最大的句子长度

        for j in range(len(tokens)):
            padding = [0] * (max_len - len(tokens[j]))
            tokens[j] += padding
            segments[j] += padding
            input_masks[j] += padding
        tokens_tensor = torch.tensor(tokens)
        segments_tensors = torch.tensor(segments)
        input_masks_tensors = torch.tensor(input_masks)
        text_hashCodes = self.text_net(tokens_tensor, segments_tensors,
                                       input_masks_tensors)  # text_hashCodes是bert模型的文本特征
        return text_hashCodes[0].detach().numpy()


class AlbertTextNet(BertTextNet):
    def __init__(self):
        """
        albert 文本模型。
        """
        super(AlbertTextNet, self).__init__()
        config, tokenizer, model = get_albert_total(al_config_path, al_vocab_path, al_model_path)
        self.textExtractor = model
        self.tokenizer = tokenizer

    def forward(self, tokens, segments, input_masks):
        output = self.textExtractor(tokens, token_type_ids=segments,
                                    attention_mask=input_masks)
        text_embeddings = output[0][:, 0, :]
        return text_embeddings


if __name__ == '__main__':
    texts = ["今天天气不错，适合出行。",
             "今天是晴天，可以出去玩。",
             "施工前需要开工前会。",
             "工作过程中安全第一。"
             ]
    last_vec = None
    distances = []
    text_net = BertTextNet()  # 选择一个文本向量化模型
    seq2vec = BertSeqVec(text_net)  # 将模型实例给向量化对象。
    for text in texts:
        vec = seq2vec.seq2vec(text)  # 向量化
        if last_vec is None:
            last_vec = vec
        else:
            dis = cosine(vec, last_vec)
            distances.append(dis)
            last_vec = vec
    print(np.array(distances))
    print('done')
