import sys
sys.path.append('../')
from albert.modeling_albert import BertConfig as AlbertConfig
from albert.modeling_albert import AlbertModel
from albert.tokenization_bert import BertTokenizer as AlbertTokenizer


def get_albert_total(config_path, vocab_path, model_path):
    MODEL_CLASSES = {
        'albert': (AlbertConfig, AlbertModel, AlbertTokenizer)
    }
    config_class, model_class, tokenizer_class = MODEL_CLASSES['albert']

    config = config_class.from_pretrained(config_path,
                                          num_labels=2,
                                          finetuning_task='xnli',
                                          share_type='all')
    tokenizer = tokenizer_class.from_pretrained(vocab_path,
                                                do_lower_case=True)
    model = model_class.from_pretrained(model_path, from_tf=False,
                                        config=config)

    return config, tokenizer, model

if __name__ == '__main__':
    config_path = '../model/albert_tiny/bert_config.json'
    model_path = '../model/albert_tiny/pytorch_model.bin'
    vocab_path = '../model/albert_tiny/vocab.txt'
    config, tokenizer, model = get_albert_total(config_path, vocab_path, model_path)