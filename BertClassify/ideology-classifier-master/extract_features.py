# Edited from code at "https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/extract_features.py"
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
import pandas as pd
from data_processing import LeftRightProcessor
import torch
import collections
import json
from tqdm import tqdm
import argparse
import os
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from collections import namedtuple, OrderedDict
from numpy import argmax

def get_dataset(features): 
    """ Converts a set of features to a TensorDataset """
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    all_case_ids = torch.tensor([f.case_id for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    return TensorDataset(all_input_ids, all_input_mask, all_example_index, all_case_ids, all_label_ids)

def get_eval_dataloader(eval_features, eval_batch_size=8):
    """ parses test examples and prepares them into a DataLoader """
    eval_dataset = get_dataset(eval_features)
    eval_sampler = SequentialSampler(eval_dataset)
    return DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

def _convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """ convert each example in a list to an InputFeatures """
    InputFeatures = namedtuple("InputFeatures", ["input_ids", "input_mask", "segment_ids", "label_id", "tokens", "case_id"])
    
    labels_to_int = {label:i for i,label in enumerate(label_list)}
    print(labels_to_int)
    
    features = []
    for (ex_index, example) in tqdm(enumerate(examples), desc="converting examples"):
        tokens = tokenizer.tokenize(example.text)
        if len(tokens) > max_seq_length - 2 :
             tokens = tokens[:(max_seq_length - 2)]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = labels_to_int[example.label]
        case_id = example.guid
        features.append(InputFeatures(input_ids, input_mask, segment_ids, label_id, tokens, case_id))
    return features

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_examples", type=int, default=10000,
        help="number of examples from which features will be extracted"
    )
    parser.add_argument("--output_dir", type=str, default="./output",
        help="place to write features.json"
    )
    parser.add_argument("--num_labels", type=int, default=2,
        help="number of classification labels"
    )
    parser.add_argument("--max_seq_length", type=int, default=128,
        help="max length of a sequence"
    )
    parser.add_argument("--cased", action="store_true",
        help="whether to consider letter case"
    )
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased",
        help="the type of pretrained bert model (default: 'bert-base-uncased')"
    )
    parser.add_argument("--trained_model_dir", type=str, default="./output",
        help="directory storing pytorch_model.bin and bert_config.json files to load model"
    )
    parser.add_argument("--seed", type=int, default=42,
        help="random seed"
    )
    parser.add_argument("--device", type=str, default="cuda",
        help="where to perform calculations. 'cpu' or 'cuda'"
    )
    parser.add_argument("--data_dir", type=str, required=True,
        help="location of stored data. Must have a test.csv"
    )
    return parser.parse_args()
    

if __name__ == "__main__":
    args = get_args()
    num_examples = args.num_examples
    output_dir = args.output_dir
    num_labels = args.num_labels
    max_seq_length = args.max_seq_length
    cased = args.cased
    bert_model = args.bert_model
    eval_batch_size = 8 # shouldn't matter
    seed = args.seed
    device = torch.device(args.device)
    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=(not cased))
    output_file = os.path.join(output_dir, "features.json")
    data_dir = args.data_dir
    processor = LeftRightProcessor(
        max_seq_length,
        tokenizer,
        args.data_dir,
        seed,
        1, # Shouldn't be used
        eval_batch_size
    )

    model = BertModel.from_pretrained(args.trained_model_dir)
    model.to(device) 
    
    label_predictor = BertForSequenceClassification.from_pretrained(args.trained_model_dir, num_labels=num_labels)
    label_predictor.to(device)
    
    examples = processor.get_examples(num_examples, "test.csv")
    features = _convert_examples_to_features(examples, processor.get_labels(), max_seq_length, tokenizer)
    unique_id_to_feature = {}
    for feature in features:
        unique_id_to_feature[feature.case_id] = feature

    unique_id_to_text = {ex.guid:ex.text for ex in examples}
    
    dl = get_eval_dataloader(features)
    
    model.eval()
    label_predictor.eval()
    with open(output_file, "w", encoding='utf-8') as writer:
        for input_ids, input_mask, example_indices, _, label_ids in tqdm(dl, desc="dataloader"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            label_ids = label_ids.to(device)

            all_encoder_layers, _ = model(input_ids, token_type_ids=None, attention_mask=input_mask)
            all_encoder_layers = all_encoder_layers

            with torch.no_grad():
                logits = label_predictor(input_ids, token_type_ids=None, attention_mask=input_mask)
            logits = logits.detach().cpu().numpy()
            predictions = argmax(logits, axis=1)
            
            for b, example_index in enumerate(example_indices):
                feature = features[example_index.item()]
                unique_id = int(feature.case_id)

                output_json = OrderedDict()
                output_json["id"] = int(unique_id)
                output_json["label"] = int(label_ids[b])
                output_json["confidence"] = float(abs(logits[b][0] - logits[b][1]))
                output_json["prediction"] = int(predictions[b])
                
                layer_output = all_encoder_layers[-1].detach().cpu().numpy() # -1 corresponds to the last layer of the model before output or the output of the Transformer (i think)
                layer_output = layer_output[b]
                values = [
                    round(x.item(), 6) for x in layer_output[0] # 0 corresponds to the token [CLS]
                ]

                output_json["vector"] = list(values)
                output_json["text"] = unique_id_to_text[unique_id]
                writer.write(json.dumps(output_json) + "\n")