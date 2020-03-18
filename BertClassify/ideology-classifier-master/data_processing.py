import os
import pandas as pd
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from collections import namedtuple

class LeftRightProcessor: 
    """ The LeftRight Processor handles getting data from csv files to a format useful to the model. """
    def __init__(self, max_seq_length, tokenizer, data_dir, seed, train_batch_size, eval_batch_size):
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.seed = seed
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        
    def get_labels(self):
        "classification labels for the dataset"
        return ["left", "right"]

    def check_data_exists(self, skip_train):
        """ Raise error if can't data can not be found """
        cant_find_train_data = (not skip_train) and (not os.path.exists(self.data_dir+"/train.csv"))
        if cant_find_train_data: raise ValueError("could not find {}/train.csv".format(self.data_dir))
        cant_find_test_data = not os.path.exists(self.data_dir+"/test.csv")
        if cant_find_test_data: raise ValueError("could not find {}/test.csv".format(self.data_dir))

    def get_examples(self, num_examples, file):
        """ Pull rows out from csv file and return them as [InputExample] """
        InputExample = namedtuple("InputExample", ["guid", "text", "label"])
        
        examples = []
        path = os.path.join(self.data_dir, file)
        data = pd.read_csv(path, lineterminator="\n")
        data = data.sample(n=num_examples, random_state=self.seed)
        for _, row in data.iterrows(): 
            guid = row.id 
            text = row.text
            label = row.label
            examples.append(InputExample(guid, text, label))
        return examples
    
    def get_dataloader(self, examples, sampling_function, batch_size):
        features = convert_examples_to_features(examples, self.get_labels(), self.max_seq_length, self.tokenizer)
        dataset = self.get_dataset(features)
        sampler = sampling_function(dataset)
        return DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    def get_dataset(self, features):
        """ Converts a set of features to a TensorDataset """
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        return TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    def get_train_dataloader(self, local_rank, num_examples):
        """ Convert training examples to a DataLoader  """
        examples = self.get_examples(num_examples, "train.csv")
        self.num_train_examples = len(examples)
        if local_rank == -1:
            sampler = RandomSampler
        else:
            sampler = DistributedSampler
            
        return self.get_dataloader(examples, sampler, self.train_batch_size)

    def get_eval_dataloader(self, num_examples):
        """ parses test examples and prepares them into a DataLoader """
        examples = self.get_examples(num_examples, "test.csv")
        self.num_eval_examples = len(examples)
        sampler = SequentialSampler
        return self.get_dataloader(examples, sampler, self.eval_batch_size)

    def get_num_train_optimization_steps(self, gradient_accumulation_steps, num_train_epochs, local_rank):
        steps_per_epoch = int(self.num_train_examples / self.train_batch_size / gradient_accumulation_steps)
        num_steps = steps_per_epoch * num_train_epochs
        
        if local_rank != -1:
            num_steps = num_steps // torch.distributed.get_world_size()
            
        return num_steps
    
def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """ convert each example in a list to an InputFeatures """
    InputFeatures = namedtuple("InputFeatures", ["input_ids", "input_mask", "segment_ids", "label_id", "case_id"])
    
    labels_to_int = {label:i for i,label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
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
        features.append(InputFeatures(input_ids, input_mask, segment_ids, label_id, case_id))
    return features


def create_output_dir(output_dir, force): 
    """creates output directories if needed """
    if not force and os.path.exists(output_dir) and ("pytorch_model.bin" in os.listdir(output_dir)):
         raise ValueError("Output directory ({}) already exists and contains a model. Use --force to force overwrite.".format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(os.path.join(output_dir, "eval_results")):
        os.makedirs(os.path.join(output_dir, "eval_results"))

