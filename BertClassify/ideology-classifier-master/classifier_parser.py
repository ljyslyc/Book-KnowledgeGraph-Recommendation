import argparse

def get_args():
    return initialize_parser().parse_args()

def initialize_parser():
    """ Defines the structure for the argparser """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .csv files (or other data files) for the task."
    )
    parser.add_argument(
        "--bert_model", default="bert-base-uncased", type=str, required=False,
        help="Bert pre-trained model selected in the list: bert-base-uncased (default), "
        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
        "bert-base-multilingual-cased, bert-base-chinese."
    )
    parser.add_argument(
        "--output_dir",
        default="/tmp/classifier_output",
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written. (default: /tmp/classifier_output)"
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. \n"
                "Sequences longer than this will be truncated, and sequences shorter \n"
                "than this will be padded."
    )
    parser.add_argument(
        "--skip_train",
        action='store_true',
        help="Whether to skip training."
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Whether to skip model evaluation"
    )
    parser.add_argument(
        "--cased",
        action='store_true',
        help="Set this flag if you are using a cased model."
    )
    parser.add_argument(
        "--train_batch_size",
        default=32,
        type=int,
        help="Total batch size for training."
    )
    parser.add_argument(
        "--eval_batch_size",
        default=8,
        type=int,
        help="Total batch size for eval."
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=1.0,
        type=float,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. "
                "E.g., 0.1 = 10%% of training."
    )
    parser.add_argument(
        "--use_cpu",
        action='store_true',
        help="Use cpu over cuda for calculations"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus"
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3"
    )
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass."
    )
    parser.add_argument(
        "--num_train_examples",
        type=int,
        default=50,
        help="The number of rows from the train data to use"
    )
    parser.add_argument(
        "--num_test_examples",
        type=int,
        default=50,
        help="The number of rows from the train data to use"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for random generators"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite of previously stored models if found."
    )
    parser.add_argument(
        "--num_labels",
        type=int,
        default=2,
        help="Number of classification groups"
    )
    return parser