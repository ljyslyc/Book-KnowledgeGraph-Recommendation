# ideology-classifier
An NLP classifier using a pre trained BERT model. 


adapted from https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_classifier.py#L486. 

data_dir should be a folder with a **train.csv** and a **test.csv**. Both CSVs should have columns "id", "tex", and "label" (with each case as a row).  

From classifier.py -h:

> usage: classifier.py 
> -                      --data_dir DATA_DIR 
> -                     [--bert_model BERT_MODEL] 
> -                     [--output_dir OUTPUT_DIR]
> -                     [--max_seq_length MAX_SEQ_LENGTH] 
> -                     [--skip_train]
> -                     [--cased]
> -                     [--train_batch_size TRAIN_BATCH_SIZE]
> -                     [--eval_batch_size EVAL_BATCH_SIZE]
> -                     [--learning_rate LEARNING_RATE]
> -                     [--num_train_epochs NUM_TRAIN_EPOCHS]
> -                     [--warmup_proportion WARMUP_PROPORTION] 
> -                     [--no_cuda]
> -                     [--local_rank LOCAL_RANK] 
> -                     [--cache_dir CACHE_DIR]
> -                     [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
> -                     [--num_train_examples NUM_TRAIN_EXAMPLES]
> -                     [--num_test_examples NUM_TEST_EXAMPLES]
                     

# document_vector.ipynb
Visualizes the spread of data and their similarities to one another. 
use extract_features.py to prepare json for document_vector notebook.
