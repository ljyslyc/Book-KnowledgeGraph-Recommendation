# PyTorch Transformers BERT Binary Text Classification

[![Made with Python](https://img.shields.io/badge/Made_with-Python-blue.svg)](https://img.shields.io/badge/Made_with-Python-blue.svg) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project demonstrates how to finetune a BERT binary text classification model with `transformers`. In fact, this is also a sentiment analysis implementation because we use [the Yelp Review Polarity dataset](https://s3.amazonaws.com/fast-ai-nlp/yelp_review_polarity_csv.tgz).

It is mainly based on [A Simple Guide on Using BERT for Binary Text Classification](https://medium.com/swlh/a-simple-guide-on-using-bert-for-text-classification-bbf041ac8d04) and the author adapted lots of code from HuggingFace's `run_classifier.py` example. However, the example has been removed and some functions in the package have been deprecated, such as `BertAdam`. Therefore, I make the following modifications:
  * Update `pytorch-pretrained-bert` to `transformers`. Many mechanisms have changed.
  * Don't use `multiprocessing` when running `convert_example_to_features` because I had difficulty making it work in Jupyter Notebook and the computation didn't take me too much time.
  * Put all code in a Jupyter notebook so that we can benefit from the advantages of interactive computing.

## Overview
Here is an overview of the notebook.
  * Import packages and declare variables.
  * Data preparation.
    * Use `pandas.read_csv()` to load the datasets into `DataFrame`s and then save them into `.tsv` files with `pandas.DataFrame.to_csv()`.
  * Data to features.
    * Define the `InputExample` class to represent a single train/dev example.
    * Define the `DataProcessor` interface and the `BinaryClassificationProcessor` class.
    * Define the `InputFeatures` class to represent a single set of features of data.
    * Create the `_truncate_seq_pair()` and `convert_example_to_feature()` functions.
  * Pickling.
    * Load in the `.tsv` files with `BinaryClassificationProcessor`.
    * Build `train_examples_for_processing` (from `train_examples`, `label_map`, `MAX_SEQ_LENGTH`, `BertTokenizer`, and `OUTPUT_MODE`).
    * Execute `convert_example_to_feature()` for each item of `train_examples_for_processing`.
    * Use `pickle.dump()` to pickle `train_features` for safekeeping.
  * Finetuning.
    * Load pre-trained model weights with `BertForSequenceClassification.from_pretrained()`.
    * Initialize the `AdamW` optimizer with defined `optimizer_grouped_parameters` and the `WarmupLinearSchedule` scheduler.
    * Load the pickled features with `pickle.load()`.
    * Create `train_data` with `TensorDataset` (combined by `all_input_ids`, `all_input_mask`, `all_segment_ids`, `all_label_ids`).
    * Setup `RandomSampler` and `DataLoader`.
    * Train the model in practice (about 5 hours with my GTX 1070).
    * Save the model, configuration file, and vocabulary. 
  * Model evaluation.
    * Create the `get_eval_report()` and `compute_metrics()` functions for evaluation.
    * Convert dev dataset into features (very similar to the pickling step).
    * Initialize `BertTokenizer` with our vocabulary and `BertForSequenceClassification` with our finetuned model.
    * Setup `SequentialSampler` and `Dataloader`.
    * Evaluate the model and run `compute_metrics()` to our predictions.

The process is tedious and complex, but it's definitely a good and rare resource for finetuning with BERT using `transformers`.

## Todos
 - Unfortunately, after I worked on some projects from Stanford's [Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/) course, I found getting your hands dirty with DNN libraries, such as PyTorch, is a faster way to learn rather than joining a class or reading a lot of papers. Just go to [PyTorch Tutorials](https://pytorch.org/tutorials/) and code.
 - Build a question answering system for competitions.

## License
[PyTorch Transformers BERT Binary Text Classification](https://github.com/yungshun317/pytorch-transformers-bert-binary-text-classification) is released under the [MIT License](https://opensource.org/licenses/MIT) by [yungshun317](https://github.com/yungshun317).