# text_classification_with_bert
Use Bert pre-trained model to fine-tune a binary text classifier with 

run_classifier.py  --task_name=my_processor  --do_train=true   --do_eval=true --data_dir=$BERT/DATA/DIR/data   --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt   --max_seq_length=128   --train_batch_size=32   --learning_rate=2e-5   --num_train_epochs=3.0   --output_dir=./result/


*** Eval Results ***
   
   | eval_accuracy |  eval_loss  | 
   | ------------- | ----------- |
   |   0.9985759   | 0.008041907 |
