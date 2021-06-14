export CUDA_VISIBLE_DEVICES=1

export BERT_BASE_DIR=/home/beyoung/nlp_cls/models/chinese_L-12_H-768_A-12

#export DATASET=../data/

python run_classifier.py \
  --data_dir=datasets \
  --task_name=nlp \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --output_dir=../output/ \
  --do_train=true \
  --do_eval=true \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=200 \
  --train_batch_size=16 \
  --learning_rate=5e-5\
  --num_train_epochs=2.0