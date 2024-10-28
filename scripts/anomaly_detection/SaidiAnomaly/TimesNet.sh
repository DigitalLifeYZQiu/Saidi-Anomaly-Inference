export CUDA_VISIBLE_DEVICES=1

python -u run.py \
  --task_name anomaly_detection \
  --is_training 0 \
  --root_path ./data \
  --data_path anomaly.csv \
  --model_id SaidiAnomaly \
  --model TimesNet \
  --data SaidiAnomaly \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 64 \
  --d_ff 64 \
  --e_layers 2 \
  --enc_in 23 \
  --c_out 23 \
  --top_k 3 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 3 \
  --crop_anomaly