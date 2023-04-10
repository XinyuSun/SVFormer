python setup.py build develop
python run_net_emamix.py \
  --cfg configs/Kinetics/TimeSformer_base_ssl.yaml \
  DATA.PATH_TO_DATA_DIR ./dataset/my/k400-0.01/ \
  OUTPUT_DIR ./output/list_k400_1/k400_1/ \
  NUM_GPUS 8 \
  TRAIN.BATCH_SIZE 8 \
  TEST.BATCH_SIZE 32 \
  TRAIN.ENABLE True \
  TRAIN.FINETUNE False \
  MODEL.MODEL_NAME "semi_m3video_base_patch16_224" \
  TIMESFORMER.PRETRAINED_MODEL "output/checkpoint-1600.pth"