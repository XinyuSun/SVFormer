python setup.py build develop
python run_net_emamix.py \
  --cfg configs/Kinetics/TimeSformer_base_ssl.yaml \
  DATA.PATH_TO_DATA_DIR ./dataset/list_ucf_1/ \
  OUTPUT_DIR ./output/list_ucf_1/ucf_1_vanilla_vit/ \
  NUM_GPUS 8 \
  TRAIN.BATCH_SIZE 8 \
  TEST.BATCH_SIZE 32 \
  TRAIN.ENABLE True \
  TRAIN.FINETUNE False \
  MODEL.MODEL_NAME "semi_m3video_base_patch16_224" \
  TIMESFORMER.PRETRAINED_MODEL "output/checkpoint-1600.pth" \
  DATA.NUM_FRAMES 16 \
  TRAIN.CHECKPOINT_FILE_PATH "output/list_ucf_1/ucf_1_vanilla_vit/checkpoints/checkpoint_epoch_00005.pyth" 
