python setup.py build develop
python run_net_emamix.py \
  --cfg configs/Kinetics/TimeSformer_base_ssl.yaml \
  DATA.PATH_TO_DATA_DIR ./dataset/list_ucf_1/ \
  OUTPUT_DIR ./output/list_ucf_1/ucf101_1_reproduce/ \
  NUM_GPUS 8 \
  TRAIN.BATCH_SIZE 8 \
  TEST.BATCH_SIZE 32 \
  TRAIN.ENABLE True \
  TRAIN.FINETUNE False
