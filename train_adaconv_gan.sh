DataPath=/NAS2020/Workspaces/MLGroup/chenxianyu/Dataset/AMSGAN
python train_gan.py --content_path $DataPath/train2014_sub \
  --style_path $DataPath/wikiart/ \
  --name AdaConv_debug_gan \
  --model adaconv \
  --dataset_mode style_group \
  --no_dropout \
  --load_size 256 \
  --crop_size 256 \
  --gpu_ids 3 \
  --batch_size 8 \
  --n_epochs 2 \
  --n_epochs_decay 3 \
  --display_freq 100 \
  --display_port 8099 \
  --display_env AdaConv \
  --lr 1e-4 \
  --lr_dis 5e-5
