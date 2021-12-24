DataPath=/NAS2020/Workspaces/MLGroup/chenxianyu/Dataset/AMSGAN
python train.py --content_path $DataPath/train2014_sub \
  --style_path $DataPath/wikiart/ \
  --name AdaConv_debug \
  --model adaconv \
  --dataset_mode unaligned \
  --no_dropout \
  --load_size 256 \
  --crop_size 256 \
  --gpu_ids 2 \
  --batch_size 8 \
  --n_epochs 2 \
  --n_epochs_decay 3 \
  --display_freq 100 \
  --display_port 8098 \
  --display_env AdaConv \
  --lr 1e-4
