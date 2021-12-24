DataPath=/NAS2020/Workspaces/MLGroup/chenxianyu/Dataset/AMSGAN
python train.py --content_path $DataPath/train2014_sub \
  --style_path $DataPath/wikiart/ \
  --name AdaIN_debug \
  --model adain \
  --dataset_mode unaligned \
  --no_dropout \
  --load_size 256 \
  --crop_size 256 \
  --gpu_ids 0 \
  --batch_size 8 \
  --n_epochs 3 \
  --n_epochs_decay 2 \
  --display_freq 100 \
  --display_port 8098 \
  --display_env AdaIn \
  --lr 0.0001
