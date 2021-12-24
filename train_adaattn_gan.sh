DataPath=/NAS2020/Workspaces/MLGroup/chenxianyu/Dataset/AMSGAN
python train_gan.py --content_path $DataPath/train2014_sub \
  --style_path $DataPath/wikiart/ \
  --name AdaAttN_debug_gan \
  --model adaattn \
  --dataset_mode style_group \
  --no_dropout \
  --load_size 256 \
  --crop_size 256 \
  --image_encoder_path checkpoints/vgg_normalised.pth \
  --gpu_ids 1 \
  --batch_size 8 \
  --n_epochs 2 \
  --n_epochs_decay 3 \
  --display_freq 2 \
  --display_port 8097 \
  --display_env AdaAttN \
  --lambda_local 3 \
  --lambda_global 10 \
  --lambda_content 1 \
  --shallow_layer \
  --skip_connection_3 \
  --lr_dis 5e-5 \
  --continue
