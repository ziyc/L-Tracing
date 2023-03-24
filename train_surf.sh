shape_ckpt_path=./logs/lego_shape/ckpts/latest.pt # the path to the trained shape ckpt

python train.py --config ./config/surfactor.yaml --training:ckpt_file $shape_ckpt_path