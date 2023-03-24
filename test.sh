test_cfg_path=./logs/lego_surf/config.yaml
val_downscale=1

python test.py --config $test_cfg_path --data:val_downscale $val_downscale