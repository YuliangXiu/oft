getenv=True
source /home/yxiu/miniconda3/bin/activate OPT
python oft-control/tool_add_control_oft.py \
  --input_path=./models/v1-5-pruned.ckpt \
  --output_path=./models/control_sd15_ini_oft.ckpt \
  --eps=1e-3 \
  --r=4 \
  --coft