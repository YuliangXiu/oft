getenv=True
source /home/yxiu/miniconda3/bin/activate OPT
python ./oft-control/train.py \
  --eps=1e-3 \
  --r=4 \
  --coft