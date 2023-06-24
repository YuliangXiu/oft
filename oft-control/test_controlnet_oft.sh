getenv=True
source /home/yxiu/miniconda3/bin/activate OPT
python ./oft-control/test_oft_parallel.py \
  --img_ID=1 \
  --eps=1e-3 \
  --r=4 \
  --coft