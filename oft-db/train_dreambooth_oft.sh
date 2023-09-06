getenv=True
source /home/yxiu/miniconda3/bin/activate OPT

# export paths
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
# export HF_HOME='/tmp'
export HF_HOME="/is/cluster/yxiu/.cache"

eps=6e-5
rank=4

# user defined
class_token="man"
unique_token="yuliang"
common_template="a high-quality, extremely detailed, 4K, HQ, DSLR photo of"
template_prompt="${common_template} a asian man with short brown hair, wearing a blue skinny jeans, a white sneakers, and a pink t-shirt"
personalized_prompt="${common_template} a yuliang asian man with short brown hair, wearing a yuliang blue skinny jeans, a yuliang white sneakers, and a yuliang pink t-shirt"

exp_name="${unique_token}-${class_token}-prior"

class_prompt="${template_prompt}"
instance_prompt="${personalized_prompt}"
validation_prompt="${personalized_prompt}"

OUTPUT_DIR="log_cot/${exp_name}"
INSTANCE_DIR="../data/dreambooth/${class_token}"
CLASS_DIR="data/class_data/${class_token}"

accelerate launch train_dreambooth_oft.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir="$CLASS_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --instance_prompt="$instance_prompt" \
  --resolution=512 \
  --train_batch_size=5 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --checkpointing_steps=500 \
  --learning_rate=1e-6 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=10000 \
  --validation_prompt="$validation_prompt" \
  --validation_steps=100 \
  --num_train_epochs=5 \
  --seed="0" \
  --name="$exp_name" \
  --eps=$eps \
  --rank=$rank \
  --enable_xformers_memory_efficient_attention \
  --use_8bit_adam \
  --set_grads_to_none \
  --num_class_images=200 \
  --class_prompt="$class_prompt" \
  --with_prior_preservation --prior_loss_weight=1.0 \
  # --resume_from_checkpoint="checkpoint-10000" \
  # --coft
  # --learning_rate=6e-5 \
