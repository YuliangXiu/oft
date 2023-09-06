exp_name="controlnet-oft"

export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="log_cot/${exp_name}"
export HF_HOME="/is/cluster/yxiu/.cache"
export TRANSFORMERS_OFFLINE=1
export TORCH_DISTRIBUTED_DEBUG="DETAIL"

getenv=True
source /home/yxiu/miniconda3/bin/activate OPT

accelerate launch train_ctrl_oft.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --report_to="wandb" \
 --dataset_name="deepfashion" \
 --resolution=512 \
 --learning_rate=1e-5 \
 --checkpointing_steps=500 \
 --validation_steps=100 \
 --num_validation_images=4 \
 --num_train_epochs=100 \
 --train_batch_size=8 \
 --resume_from_checkpoint="checkpoint-16000" \
 --seed="0" \
 --name="$exp_name" \
 --gradient_accumulation_steps=4 \
 --gradient_checkpointing \
 --use_8bit_adam \
 --enable_xformers_memory_efficient_attention \
 --set_grads_to_none \
 --rank=4 \
 --eps=6e-5 \