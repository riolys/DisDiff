export CUDA_VISIBLE_DEVICES=0
export EXPERIMENT_NAME="disdiff_layer/ASPL"
export MODEL_PATH="./stable-diffusion/stable-diffusion-2-1-base"
export BASE_DIR="/data2/public/liuyisu/adb_set/VGGFace2"
export CLASS_DIR="data/class-person"

specified_dirs=(
  "$BASE_DIR/n000050"
)

for sub_dir in "${specified_dirs[@]}"; do
  CLEAN_TRAIN_DIR="$sub_dir/set_A"
  CLEAN_ADV_DIR="$sub_dir/set_B"
  # Check if set_A and set_B directories exist in the current subdirectory
  if [ -d "$CLEAN_TRAIN_DIR" ] && [ -d "$CLEAN_ADV_DIR" ]; then
    OUTPUT_DIR="$EXPERIMENT_NAME/$(basename "$sub_dir")_ADVERSARIAL"
# ------------------------- Train ASPL on set B -------------------------
  mkdir -p $OUTPUT_DIR

  accelerate launch attacks/disdiff.py \
    --pretrained_model_name_or_path=$MODEL_PATH  \
    --enable_xformers_memory_efficient_attention \
    --instance_data_dir_for_train=$CLEAN_TRAIN_DIR \
    --instance_data_dir_for_adversarial=$CLEAN_ADV_DIR \
    --instance_prompt="a photo of sks person" \
    --class_data_dir=$CLASS_DIR \
    --num_class_images=200 \
    --class_prompt="a photo of person" \
    --output_dir=$OUTPUT_DIR \
    --center_crop \
    --prior_loss_weight=1.0 \
    --resolution=512 \
    --train_text_encoder \
    --train_batch_size=1 \
    --max_train_steps=50 \
    --max_f_train_steps=3 \
    --max_adv_train_steps=6 \
    --checkpointing_iterations=50 \
    --learning_rate=5e-7 \
    --pgd_alpha=5e-3 \
    --pgd_eps=5e-2 \
    --with_prior_preservation \
    --use_CAE \
    --att_param=0.1 \
    --loss_func="energy" \
    --use_MSS \
    --use_search \
    --seed=0
    
  # ------------------------- Train DreamBooth on perturbed examples -------------------------
  export INSTANCE_DIR="$OUTPUT_DIR/noise-ckpt/50"
  export DREAMBOOTH_OUTPUT_DIR="$EXPERIMENT_NAME/AB/$(basename "$sub_dir")_dreambooth"

  accelerate launch train_dreambooth.py \
    --pretrained_model_name_or_path=$MODEL_PATH  \
    --enable_xformers_memory_efficient_attention \
    --train_text_encoder \
    --instance_data_dir=$INSTANCE_DIR \
    --class_data_dir=$CLASS_DIR \
    --output_dir=$DREAMBOOTH_OUTPUT_DIR \
    --with_prior_preservation \
    --prior_loss_weight=1.0 \
    --instance_prompt="a photo of sks person" \
    --class_prompt="a photo of person" \
    --inference_prompt="a photo of sks person;a dslr portrait of sks person" \
    --resolution=512 \
    --train_batch_size=2 \
    --gradient_accumulation_steps=1 \
    --learning_rate=5e-7 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --num_class_images=200 \
    --max_train_steps=1000 \
    --checkpointing_steps=1000 \
    --center_crop \
    --mixed_precision=bf16 \
    --prior_generation_precision=bf16 \
    --sample_batch_size=8\
    --not_save \
    # --seed=0
  fi
done