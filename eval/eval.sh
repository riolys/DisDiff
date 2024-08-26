data_dir=""

# 设置prompt参数列表
prompts=(
        "a_photo_of_sks_person" 
        "a_dslr_portrait_of_sks_person"
        # "a_photo_of_sks_person_in_front_of_eiffel_tower" 
        # "a_photo_of_sks_person_looking_at_the_mirror"
        )

output_file="$data_dir/eval.txt"
emb_file="../VGGFace2"
for prompt in "${prompts[@]}"; do
    echo "---------EVAL----------" >> "$output_file"
    echo "The prompt is \"$prompt\"" >> "$output_file"
    python fid.py \
        --data_dir="$data_dir" \
        --prompt="$prompt" \
        --emb_dir="$emb_file"
    python ism_fdfr.py \
        --data_dir="$data_dir" \
        --prompt="$prompt" \
        --emb_dir="$emb_file"
    python brisques.py \
        --data_dir="$data_dir" \
        --prompt="$prompt"

done