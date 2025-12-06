gpu_id=$1

CUDA_VISIBLE_DEVICES=$gpu_id \
python clearml_train_v2.py \
    configs/hsface20k_r50_arcface \
    --use_clearml True \
    --project_name nakajmiya_synthetic_train \
    --gpu_id $gpu_id
