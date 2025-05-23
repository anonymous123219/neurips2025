#!/bin/bash

# Set GPU info
CUDA_VISIBLE_DEVICES=0,1 #$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd "," -)
NUM_GPUS=2 #$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# List of result directories to evaluate
RESULT_PATHS=(
    "./results/aesthetic/dpo_beta8k/checkpoint-100"
)


for RESULT_DIR in "${RESULT_PATHS[@]}"; do
    OUTDIR="${RESULT_DIR}/geneval_img"
    OUTFILE="${RESULT_DIR}/geneval_results.jsonl"
    UNET_PATH="${RESULT_DIR}/unet"
    METADATA="./evaluation/geneval/prompts/evaluation_metadata.jsonl"
    DETECTOR="./evaluation/geneval/OBJECT_DETECTOR_FOLDER"

    COMMON_ARGS="$METADATA \
    --model "CompVis/stable-diffusion-v1-4" \
    --outdir "$OUTDIR" \
    --unet_path "$UNET_PATH" \
    --batch_size 4 \
    --scale 7.5 \
    --steps 25 \
    --cache_dir "./cache"
    "

    COMMON_ARGS2="$OUTDIR \
    --outfile "$OUTFILE" \
    --model-path "$DETECTOR"
    "

    echo "🔁 Running generation and evaluation for: $RESULT_DIR"

    PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
    if [ ${NUM_GPUS} -gt 1 ]; then
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
        accelerate launch --multi_gpu --num_processes ${NUM_GPUS} --main_process_port $PORT evaluation/geneval_generate.py $COMMON_ARGS
    else
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
        accelerate launch evaluation/geneval_generate.py $COMMON_ARGS
    fi

done




for ckpt in "${RESULT_PATHS[@]}"; do
    outdir="${ckpt}/HPS_img"
    COMMON_ARGS="\
    --model_id "CompVis/stable-diffusion-v1-4" \
    --ckpt "$ckpt" \
    --outdir "$outdir" \
    --guidance_scale 7.5 \
    --num_inference_steps 25 \
    --batch_size 8
    "
    COMMON_ARGS2="--image_path "$outdir" \
    --hps_version "v2.0" \
    "

    PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
    if [ ${NUM_GPUS} -gt 1 ]; then
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port $PORT \
        --multi_gpu --num_processes ${NUM_GPUS} evaluation/HPSv2_generate.py $COMMON_ARGS
    else
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch evaluation/HPSv2_generate.py $COMMON_ARGS
    fi
done




file_list=(
    "./evaluation/T2I-CompBench/examples/dataset/color_val.txt"
    "./evaluation/T2I-CompBench/examples/dataset/shape_val.txt"
    "./evaluation/T2I-CompBench/examples/dataset/texture_val.txt"
    "./evaluation/T2I-CompBench/examples/dataset/spatial_val.txt"
    "./evaluation/T2I-CompBench/examples/dataset/non_spatial_val.txt"
    "./evaluation/T2I-CompBench/examples/dataset/complex_val.txt"
)

batch_size=10
n_iter=1

for ckpt in "${RESULT_PATHS[@]}"; do
    for file in "${file_list[@]}"; do

        dataset_name=$(basename "$file" | sed 's/_val.*//')

        outdir="${ckpt}/compbench_img/${dataset_name}"
        mkdir -p "$outdir"

        echo "Running with checkpoint: $ckpt, file: $file, and output directory: $outdir"
        
        
        COMMON_ARGS="--from_file "$file" \
            --ckpt "$ckpt" \
            --batch_size "$batch_size" \
            --n_iter "$n_iter" \
            --outdir "$outdir" \
            --model_id "CompVis/stable-diffusion-v1-4" \
            --cache_dir "./cache" 
        "
        PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

        if [ ${NUM_GPUS} -gt 1 ]; then
            CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port $PORT \
            --multi_gpu --num_processes ${NUM_GPUS} evaluation/compbench_generate.py $COMMON_ARGS
        else
            CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch evaluation/compbench_generate.py $COMMON_ARGS
        fi

    done
done

# FID generation
echo "Running FID generation for all checkpoints"

batch_size=16

for ckpt in "${RESULT_PATHS[@]}"; do
    outdir="${ckpt}/MSCOCO_img"

    COMMON_ARGS="--ckpt "$ckpt" \
    --batch_size "$batch_size" \
    --outdir "$outdir" \
    --model_id "CompVis/stable-diffusion-v1-4" \
    --cache_dir "./cache" 
    "

    echo "Running with checkpoint: $ckpt, and output directory: $outdir"

    PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
    if [ ${NUM_GPUS} -gt 1 ]; then
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port $PORT \
        --multi_gpu --num_processes ${NUM_GPUS} evaluation/FID_generation.py $COMMON_ARGS
    else
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch evaluation/FID_generation.py $COMMON_ARGS
    fi
done


