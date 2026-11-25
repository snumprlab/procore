#!/usr/bin/env bash
# Extract faithful MTV (Huang et al. 2024) + FV (Todd et al. 2024) task vectors from ICL TRAIN data only.
# Writes MTV/storage_v2/<arch>/<bench>/ and fv_v2/<arch>/<bench>/ (old MTV/storage and fv/ are untouched).
# Evaluate afterwards with:  METHOD=mtv_v2 DATASET_NAME=<bench> bash eval.sh <GPU>   (or METHOD=fv_v2)
#
# Usage: GPU=0 ARCHS="qwen3-8b internvl3_5-8b idefics-8b onevision1_5-8b" BENCHES="mic vliclv2 truemicl isekai" \
#        bash extract_taskvec_v2.sh
# Optional: SCENARIOS="1 2" (subset of the benchmark's eval scenarios), TV_METHOD=both|mtv|fv,
#           SMOKE=1 (tiny settings, writes to MTV/storage_v2_smoke + fv_v2_smoke), EXTRA="--tv_epochs 100"
set -u
cd "$(dirname "$0")"
GPU="${GPU:-0}"
ARCHS=(${ARCHS:-qwen3-8b})
BENCHES=(${BENCHES:-mic vliclv2 truemicl isekai})
SCENARIOS="${SCENARIOS:-}"
TV_METHOD="${TV_METHOD:-both}"
SMOKE="${SMOKE:-0}"
EXTRA="${EXTRA:-}"
LOGDIR="${LOGDIR:-./nohup/taskvec_v2}"
mkdir -p "$LOGDIR"
export HF_HOME="${HF_HOME:-/tmp/hf_cache}"
PY="${PY:-/tmp/procore/venv/bin/python}"

if [ "$SMOKE" == "1" ]; then
    TV_ARGS="--tv_n_mean 4 --tv_n_reinforce_items 4 --tv_epochs 3 --tv_samples 4 --tv_n_val 3 --tv_n_ie 2 --tv_n_fv_check 2 --tv_out_tag _smoke"
else
    TV_ARGS=""
fi

for MODEL_ARCH in "${ARCHS[@]}"; do
    case "$MODEL_ARCH" in
        qwen3-8b)        MODEL_NAME="Qwen/Qwen3-VL-8B-Instruct"; VERSION="None"; VISION_TOWER="None"; MODEL_TYPE="None" ;;
        internvl3_5-8b)  MODEL_NAME="OpenGVLab/InternVL3_5-8B"; VERSION="v1"; VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"; MODEL_TYPE="qwen" ;;
        idefics-8b)      MODEL_NAME="HuggingFaceM4/Idefics3-8B-Llama3"; VERSION="llama"; VISION_TOWER="google/siglip-so400m-patch14-384"; MODEL_TYPE="llama3-8b" ;;
        onevision1_5-8b) MODEL_NAME="lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"; VERSION="None"; VISION_TOWER="None"; MODEL_TYPE="None" ;;
        *) echo "unknown arch $MODEL_ARCH"; exit 1 ;;
    esac
    for BENCH in "${BENCHES[@]}"; do
        LOG="$LOGDIR/${MODEL_ARCH}_${BENCH}$( [ "$SMOKE" == "1" ] && echo _smoke ).log"
        echo "[extract_taskvec_v2] $MODEL_ARCH $BENCH -> $LOG"
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPU $PY taskvec_v2_extract.py \
            --model_name_or_path $MODEL_NAME --model_name_for_dataarg $MODEL_NAME --model_name ${MODEL_ARCH}_tv2 \
            --model_type $MODEL_TYPE --version $VERSION --vision_tower $VISION_TOWER --bits 16 --bf16 True --tf32 True \
            --mode tv_extract --is_eval True --lora_enable False --zeroshot True --attn_impl flash_attention_2 --model_max_length 40000 \
            --output_dir ./nohup \
            --tv_bench $BENCH --tv_arch $MODEL_ARCH --tv_scenarios "$SCENARIOS" --tv_method $TV_METHOD $TV_ARGS $EXTRA \
            > "$LOG" 2>&1
        status=$?
        if [ $status -ne 0 ]; then echo "  FAILED ($status), see $LOG"; else tail -n 2 "$LOG"; fi
    done
done
