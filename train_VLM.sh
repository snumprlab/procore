# #!/bin/bash
# #SBATCH -p suma_a6000
# #SBATCH --job-name=random
# #SBATCH --output=logs/%x_%j.out
# #SBATCH --gres=gpu:1
# #SBATCH --exclude=node37
# ##SBATCH -c 32

# ulimit -u 200000
# source ~/.bashrc
# ml purge
# conda init bash
# conda activate sampling
# export CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.5

#!/bin/bash
# CIL CONFIG
export WANDB_DISABLED=true
# =============================================================================
# Pick the experiment with METHOD (same names as eval.sh METHOD). Everything that
# decides the checkpoint name (MODE / SHOT / SCENARIO / LR / LoRA / dataset) is
# derived below so that ./client_states_<NOTE>/server_model_round0.pth is exactly
# the file eval.sh loads for that METHOD.
#
#   METHOD          trains with                         checkpoint dir produced
#   dara            mode=dara  shot=original scen=S     client_states_dara_<ARCH>_<DS>_1_original_scen<S>_1e-3
#   mimic_mixed     mode=mimic shot=mixed    scen=0     client_states_mimic_<ARCH>_<DS>_1_mixed_scen0_5e-3   (the mimic checkpoints that exist on disk)
#   mt-icl          mode=sft   shot=mixed    scen=0     client_states_sft_<ARCH>_<DS>_1_mixed_scen0         (eval.sh METHOD=mt-icl (alias Finetune-Mixed) loads this ONE checkpoint for every eval shot)
#   mt-icl-prev     mode=sft   shot=mixed    scen=S     client_states_sft_<ARCH>_<DS>_1_mixed_scen<S>       (legacy per-shot recipe; not loaded by eval.sh any more)
#   Finetune        mode=sft   shot=original scen=2     client_states_sft_<ARCH>_inpaintcoco_1_original_scen2  (dataset hard-coded to inpaintcoco)
#   stv             mode=stv   shot=mixed    scen=0     client_states_stv_<ARCH>_<DS>_1_mixed_scen0
#   lrv / volcano / lure  mode=sft shot=original scen=2  client_states_sft_<ARCH>_lrv_1_original_scen2 / ..._volcano_... / ..._lure_... (lure = the LURE revisor)
#   grpo            -> use train_GRPO_VLM.sh (its NOTE carries the reward tag), not this script
#
#   S = TARGET_SHOT = the EVAL shot this checkpoint is for (dara / mt-icl-prev only; eval.sh sets
#       TRAIN_SCENARIO from EVAL_SHOT for those, so one checkpoint per shot is needed).
#   DS = DATASET_NAME without a trailing "_new" (eval.sh strips it when it builds the NOTE).
#
# Usage:   bash train_VLM.sh                      (edit the knobs below)
#          METHOD=mt-icl DATASET_NAME=truemicl MODEL_ARCH=qwen3-8b bash train_VLM.sh
#          METHOD=dara TARGET_SHOT=4 DATASET_NAME=truemicl MODEL_ARCH=qwen3-8b bash train_VLM.sh
#          DRY_RUN=1 bash train_VLM.sh            (only print the NOTE / checkpoint path and exit)
# =============================================================================
METHOD="${METHOD:-mimic_mixed}"      # dara mimic_mixed mt-icl mt-icl-prev Finetune stv lrv volcano
export CKPT_ROOT="${CKPT_ROOT:-/tmp/procore/checkpoints_verified}"; mkdir -p "$CKPT_ROOT"
MODEL_ARCH="${MODEL_ARCH:-qwen3-4b}" # llava "qwen-7b" "onevision-7b" "internvl2_5-8b" "otter-9b" "idefics-8b" "internvl3_5-8b" "qwen3-8b" "onevision1_5-8b" "qwen3-4b" "qwen3-2b"
DATASET_NAME="${DATASET_NAME:-pororov4}" # truemicl mic vliclv2 isekai pororov4 storystreamv2 flintstonesv3 (inpaintcoco/lrv/volcano are forced by METHOD)
TARGET_SHOT="${TARGET_SHOT:-2}"      # eval shot number this checkpoint serves (dara / mt-icl-prev only)
EPOCH=1
GPU="${GPU:-0}"
MASTER_PORT="${MASTER_PORT:-21165}"
FORCE="${FORCE:-0}"                  # 1 = retrain even if the checkpoint already exists
DRY_RUN="${DRY_RUN:-0}"
MAX_STEPS="${MAX_STEPS:--1}"           # >0 only for smoke tests (optimizer steps)
GRAD_ACC="${GRAD_ACC:-16}"             # smoke tests may lower it; batch 16 = paper A.2


# Don't Change Arguments Below-----------------------------------------------------------------

RND_SEED=1
RESUME_TRAIN_ROUND=-1
LOAD_CHECKPOINT="/disk1/mjlee/LLaVA/checkpoints/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin"
VIS_TOKEN_AMPLIFIER=0.1

EVAL_PERIOD=100000
NUM_ITER=1
NUM_TASKS=1

# 0.1 -> 1.28, 0.2 -> 0.84, 0.3 -> 0.52

MODEL_MAX_LEN=20000
MEMORY_SIZE=10000000
IS_STREAMONLY=False

USE_PROMPT=False

SAVE_OPTIM=True

USE_TASK_VECTOR=False
USE_FISHER=False

GENERATOR_OUTPUT_SIZE=1024
GENERATOR_HIDDEN_DIM=8
GENERATOR_HIDDEN_FEATURE=8
KEY_EMBED_SIZE=64
POOL_SIZE=4
PROMPT_TOP_K=1

BATCHSIZE=1
GRAD_CHECKPOINT=True
if [ "$MODEL_ARCH" == "internvl2_5-8b" ]; then
    MODEL_NAME="OpenGVLab/InternVL2_5-8B"
    VERSION="v1"
    VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
    MODEL_TYPE="llama"
    LR=2e-5
    BITS=16
elif [ "$MODEL_ARCH" == "internvl3-8b" ]; then
    MODEL_NAME="OpenGVLab/InternVL3-8B"
    VERSION="v1"
    VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
    MODEL_TYPE="qwen"
    LR=2e-5
    BITS=16
elif [ "$MODEL_ARCH" == "internvl3_5-8b" ]; then
    MODEL_NAME="OpenGVLab/InternVL3_5-8B"
    VERSION="v1"
    VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
    MODEL_TYPE="qwen"
    LR=2e-6   # paper A.2: InternVL 2e-6 (Qwen-VL-3 / OV-1.5: 2e-5); dara/mimic override below
    BITS=16
elif [ "$MODEL_ARCH" == "qwen-7b" ]; then
    MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
    VERSION="v1"
    VISION_TOWER="openai/clip-vit-large-patch14-336"
    MODEL_TYPE="llama"
    LR=2e-5 
    BITS=16
elif [ "$MODEL_ARCH" == "qwen3-8b" ]; then
    MODEL_NAME="Qwen/Qwen3-VL-8B-Instruct"
    VERSION="None"
    VISION_TOWER="None"
    MODEL_TYPE="None"
    LR=2e-5 
    BITS=16
elif [ "$MODEL_ARCH" == "qwen3-4b" ]; then
    MODEL_NAME="Qwen/Qwen3-VL-4B-Instruct"
    VERSION="None"
    VISION_TOWER="None"
    MODEL_TYPE="None"
    LR=2e-5 
    BITS=16
elif [ "$MODEL_ARCH" == "qwen3-2b" ]; then
    MODEL_NAME="Qwen/Qwen3-VL-2B-Instruct"
    VERSION="None"
    VISION_TOWER="None"
    MODEL_TYPE="None"
    LR=2e-5 
    BITS=16
elif [ "$MODEL_ARCH" == "onevision-7b" ]; then
    MODEL_NAME="lmms-lab/llava-onevision-qwen2-7b-ov"
    VERSION="llama"
    VISION_TOWER="google/siglip-so400m-patch14-384"
    MODEL_TYPE="llama3-8b"
    LR=2e-5 
    BITS=16
elif [ "$MODEL_ARCH" == "onevision1_5-8b" ]; then
    MODEL_NAME="lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"
    VERSION="None"
    VISION_TOWER="None"
    MODEL_TYPE="None"
    LR=2e-5 
    BITS=16
elif [ "$MODEL_ARCH" == "idefics-8b" ]; then
    MODEL_NAME="HuggingFaceM4/Idefics3-8B-Llama3"
    VERSION="llama"
    VISION_TOWER="google/siglip-so400m-patch14-384"
    MODEL_TYPE="llama3-8b"
    LR=2e-5
    BITS=16
elif [ "$MODEL_ARCH" == "otter-9b" ]; then
    MODEL_NAME="luodian/otter-9b-hf"
    VERSION="otter"
    VISION_TOWER="openai/clip-vit-large-patch14"
    MODEL_TYPE="otter"
    LR=2e-5
    BITS=16
    GRAD_CHECKPOINT=False
elif [ "$MODEL_ARCH" == "qwen3-8b" ]; then
    MODEL_NAME="Qwen/Qwen3-VL-8B-Instruct"
    VERSION="None"
    VISION_TOWER="None"
    MODEL_TYPE="None"
    LR=2e-5
    BITS=16
elif [ "$MODEL_ARCH" == "internvl3_5-8b" ]; then
    MODEL_NAME="OpenGVLab/InternVL3_5-8B"
    VERSION="v1"
    VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
    MODEL_TYPE="qwen"
    LR=2e-6   # paper A.2: InternVL 2e-6 (Qwen-VL-3 / OV-1.5: 2e-5); dara/mimic override below
    BITS=16
else
    echo "Undefined setting"
    exit 1
fi

# --- METHOD -> training recipe (must stay in sync with eval.sh) -----------------
# eval.sh strips a trailing "_new" before building the NOTE: the *_new eval sets reuse
# the checkpoints trained on the base dataset, so train on the base dataset.
if [[ "$DATASET_NAME" == *_new ]]; then
    echo "[train_VLM.sh] DATASET_NAME=$DATASET_NAME -> training on ${DATASET_NAME%_new} (eval.sh maps *_new to the base checkpoint)"
    DATASET_NAME="${DATASET_NAME%_new}"
fi
ATTEN_IMPLE="flash_attention_2"
LORA_ENABLE=True
LR_SUFFIX=False          # dara / mimic append _<LR> to the NOTE (see eval.sh)
INTERVENTION_MODE="clean"
case "$METHOD" in
    dara)            MODE=dara;  SHOT=original; SCENARIO=$TARGET_SHOT; LR=1e-3; LORA_ENABLE=False; ATTEN_IMPLE="eager"; LR_SUFFIX=True ;;
    mimic_mixed)     MODE=mimic; SHOT=mixed;    SCENARIO=0;            LR=5e-3; LORA_ENABLE=False; LR_SUFFIX=True ;;
    mt-icl)          MODE=sft;   SHOT=mixed;    SCENARIO=0 ;;                                  # was "Finetune-Mixed"; LR from the MODEL_ARCH block (2e-5)
    mt-icl-prev)     MODE=sft;   SHOT=mixed;    SCENARIO=$TARGET_SHOT ;;                       # old per-shot mt-icl recipe
    Finetune)        MODE=sft;   SHOT=original; SCENARIO=2; DATASET_NAME="inpaintcoco" ;;      # dataset hard-coded
    stv)             MODE=stv;   SHOT=mixed;    SCENARIO=0; LORA_ENABLE=False ;;                # frozen model; stv_train.py learns only the selection logits
    lrv)             MODE=sft;   SHOT=original; SCENARIO=2; DATASET_NAME=lrv ;;
    volcano)         MODE=sft;   SHOT=original; SCENARIO=2; DATASET_NAME=volcano ;;  # kaist-ai/volcano-train feedback+revision (tools/build_volcano_datalist.py)
    lure)            MODE=sft;   SHOT=original; SCENARIO=2; DATASET_NAME=lure ;;     # LURE revisor: masked hallucinatory caption -> correct caption (COCO)
    grpo)            echo "METHOD=grpo is trained by train_GRPO_VLM.sh (reward-tagged NOTE), not train_VLM.sh"; exit 1 ;;
    mimic|mimic_static|Finetune-Mixed)
                     echo "METHOD=$METHOD was removed: use mimic_mixed (the only mimic recipe with checkpoints) or mt-icl (formerly Finetune-Mixed)"; exit 1 ;;
    *)               echo "Unknown METHOD '$METHOD' (dara|mimic_mixed|mt-icl|mt-icl-prev|Finetune|stv|lrv|volcano|lure)"; exit 1 ;;
esac
# Smoke tests only: SCENARIO_OVERRIDE=<n> trains on scenarios/<SHOT>/<DATASET_NAME>/scenario-<n>.json
# (e.g. a tiny datalist under datalists/_smoke/); the NOTE then carries scen<n>, never the real one.
if [ -n "${SCENARIO_OVERRIDE:-}" ]; then SCENARIO=$SCENARIO_OVERRIDE; fi

if [[ "$INTERVENTION_MODE" != "clean" && "$INTERVENTION_MODE" != "interv" && "$INTERVENTION_MODE" != "both" ]]; then
    echo "Invalid STV_INTERVENTION_MODE '$INTERVENTION_MODE' (expected clean|interv|both)" >&2
    exit 2
fi

# STV knobs (stv_train.py): top-k heads / KMeans clusters per head / #ICL items for sensitivity+bank /
# REINFORCE passes over STV_RL_ITEMS items / samples per update.
INTERVENTION_TOPK="${STV_INTERVENTION_TOPK:-64}"
INTERVENTION_NUM_CLUSTERS="${STV_INTERVENTION_NUM_CLUSTERS:-16}"
INTERVENTION_N_TRIALS="${STV_INTERVENTION_N_TRIALS:-200}"
INTERVENTION_NUM_SHOT="${STV_INTERVENTION_NUM_SHOT:-4}"
REINFORCE_N_EPOCHS="${STV_REINFORCE_N_EPOCHS:-10}"
REINFORCE_SAMPLES_PER_EPOCH="${STV_REINFORCE_SAMPLES_PER_EPOCH:-16}"
REINFORCE_LR="${STV_REINFORCE_LR:-0.1}"
REINFORCE_EVAL_EVERY="${STV_REINFORCE_EVAL_EVERY:-50}"
REINFORCE_FINAL_NUM_SAMPLES="${STV_REINFORCE_FINAL_NUM_SAMPLES:-10}"

MM_PROJECTOR_LR=0 #3e-4
FINAL_LR=$LR #3e-4
MM_FINAL_LR=$LR #3e-4
OPT_NAME="adamw_torch" # adam8bit_bnb adamw_torch
SCHED_NAME="cosine" #cosine
WARMUP_RATIO=0.1 # SHOULD BE 0.03 / NUM_ROUNDS
DECAY_RATIO=0.9

mkdir -p "nohup"
NOTE=${MODE}_${MODEL_ARCH}_${DATASET_NAME}_${EPOCH}_${SHOT}_scen${SCENARIO}

BF16=True
if [ "$LR_SUFFIX" == "True" ]; then
    NOTE=${NOTE}_${LR}
fi

CKPT="${CKPT_ROOT}/client_states_${NOTE}/server_model_round0.pth"
echo "[train_VLM.sh] METHOD=$METHOD -> MODE=$MODE SHOT=$SHOT SCENARIO=$SCENARIO LR=$LR LoRA=$LORA_ENABLE attn=$ATTEN_IMPLE dataset=$DATASET_NAME"
echo "[train_VLM.sh] NOTE=$NOTE"
echo "[train_VLM.sh] checkpoint -> $CKPT"
if [ -f "$CKPT" ]; then
    if [ "$FORCE" == "1" ]; then
        echo "[train_VLM.sh] checkpoint exists, FORCE=1 -> retraining (it will be overwritten)"
    else
        echo "[train_VLM.sh] checkpoint already exists -> nothing to do (set FORCE=1 to retrain)"
        exit 0
    fi
fi
if [ "$DRY_RUN" == "1" ]; then
    exit 0
fi
echo $NOTE
if [ "$METHOD" == "stv" ]; then
    # STV (Ma et al., AAAI26): sensitivity-selected heads + per-head KMeans bank + REINFORCE
    # selection logits (no LoRA). Writes the steering artifact to $CKPT. See stv_train.py.
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPU \
    python stv_train.py \
        --model_name_or_path $MODEL_NAME --model_name_for_dataarg $MODEL_NAME \
        --model_type $MODEL_TYPE --version $VERSION --vision_tower $VISION_TOWER \
        --bits $BITS --bf16 $BF16 --attn_impl $ATTEN_IMPLE --mode stv --is_eval True --lora_enable False \
        --dataset_name $DATASET_NAME --note $NOTE --seed $RND_SEED --output_dir "./results/test/" \
        --intervention_topk "$INTERVENTION_TOPK" \
        --intervention_num_clusters "$INTERVENTION_NUM_CLUSTERS" \
        --intervention_n_trials "$INTERVENTION_N_TRIALS" \
        --reinforce_n_epochs "$REINFORCE_N_EPOCHS" \
        --reinforce_samples_per_epoch "$REINFORCE_SAMPLES_PER_EPOCH" \
        --reinforce_lr "$REINFORCE_LR" \
        --reinforce_final_num_samples "$REINFORCE_FINAL_NUM_SAMPLES" \
        --stv_rl_items "${STV_RL_ITEMS:-64}" --stv_val_items "${STV_VAL_ITEMS:-16}" \
        --stv_max_answer_tokens "${STV_MAX_ANSWER_TOKENS:-16}" > ./nohup/${NOTE}.log 2>&1 &
    exit 0
fi
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
deepspeed --master_port $MASTER_PORT \
    --include localhost:$GPU \
    train_VLM.py \
    --deepspeed ./deepspeed_script/zero2.json \
    --dataset_name $DATASET_NAME \
    --model_name_or_path $MODEL_NAME \
    --model_name_for_dataarg $MODEL_NAME \
    --model_type $MODEL_TYPE \
    --version $VERSION \
    --model_max_length $MODEL_MAX_LEN \
    --num_tasks $NUM_TASKS \
    --scenario $SCENARIO \
    --vision_tower $VISION_TOWER \
    --gradient_checkpointing $GRAD_CHECKPOINT \
    --num_train_epochs $EPOCH \
    --num_iter $NUM_ITER \
    --gradient_accumulation_steps $GRAD_ACC \
    --max_steps $MAX_STEPS \
    --bits $BITS \
    --bf16 $BF16 \
    --tf32 False \
    --mode $MODE --dataloader_num_workers 2 \
    --intervention_mode "$INTERVENTION_MODE" \
    --intervention_topk "$INTERVENTION_TOPK" \
    --intervention_num_clusters "$INTERVENTION_NUM_CLUSTERS" \
    --intervention_n_trials "$INTERVENTION_N_TRIALS" \
    --intervention_num_shot "$INTERVENTION_NUM_SHOT" \
    --reinforce_n_epochs "$REINFORCE_N_EPOCHS" \
    --reinforce_samples_per_epoch "$REINFORCE_SAMPLES_PER_EPOCH" \
    --reinforce_lr "$REINFORCE_LR" \
    --reinforce_eval_every "$REINFORCE_EVAL_EVERY" \
    --reinforce_final_num_samples "$REINFORCE_FINAL_NUM_SAMPLES" \
    --seed $RND_SEED \
    --optim $OPT_NAME \
    --lr_scheduler_type $SCHED_NAME \
    --weight_decay 0. \
    --warmup_ratio $WARMUP_RATIO \
    --decay_ratio $DECAY_RATIO \
    --learning_rate $LR --per_gpu_train_batch_size $BATCHSIZE \
    --final_lr $FINAL_LR --mm_final_lr $MM_FINAL_LR \
    --mm_projector_lr $MM_PROJECTOR_LR \
    --save_strategy "no" \
    --logging_steps 2 \
    --note $NOTE \
    --memory_size $MEMORY_SIZE \
    --is_streamonly $IS_STREAMONLY \
    --prompt_num 1 \
    --lora_enable $LORA_ENABLE \
    --get_prompt $USE_PROMPT \
    --generator_output_size $GENERATOR_OUTPUT_SIZE \
    --generator_hidden_dim $GENERATOR_HIDDEN_DIM \
    --generator_hidden_feature $GENERATOR_HIDDEN_FEATURE \
    --key_embed_size $KEY_EMBED_SIZE \
    --pool_size $POOL_SIZE \
    --prompt_top_k $PROMPT_TOP_K \
    --save_optim $SAVE_OPTIM \
    --use_task_vector $USE_TASK_VECTOR \
    --use_fisher $USE_FISHER \
    --eval_period $EVAL_PERIOD \
    --resume_train_round $RESUME_TRAIN_ROUND \
    --load_checkpoint $LOAD_CHECKPOINT \
    --shot $SHOT \
    --attn_impl $ATTEN_IMPLE \
    --train_scenario $SCENARIO \
    --vis_token_amplifier $VIS_TOKEN_AMPLIFIER \
    --state_dir "${CKPT_ROOT}/client_states" \
    --output_dir "./results/test/" > ./nohup/${NOTE}.log 2>&1 &
