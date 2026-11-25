#!/bin/bash
# GPU job-queue worker. Usage: worker.sh <GPU> <QUEUE_FILE> <MASTER_PORT>
# Queue line: "<ENV assignments> :: <train|eval> [:: <needed files...>]".
# pop.py (under flock) takes the first line whose needed files all exist.
G=$1; Q=$2; PORT=$3
cd /tmp/procore
export PATH=/tmp/procore/venv/bin:/tmp/procore/jdk/jdk-17.0.21.1+2-jre/bin:$PATH HF_HOME=/tmp/hf_cache NLTK_DATA=/tmp/nltk_data
export CKPT_ROOT=/tmp/procore/checkpoints_verified
log(){ echo "[$(date +%F_%T)] gpu$G $*" | tee -a nohup/queue/queue.log; }
pop(){ ( flock 9; python3 tools/queue/pop.py "$Q" ) 9>"$Q.lock"; }
alive(){ ps -eo stat=,args= | grep -v "^Z" | grep -v -e "grep" | grep -F -- "--note $1 " >/dev/null; }
gpu_busy(){ [ "$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -gt 2000 ]; }
while true; do
  while gpu_busy; do sleep 120; done
  L=$(pop)
  if [ -z "$L" ]; then
    if [ -s "$Q" ]; then sleep 300; continue; fi
    log "queue empty, exit"; break
  fi
  ENVS=${L%% :: *}; REST=${L#* :: }; KIND=${REST%% :: *}
  log "START $KIND $ENVS"
  if [ "$KIND" = train ]; then
    OUT=$(eval "env $ENVS GPU=$G MASTER_PORT=$PORT bash train_VLM.sh" 2>&1); echo "$OUT" >> nohup/queue/gpu$G.out
    NOTE=$(echo "$OUT" | sed -n 's/^\[train_VLM.sh\] NOTE=//p' | tail -1)
    if echo "$OUT" | grep -q "already exists"; then log "SKIP exists $NOTE"; continue; fi
    sleep 90; while [ -n "$NOTE" ] && alive "$NOTE"; do sleep 60; done
    if [ -f "$CKPT_ROOT/client_states_$NOTE/server_model_round0.pth" ]; then log "DONE train $NOTE"; else log "FAIL train $NOTE (see nohup/$NOTE.log)"; fi
  else
    eval "env $ENVS GPU=$G bash eval.sh $G" >> nohup/queue/gpu$G.out 2>&1; log "END eval rc=$? $ENVS"
  fi
done
