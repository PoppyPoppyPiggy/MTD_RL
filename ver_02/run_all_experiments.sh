#!/usr/bin/env bash
# ver_02/run_all_experiments.sh — v23 호환, 견고성 강화
# [MODIFIED] --wandb_group 인자를 명시적으로 전달하여 로그 그룹 통일

set -Eeuo pipefail
# ... (기존 코드와 동일) ...
trap 'echo "[ERROR] 라인 $LINENO 에서 실패했습니다."; exit 1' ERR

echo "=========================================================="
# ... (기존 코드와 동일) ...
echo "=========================================================="

# 학습할 레벨 목록
SEEKER_LEVELS=("L0" "L1" "L2" "L3" "L4")

# 총 에피소드 수 (환경변수로 덮어쓰기 가능: MAX_EPISODES=500 ./run_all_experiments.sh)
# [MODIFIED] 테스트를 위해 기본 에피소드 수를 1000 -> 500으로 줄임 (원래대로 1000으로 두셔도 됩니다)
MAX_EPISODES="${MAX_EPISODES:-500}"

# 기본 저장 경로(Colab 기준). 환경변수로 덮어쓰기 가능
# ... (기존 코드와 동일) ...
BASE_LOG_DIR="${BASE_LOG_DIR:-$DEFAULT_BASE/logs}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-$DEFAULT_BASE/models}"

# Python 실행기 선택
# ... (기존 코드와 동일) ...
if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "[ERROR] python 실행기를 찾을 수 없습니다."; exit 1
fi

# 스크립트 위치 기준으로 이동 후 학습 스크립트 존재 확인
# ... (기존 코드와 동일) ...
cd "$SCRIPT_DIR"
if [[ ! -f "train_mtd_only.py" ]]; then
  echo "[ERROR] $SCRIPT_DIR 에 train_mtd_only.py 가 없습니다."; exit 1
fi

# [MODIFIED] W&B 프로젝트 이름 (rl_driven_deception_manager.py와 일치)
WANDB_PROJECT="mtd_testbed_live"

for LEVEL in "${SEEKER_LEVELS[@]}"; do
  echo ""
# ... (기존 코드와 동일) ...
  echo "[!] Seeker 레벨: $LEVEL 에 대한 학습을 시작합니다."
  echo "-----------------------------------------------------"

  # 레벨별 경로/파일
  LOG_PATH="${BASE_LOG_DIR}/${LEVEL}_Seeker"
  MODEL_PATH="${BASE_MODEL_DIR}/${LEVEL}_Seeker"
  POLICY_NAME="defender_policy_${LEVEL}.pth"
  
  # [MODIFIED] W&B 그룹 이름 (rl_driven_deception_manager.py와 일치시킴)
  WANDB_GROUP="vs_${LEVEL}_Seeker"

  # 디렉터리 생성
  mkdir -p "$LOG_PATH" "$MODEL_PATH"

  # 학습 실행 (v23 대응 인자)
  echo "[CMD] $PY train_mtd_only.py --wandb --seeker_level $LEVEL --max_episodes $MAX_EPISODES --save_dir $MODEL_PATH --policy_name $POLICY_NAME --wandb_project $WANDB_PROJECT --wandb_group $WANDB_GROUP"
  
  "$PY" train_mtd_only.py \
      --wandb \
      --seeker_level "$LEVEL" \
      --max_episodes "$MAX_EPISODES" \
      --save_dir "$MODEL_PATH" \
      --policy_name "$POLICY_NAME" \
      --wandb_project "$WANDB_PROJECT" \
      --wandb_group "$WANDB_GROUP"
      # [REMOVED] --log_dir "$LOG_PATH" (train_mtd_only.py가 W&B 로그만 사용하므로 제거)

  echo "[!] $LEVEL 레벨 학습 완료. 모델 저장 위치: $MODEL_PATH/$POLICY_NAME"
# ... (기존 코드와 동일) ...
done

echo "=========================================================="
# ... (기존 코드와 동일) ...
echo "=========================================================="