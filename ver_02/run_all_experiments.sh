#!/usr/bin/env bash
# ver_02/run_all_experiments.sh — v23 호환, 견고성 강화

set -Eeuo pipefail
IFS=$'\n\t'
trap 'echo "[ERROR] 라인 $LINENO 에서 실패했습니다."; exit 1' ERR

echo "=========================================================="
echo "MTD 정책 학습 시작 (v23 - 배포 환경 호환)"
echo "Seeker 레벨 L0 ~ L4"
echo "=========================================================="

# 학습할 레벨 목록
SEEKER_LEVELS=("L0" "L1" "L2" "L3" "L4")

# 총 에피소드 수 (환경변수로 덮어쓰기 가능: MAX_EPISODES=500 ./run_all_experiments.sh)
MAX_EPISODES="${MAX_EPISODES:-1000}"

# 기본 저장 경로(Colab 기준). 환경변수로 덮어쓰기 가능
DEFAULT_BASE="/content/drive/MyDrive/MTD_RL_Testbed_v23"
BASE_LOG_DIR="${BASE_LOG_DIR:-$DEFAULT_BASE/logs}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-$DEFAULT_BASE/models}"

# Python 실행기 선택
if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "[ERROR] python 실행기를 찾을 수 없습니다."; exit 1
fi

# 스크립트 위치 기준으로 이동 후 학습 스크립트 존재 확인
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
if [[ ! -f "train_mtd_only.py" ]]; then
  echo "[ERROR] $SCRIPT_DIR 에 train_mtd_only.py 가 없습니다."; exit 1
fi

for LEVEL in "${SEEKER_LEVELS[@]}"; do
  echo ""
  echo "-----------------------------------------------------"
  echo "[!] Seeker 레벨: $LEVEL 에 대한 학습을 시작합니다."
  echo "-----------------------------------------------------"

  # 레벨별 경로/파일
  LOG_PATH="${BASE_LOG_DIR}/${LEVEL}_Seeker"
  MODEL_PATH="${BASE_MODEL_DIR}/${LEVEL}_Seeker"
  POLICY_NAME="defender_policy_${LEVEL}.pth"

  # 디렉터리 생성
  mkdir -p "$LOG_PATH" "$MODEL_PATH"

  # 학습 실행 (v23 대응 인자)
  "$PY" train_mtd_only.py \
      --wandb \
      --log_dir "$LOG_PATH" \
      --save_dir "$MODEL_PATH" \
      --seeker_level "$LEVEL" \
      --max_episodes "$MAX_EPISODES" \
      --policy_name "$POLICY_NAME"

  echo "[!] $LEVEL 레벨 학습 완료. 모델 저장 위치: $MODEL_PATH/$POLICY_NAME"
  echo "-----------------------------------------------------"
done

echo "=========================================================="
echo "모든 Seeker 레벨(L0~L4)에 대한 학습이 완료되었습니다."
echo "=========================================================="
