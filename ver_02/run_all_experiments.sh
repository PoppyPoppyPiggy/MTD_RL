#!/usr/bin/env bash
# [수정본] ver_03/config.py (argparse) 호환 스크립트
# - DEFAULT_BASE 변수 정의 순서 오류 수정
# - ver_03/config.py에 정의된 인자(argparse) 기준으로 명령어 수정

set -Eeuo pipefail
IFS=$'\n\t'
trap 'echo "[ERROR] 라인 $LINENO 에서 실패했습니다."; exit 1' ERR

echo "=========================================================="
echo "MTD 정책 학습 시작 (v23 - argparse 기반)"
echo "Seeker 레벨 L0 ~ L4"
echo "=========================================================="

# 학습할 레벨 목록
SEEKER_LEVELS=("L0" "L1" "L2" "L3" "L4")

# 총 에피소드 수 (환경변수로 덮어쓰기 가능)
MAX_EPISODES="${MAX_EPISODES:-500}" # 테스트를 위해 500으로 설정 (원래 1000 또는 50000)

# [수정] DEFAULT_BASE를 먼저 정의해야 합니다.
# Colab 또는 로컬 환경에 맞게 이 경로를 수정하세요.
DEFAULT_BASE="/content/drive/MyDrive/MTD_RL_Testbed_v25"
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

# 스크립트 위치 기준으로 이동 (ver_03 디렉토리라고 가정)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ver_03/config.py는 train_mtd_only.py가 아닌, 별도의 config 파일입니다.
# train_mtd_only.py가 같은 디렉토리에 있다고 가정합니다.
if [[ ! -f "train_mtd_only.py" ]]; then
  echo "[ERROR] $SCRIPT_DIR 에 train_mtd_only.py 가 없습니다."; exit 1
fi

# W&B 프로젝트 이름 (config.py의 --project_name 기본값과 일치시킴)
WANDB_PROJECT="MTD-RL-Testbed-Trainer-v23"

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

  # 학습 실행 (ver_03/config.py의 argparse 인자 사용)
  "$PY" train_mtd_only.py \
      --wandb \
      --seeker_level "$LEVEL" \
      --max_episodes "$MAX_EPISODES" \
      --save_dir "$MODEL_PATH" \
      --policy_name "$POLICY_NAME" \
      --log_dir "$LOG_PATH" \
      --project_name "$WANDB_PROJECT"
      # 참고: ver_03/config.py에는 --wandb_group 인자가 없습니다.
      # W&B 그룹화는 train_mtd_only.py 내부에서 --seeker_level을 기반으로 처리해야 합니다.

  echo "[!] $LEVEL 레벨 학습 완료. 모델 저장 위치: $MODEL_PATH/$POLICY_NAME"
  echo "-----------------------------------------------------"
done

echo "=========================================================="
echo "모든 Seeker 레벨(L0~L4)에 대한 학습이 완료되었습니다."
echo "=========================================================="