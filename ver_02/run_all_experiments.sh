File: ver_02/run_all_experiments.sh

#!/bin/bash

모든 Seeker 레벨 (L0-L4)에 대해 MTD 정책 학습을 실행합니다. (v23 호환)

[수정] train_mtd_only.py (v23)를 호출합니다.

echo "=========================================================="
echo "MTD 정책 학습 시작 (v23 - 배포 환경 호환)"
echo "Seeker 레벨 L0 ~ L4"
echo "=========================================================="

학습할 레벨 목록

SEEKER_LEVELS=("L0" "L1" "L2" "L3" "L4")

총 에피소드 수 (빠른 테스트를 원하면 50000 -> 10000 등으로 줄이세요)

MAX_EPISODES=1000

Google Drive 내의 기본 저장 경로 (원하는 경로로 수정 가능)

BASE_LOG_DIR="/content/drive/MyDrive/MTD_RL_Testbed_v25/logs"
BASE_MODEL_DIR="/content/drive/MyDrive/MTD_RL_Testbed_v25/models"

for LEVEL in "${SEEKER_LEVELS[@]}"
do
echo ""
echo "-----------------------------------------------------"
echo "[!] Seeker 레벨: $LEVEL 에 대한 학습을 시작합니다."
echo "-----------------------------------------------------"

# 레벨별로 고유한 경로 및 파일 이름 설정
LOG_PATH="${BASE_LOG_DIR}/${LEVEL}_Seeker"
MODEL_PATH="${BASE_MODEL_DIR}/${LEVEL}_Seeker"
POLICY_NAME="defender_policy_${LEVEL}.pth"

# [수정] train_mtd_only.py (v25)를 실행
python train_mtd_only.py \
    --wandb \
    --log_dir $LOG_PATH \
    --save_dir $MODEL_PATH \
    --seeker_level $LEVEL \
    --max_episodes $MAX_EPISODES \
    --policy_name $POLICY_NAME
    
echo "[!] $LEVEL 레벨 학습 완료. 모델 저장 위치: $MODEL_PATH/$POLICY_NAME"
echo "-----------------------------------------------------"


done

echo "=========================================================="
echo "모든 Seeker 레벨(L0~L4)에 대한 학습이 완료되었습니다."
echo "=========================================================="