#!/bin/bash
# 'rl/ver_03'용 학습 스크립트 실행기
# 'rl/ver_02'의 run_all_experiments.sh와 동일한 구조로 생성됨.

# Define seeker levels (L0 to L4)
# ver_03의 MTDEnv가 시커 레벨을 인자로 받도록 수정되었다고 가정합니다.
SEEKER_LEVELS=(0 1 2 3 4)

# Loop through each seeker level and run the training
for level in "${SEEKER_LEVELS[@]}"
do
    echo "------------------------------------------------"
    echo "Running ver_03 experiment for Seeker Level: L$level"
    echo "------------------------------------------------"
    
    # Define the output directory for this level
    OUTPUT_DIR="models/L${level}_Seeker_v3"
    
    # Create the directory if it doesn't exist
    mkdir -p $OUTPUT_DIR
    
    # Run the training script (ver_03)
    # --seeker_level 인자가 train_mtd_only.py에 구현되어 있어야 함.
    # --output_policy 인자가 train_mtd_only.py에 구현되어 있어야 함.
    python train_mtd_only.py --seeker_level $level --output_policy "$OUTPUT_DIR/defender_policy_v3_L$level.pth"
    
    echo "Experiment for L$level completed."
    echo "Policy saved to $OUTPUT_DIR/defender_policy_v3_L$level.pth"
done

echo "------------------------------------------------"
echo "All ver_03 experiments completed."
echo "------------------------------------------------"