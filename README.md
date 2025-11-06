# MTD_RL
강화학습 코드
코랩으로 가져가서 하세여~

#@title [RUN] 깨끗한 학습 스크립트 (시나리오 실행)
# 이 셀을 실행하여 모든 시나리오를 순서대로 실행하세요.

import os
import sys

# --- 1. 경로 설정 및 Google Drive 마운트 ---
drive_path = "/content/drive/MyDrive/dvd_attacks_lpc/marl_agent7"

if 'google.colab' in sys.modules:
    if not os.path.exists("/content/drive"):
        from google.colab import drive
        print("Google Drive 마운트 중...")
        drive.mount('/content/drive')
    
    if not os.path.exists(drive_path):
        os.makedirs(drive_path)
        print(f"디렉토리 생성: {drive_path}")
    
    # 작업 디렉토리로 이동 (필수)
    os.chdir(drive_path)
    print(f"현재 작업 디렉토리: {os.getcwd()}")
else:
    print("Google Colab 환경이 아닙니다. 로컬에서 실행합니다.")
    drive_path = "." # 로컬 환경일 경우 현재 디렉토리

# --- 2. train.py 존재 여부 확인 ---
if not os.path.exists("train.py"):
    print("="*50)
    print(f" [오류] train.py 파일을 찾을 수 없습니다!")
    print(f" [확인] 현재 경로: {os.getcwd()}")
    print(f" [조치] 이전 단계에서 %%writefile 셀을 실행하여 train.py와 7개의 다른 .py 파일을 생성했는지 확인하세요.")
    print("="*50)
else:
    print("train.py 확인 완료. 시나리오 실행을 시작합니다.")
    print("="*50)

    # --- 시나리오 1 ---
    print("\nStarting Scenario 1: ARL (MTD) vs ARL (Seeker)")
    print("-" * 50)
    !python train.py --mtd-mode arl --seeker-mode arl --updates 100 --seed 42

    # --- 시나리오 2a ---
    print("\nStarting Scenario 2a: ARL (MTD) vs Static Seeker (L0 Naive)")
    print("-" * 50)
    !python train.py --mtd-mode arl --seeker-mode static_behavior --seeker-level 0 --updates 100 --seed 100

    # --- 시나리오 2b ---
    print("\nStarting Scenario 2b: ARL (MTD) vs Static Seeker (L1 Scanner)")
    print("-" * 50)
    !python train.py --mtd-mode arl --seeker-mode static_behavior --seeker-level 1 --updates 100 --seed 101

    # --- 시나리오 2c ---
    print("\nStarting Scenario 2c: ARL (MTD) vs Static Seeker (L2 Stealthy)")
    print("-" * 50)
    !python train.py --mtd-mode arl --seeker-mode static_behavior --seeker-level 2 --updates 100 --seed 102

    # --- 시나리오 3a ---
    print("\nStarting Scenario 3a: Static MTD (L0) vs ARL (Seeker)")
    print("-" * 50)
    !python train.py --mtd-mode static --mtd-level 0 --seeker-mode arl --updates 100 --seed 200

    # --- 시나리오 3b ---
    print("\nStarting Scenario 3b: Static MTD (L2) vs ARL (Seeker)")
    print("-" * 50)
    !python train.py --mtd-mode static --mtd-level 2 --seeker-mode arl --updates 100 --seed 201

    # --- 시나리오 3c ---
    print("\nStarting Scenario 3c: Static MTD (L5) vs ARL (Seeker)")
    print("-" * 50)
    !python train.py --mtd-mode static --mtd-level 5 --seeker-mode arl --updates 100 --seed 202

    print("\n" + "="*50)
    print("모든 시나리오 실행 완료.")
    print(f"결과는 '{drive_path}/results' 디렉토리에서 확인할 수 있습니다.")
    print("="*50)
