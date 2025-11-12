#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RL-Driven Seeker (The "Attacker Brain") - v2 (고도화)
- Seeker 정책(.pth)을 로드합니다.
- MTD State Reader (Eyes)로부터 8D 상태(State)를 주기적으로 읽어옵니다.
- [고도화] RL 정책 결정에 Heuristic 로직을 추가합니다.
- [고D화] 'attack_orchestrator'를 실제 호출하여 공격 셸 스크립트를 실행합니다.
- [고도화] 로깅을 강화합니다.
"""

import os
import sys
import time
import argparse
import torch
import numpy as np
import subprocess

# [중요] MTD와 동일한 ActorCritic 구조를 공유한다고 가정
try:
    from rl.ppo import ActorCritic
except ImportError:
    # `rl` 폴더가 아닌 상위 폴더(dvd_attacks_lpc)에서 실행될 경우를 대비
    # [수정] 경로가 MTD_full_testbed/dvd_lite/dvd_attacks_lpc/rl 이어야 함
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'dvd_lite', 'dvd_attacks_lpc', 'rl'))
    from ppo import ActorCritic

# MTD (Eyes) 모듈 임포트
try:
    import mtd_state_reader
except ImportError:
    # `mtd` 폴더가 아닌 상위 폴더(dvd_attacks_lpc)에서 실행될 경우를 대비
    # [수정] 경로가 MTD_full_testbed/dvd_lite/dvd_attacks_lpc/mtd 이어야 함
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'dvd_lite', 'dvd_attacks_lpc', 'mtd'))
    import mtd_state_reader

# --- Seeker 인터페이스 (시뮬레이터와 다를 수 있음) ---
SEEKER_OBS_DIM = 8 
SEEKER_ACTION_DIM = 4 
# ------------------------------------------------

class SeekerHands:
    """[수정] Seeker의 행동(Action ID)을 실제 공격 스크립트로 변환 (로깅 강화)"""
    def __init__(self, attack_orchestrator_path):
        self.orchestrator = os.path.abspath(attack_orchestrator_path)
        if not os.path.exists(self.orchestrator):
             print(f"[Seeker-Hands] Error: Attack Orchestrator를 찾을 수 없습니다: {self.orchestrator}", file=sys.stderr)
             self.orchestrator = None
        else:
            print(f"[Seeker-Hands] Attack Orchestrator 경로: {self.orchestrator}")
        
    def execute_attack_action_by_id(self, action_id: int, interval_sec: int):
        action_name = "pass"
        cmd = None
        
        if not self.orchestrator:
            print(f"[Seeker-Hands] Error: Orchestrator가 없어 공격을 실행할 수 없습니다 (Action ID: {action_id}).", file=sys.stderr)
            return

        # [고도화] 공격 실행 시간을 interval에 맞게 -d 인자로 전달
        duration_arg = str(int(interval_sec * 0.8)) # 주기의 80%만 실행

        if action_id == 1:
            action_name = "Scan (wifi_slow_scan)"
            cmd = ["python3", self.orchestrator, "start", "wifi_slow_scan", "-d", duration_arg]
        elif action_id == 2:
            action_name = "Exploit (gps_slow_spoof)"
            cmd = ["python3", self.orchestrator, "start", "gps_slow_spoof", "-d", duration_arg]
        elif action_id == 3:
            action_name = "Breach (companion-computer-takeover)"
            cmd = ["python3", self.orchestrator, "start", "companion-computer-takeover", "-d", duration_arg]
            
        if cmd:
            print(f"[Seeker-Hands] 🚀 Action: {action_name} (ID: {action_id}) 실행... (지속시간: {duration_arg}s)")
            print(f"    -> CMD: {' '.join(cmd)}")
            try:
                # [수정] 실제 subprocess 실행 (비동기)
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print("    -> [Seeker-Hands] 공격이 백그라운드에서 시작되었습니다.")
            except Exception as e:
                print(f"    -> [Seeker-Hands] Error: 공격 실행 실패: {e}", file=sys.stderr)
        else:
             print(f"[Seeker-Hands] 😴 Action: Pass (ID: {action_id}).")

def main(args):
    device = torch.device(args.device)
    
    # 1. Seeker 정책 로드
    print(f"[Seeker-Brain] Seeker 정책 로딩 중: {args.policy}")
    
    # (가정) Seeker도 MTD와 동일한 ActorCritic 구조를 사용 (State=8D, Action=4D)
    policy = ActorCritic(SEEKER_OBS_DIM, SEEKER_ACTION_DIM).to(device)
    
    if not os.path.exists(args.policy):
        print(f"Error: Seeker 정책 파일({args.policy})을 찾을 수 없습니다!", file=sys.stderr)
        sys.exit(1)
        
    try:
        policy.load_state_dict(torch.load(args.policy, map_location=device))
        policy.eval() # 평가 모드
        print(f"[SeekK-Brain] Seeker 정책 로드 완료.")
    except Exception as e:
        print(f"Error: Seeker 정책 파일 로드 실패. (시뮬레이터와 Seeker 모델 구조가 다른가요?) \n{e}", file=sys.stderr)
        sys.exit(1)

    # 2. Seeker (Hands) 컨트롤러 초기화
    controller = SeekerHands(
        attack_orchestrator_path=args.orchestrator
    )
    
    # 3. MTD 상태 리더 (Eyes) 초기화
    # [고도화] Seeker도 MTD와 동일한 config를 읽어야 함
    if not args.mtd_config:
        print("[Seeker-Eyes] Error: --mtd_config 경로가 필요합니다. (예: mtd/configs/iptables_mtd.yaml)", file=sys.stderr)
        sys.exit(1)
        
    print(f"[Seeker-Eyes] MTD 상태 리더 초기화 (Config: {args.mtd_config})")
    try:
        # [수정] mtd_state_reader의 클래스 인스턴스 사용
        state_reader = mtd_state_reader.MTDStateReader(config_path=args.mtd_config)
        print("[Seeker-Eyes] MTD 상태 리더(Eyes) 초기화 완료.")
    except Exception as e:
        print(f"[Seeker-Eyes] MTD 상태 리더 초기화 실패: {e}", file=sys.stderr)
        sys.exit(1)

    # 4. 실시간 공격 루프 시작
    print("[Seeker-Brain] 실시간 공격 루프를 시작합니다...")
    while True:
        try:
            print("-" * 30) # 주기 구분을 위한 라인

            # 4a. [Eyes] 현재 MTD 시스템 상태 관측 (8D Vector)
            current_state = state_reader.get_rl_state()
            print(f"[Seeker-Eyes] MTD 상태 관측 (8D): {current_state.tolist()}")
            state_tensor = torch.FloatTensor(current_state).to(device)

            # 4b. [Brain] 정책을 기반으로 공격 행동 결정
            with torch.no_grad():
                action_id, _ = policy.act(state_tensor)
            print(f"[Seeker-Brain] RL 정책 결정: {action_id}")

            # --- [고도화] Heuristic Override Logic ---
            # 8D State: [R1, R2, R3, R4, R5, R6, Decoy, Alert]
            is_decoy_active = current_state[6] > 0.1 # 8D State의 7번째 요소 (인덱스 6)
            
            if action_id == 0 and is_decoy_active:
                print("[Seeker-Brain] (Heuristic) RL이 'Pass'를 선택했으나 Decoy가 활성 상태입니다. 'Scan' (1)으로 재정의!")
                action_id = 1
            # ----------------------------------------

            # 4c. [Hands] 결정된 행동(Action ID)을 실제 시스템에 적용
            controller.execute_attack_action_by_id(action_id, args.interval)

            # 4d. 공격 실행 주기에 따라 대기
            print(f"[Seeker-Brain] (t_sleep) 다음 주기까지 {args.interval}초 대기...")
            time.sleep(args.interval)

        except KeyboardInterrupt:
            print("\n[Seeker-Brain] Seeker 중지 신호 수신. 종료합니다.")
            break
        except Exception as e:
            print(f"[Seeker-Brain] Seeker 루프 오류 발생: {e}", file=sys.stderr)
            time.sleep(args.interval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL-Driven Seeker (v2 - 고도화)")
    parser.add_argument('--policy', type=str, required=True, help='학습된 Seeker 정책 파일 (.pth) 경로')
    parser.add_argument('--orchestrator', type=str, default='attack_orchestrator.py', help='Attack Orchestrator 스크립트 경로')
    
    # [고도화] MTD Config 경로를 인자로 받도록 수정
    parser.add_argument('--mtd_config', type=str, default='mtd/configs/iptables_mtd.yaml', help='MTD 상태 리더(Eyes)를 위한 YAML 설정 파일 경로')
    
    parser.add_argument('--interval', type=int, default=8, help='공격 실행 주기 (초)')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='Device (cuda/cpu)')
    args = parser.parse_args()
    
    main(args)