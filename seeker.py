#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RL-Driven Seeker (The "Attacker Brain") - 완성본
- Seeker 정책(.pth)을 로드합니다.
- MTD State Reader (Eyes)로부터 8D 상태(State)를 주기적으로 읽어옵니다.
- [수정] 'attack_orchestrator'를 실제 호출하여 공격 셸 스크립트를 실행합니다.
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
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from rl.ppo import ActorCritic

# MTD (Eyes) 모듈 임포트
try:
    import mtd_state_reader
except ImportError:
    # `mtd` 폴더가 아닌 상위 폴더(dvd_attacks_lpc)에서 실행될 경우를 대비
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mtd'))
    import mtd_state_reader

# --- Seeker 인터페이스 (시뮬레이터와 다를 수 있음) ---
SEEKER_OBS_DIM = 8 
SEEKER_ACTION_DIM = 4 
# ------------------------------------------------

class SeekerHands:
    """[수정] Seeker의 행동(Action ID)을 실제 공격 스크립트로 변환 (TODO 제거)"""
    def __init__(self, attack_orchestrator_path):
        if not os.path.exists(attack_orchestrator_path):
             print(f"[Seeker-Hands] Warning: {attack_orchestrator_path} 경로를 찾을 수 없습니다.", file=sys.stderr)
             self.orchestrator = None
        else:
            self.orchestrator = attack_orchestrator_path
            print(f"[Seeker-Hands] Attack Orchestrator 경로: {self.orchestrator}")
        
    def execute_attack_action_by_id(self, action_id: int):
        action_name = "pass"
        cmd = None
        
        if not self.orchestrator:
            print(f"[Seeker-Hands] Error: Orchestrator가 없어 공격을 실행할 수 없습니다 (Action ID: {action_id}).", file=sys.stderr)
            return

        if action_id == 1:
            action_name = "Scan (wifi_slow_scan)"
            cmd = ["python", self.orchestrator, "start", "wifi_slow_scan", "-d", "5"] # 5초간 실행
        elif action_id == 2:
            action_name = "Exploit (gps_slow_spoof)"
            cmd = ["python", self.orchestrator, "start", "gps_slow_spoof", "-d", "5"]
        elif action_id == 3:
            action_name = "Breach (companion-computer-takeover)"
            cmd = ["python", self.orchestrator, "start", "companion-computer-takeover", "-d", "5"]
            
        if cmd:
            print(f"[Seeker-Hands] Action: {action_name} (ID: {action_id}) 실행...")
            try:
                # [수정] 실제 subprocess 실행 (TODO 제거)
                # (비동기 실행: 공격이 백그라운드에서 5초간 실행되도록 Popen 사용)
                subprocess.Popen(cmd)
            except Exception as e:
                print(f"    -> [Seeker-Hands] Error: 공격 실행 실패: {e}", file=sys.stderr)
        else:
             print(f"[Seeker-Hands] Action: Pass (ID: {action_id}).")

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
        print(f"[Seeker-Brain] Seeker 정책 로드 완료.")
    except Exception as e:
        print(f"Error: Seeker 정책 파일 로드 실패. (시뮬레이터와 Seeker 모델 구조가 다른가요?) \n{e}", file=sys.stderr)
        sys.exit(1)

    # 2. Seeker (Hands) 컨트롤러 초기화
    controller = SeekerHands(
        attack_orchestrator_path=args.orchestrator
    )
    
    # 3. 실시간 공격 루프 시작
    print("[Seeker-Brain] 실시간 공격 루프를 시작합니다...")
    while True:
        try:
            # 3a. [Eyes] 현재 MTD 시스템 상태 관측 (8D Vector)
            current_state = mtd_state_reader.get_rl_state()
            state_tensor = torch.FloatTensor(current_state).to(device)

            # 3b. [Brain] 정책을 기반으로 공격 행동 결정
            with torch.no_grad():
                action_id, _ = policy.act(state_tensor)

            # 3c. [Hands] 결정된 행동(Action ID)을 실제 시스템에 적용
            controller.execute_attack_action_by_id(action_id)

            # 3d. 공격 실행 주기에 따라 대기
            time.sleep(args.interval)

        except KeyboardInterrupt:
            print("\n[Seeker-Brain] Seeker 중지 신호 수신. 종료합니다.")
            break
        except Exception as e:
            print(f"[Seeker-Brain] Seeker 루프 오류 발생: {e}", file=sys.stderr)
            time.sleep(args.interval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL-Driven Seeker")
    parser.add_argument('--policy', type=str, required=True, help='학습된 Seeker 정책 파일 (.pth) 경로')
    parser.add_argument('--orchestrator', type=str, default='attack_orchestrator.py', help='Attack Orchestrator 스크립트 경로')
    parser.add_argument('--interval', type=int, default=8, help='공격 실행 주기 (초)')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='Device (cuda/cpu)')
    args = parser.parse_args()
    
    main(args)