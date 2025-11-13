#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MTD State Reader (The "Eyes") - v2 (JSON-based)

- 'Brain'(rl_driven_deception_manager.py)과 'Seeker'(mtd/seeker.py)가 호출하는 모듈.
- 3개의 핵심 JSON 파일 (Action, CTI, Health)을 읽어옵니다.
- 8차원(8D) 상태 벡터를 조립하여 반환합니다.
"""

import os
import sys
import json
import time
import numpy as np
import yaml

class MTDStateReader:
    
    def __init__(self, config_path):
        """
        MTD 상태 리더를 초기화하고 3개의 JSON 파일 경로를 로드합니다.
        """
        print(f"[Eyes] MTD 상태 리더 초기화. 규칙서 로드: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            print(f"[Eyes] Error: MTD 규칙서({config_path}) 로드 실패: {e}", file=sys.stderr)
            sys.exit(1)

        # 1. 8D State 벡터의 0-5 인덱스를 정의하는 Real Targets 로드
        self.real_targets = config.get('real_targets', [])
        print(f"[Eyes] {len(self.real_targets)}개의 Real Targets 로드. 8D 상태 벡터 사용.")
        
        if len(self.real_targets) != 6:
            print(f"Warning: real_targets 개수가 6개가 아닙니다 ({len(self.real_targets)}개). 8D 벡터 생성에 문제가 발생할 수 있습니다.", file=sys.stderr)

        # 2. 3개의 JSON 파일 경로 로드
        shared_files_config = config.get('shared_files', {})
        state_dir = os.path.join(os.path.dirname(config_path), '..', 'shared_state')
        
        self.action_state_file = os.path.join(state_dir, shared_files_config.get('action_state', 'mtd_action_state.json'))
        self.cti_assessment_file = os.path.join(state_dir, shared_files_config.get('cti_assessment', 'cti_threat_assessment.json'))
        self.system_health_file = os.path.join(state_dir, shared_files_config.get('system_health', 'dvd_system_health.json'))

        # print(f"[Eyes] Action State 파일 경로: {self.action_state_file}")
        # print(f"[Eyes] CTI Assessment 파일 경로: {self.cti_assessment_file}")
        # print(f"[Eyes] System Health 파일 경로: {self.system_health_file}")

    def _load_json_state(self, file_path, max_age_sec=15):
        """(내부 함수) JSON 파일을 안전하게 읽어옵니다."""
        try:
            if os.path.exists(file_path):
                # 파일이 너무 오래되었는지 확인 (max_age_sec 초 이내)
                file_age = time.time() - os.path.getmtime(file_path)
                if file_age > max_age_sec:
                    # print(f"Warning: JSON 파일이 너무 오래되었습니다 (Age: {file_age:.1f}s): {file_path}")
                    return None
                
                with open(file_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            # print(f"Warning: JSON 파일 읽기 오류 ({file_path}): {e}", file=sys.stderr)
            pass
        return None

    def get_rl_state(self) -> np.ndarray:
        """
        [핵심] 3개의 JSON 파일을 읽어 8D 상태 벡터를 조립합니다.
        - 0-5 (Real Targets): 현재 MTD 타겟이 Real Target 중 하나와 일치하는가?
        - 6 (Decoy Flag):   현재 MTD 타겟이 디코이인가?
        - 7 (Alert Flag):   현재 CTI가 위협을 탐지했는가?
        """
        
        # 8D 벡터 초기화 (전부 0)
        state_vector = np.zeros(8)
        
        # 1. Action State 파일 읽기 (MTD의 '손' 상태)
        action_data = self._load_json_state(self.action_state_file)
        
        current_target_str = None
        is_decoy_active = False

        if action_data:
            current_action = action_data.get('current_action', {})
            current_target_str = current_action.get('active_target')
            is_decoy_active = current_action.get('is_decoy', False)

        # 2. CTI Assessment 파일 읽기 (위협 '분석가' 상태)
        cti_data = self._load_json_state(self.cti_assessment_file)
        
        is_alert_detected = False
        if cti_data:
            is_alert_detected = cti_data.get('alert_detected', False)

        # 3. 8D 벡터 조립
        
        # 인덱스 0-5 (Real Targets)
        if current_target_str:
            for i, target in enumerate(self.real_targets):
                if current_target_str == target:
                    state_vector[i] = 1.0
                    break # 하나만 활성화될 수 있음

        # 인덱스 6 (Decoy Flag)
        if is_decoy_active:
            state_vector[6] = 1.0
            
        # 인덱스 7 (Alert Flag)
        if is_alert_detected:
            state_vector[7] = 1.0

        return state_vector

# --- 모듈 테스트용 ---
if __name__ == "__main__":
    print("[Eyes] MTD State Reader 모듈 테스트...")
    
    # 이 스크립트는 mtd/ 폴더에 있으므로, config 경로는 ../mtd/configs/iptables_mtd.yaml
    config_file_path = os.path.join(script_dir, 'configs', 'iptables_mtd.yaml')
    
    reader = MTDStateReader(config_path=config_file_path)
    
    print("\n--- 10초간 1초마다 상태 벡터 읽기 테스트 ---")
    for i in range(10):
        state = reader.get_rl_state()
        print(f"[T={i+1}s] State Vector: {state.tolist()}")
        time.sleep(1)