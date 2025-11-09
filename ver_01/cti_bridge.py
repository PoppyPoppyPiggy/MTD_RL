# File: MTD_RL/ver_01/cti_bridge.py
# [완성본 v2] MTD_RL(학습 환경)과 MTD_full_testbed(CTI 모델)를 연결하는 브릿지
import numpy as np
import os
import sys
import json
import logging

# --- 로거 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [CTIBridge] %(message)s')
logger = logging.getLogger("CTIBridge")

# --- [보완됨] 테스트베드(MTD_full_testbed)의 ml/ 경로 설정 ---
# 이 스크립트(MTD_RL/ver_01)는 MTD_full_testbed/dvd_lite/dvd_attacks_lpc/rl/ver_01에 위치한다고 가정
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
RL_DIR = os.path.dirname(SCRIPT_DIR) # /rl
ATTACKS_LPC_DIR = os.path.dirname(RL_DIR) # /dvd_attacks_lpc

# [!!!] MTD_full_testbed의 ml/ 폴더 경로 (수동 설정 필요 가능성 있음)
DVD_ATTACKS_LPC_PATH = ATTACKS_LPC_DIR 
ML_PATH = os.path.join(DVD_ATTACKS_LPC_PATH, "ml")
sys.path.append(ML_PATH)

try:
    from cti_agent import CTIAgent
    import data_builder
    CTI_LIBS_IMPORTED = True
    logger.info(f"성공: {ML_PATH}에서 CTI 라이브러리(CTIAgent, data_builder) 임포트")
except ImportError as e:
    logger.error(f"실패: {ML_PATH}에서 CTI 라이브러리 임포트. {e}")
    logger.error("       MTD_RL 서브모듈이 MTD_full_testbed/dvd_lite/dvd_attacks_lpc/rl에 위치하는지 확인하세요.")
    CTIAgent = None
    data_builder = None
    CTI_LIBS_IMPORTED = False

# --- CTI 모델 및 입력 파일 경로 ---
CTI_MODEL_PATH = os.path.join(ML_PATH, "cti_classifier_model.joblib")
EVENTS_FILE_PATH = os.path.join(DVD_ATTACKS_LPC_PATH, "monitors", "system_events.json")


class CTIBridge:
    def __init__(self):
        self.cti_agent = None
        if CTI_LIBS_IMPORTED and os.path.exists(CTI_MODEL_PATH):
            try:
                self.cti_agent = CTIAgent(model_path=CTI_MODEL_PATH)
            except Exception as e:
                logger.error(f"CTIAgent 초기화 실패: {e}")
        else:
            logger.warning("CTI 에이전트가 초기화되지 않았습니다. CTI 우선순위는 랜덤으로 제공됩니다.")
            
    def get_cti_priorities(self) -> np.ndarray:
        """
        [보완됨] MTD_full_testbed의 CTI 모델을 호출하여 공격 우선순위를 반환합니다.
        (HeuristicSeeker가 호출)
        """
        if self.cti_agent is None:
            return np.random.rand(4) # 4는 config.ACTION_DIM (가정)

        try:
            # 1. CTI 입력 데이터(이벤트 로그) 로드
            if os.path.exists(EVENTS_FILE_PATH):
                with open(EVENTS_FILE_PATH, 'r') as f:
                    raw_logs = json.load(f)
            else:
                raw_logs = [] # 이벤트 파일이 없으면 빈 로그

            # 2. data_builder로 피처 벡터 생성
            # (가정) data_builder.build_features_from_logs 함수가 (1, N) 벡터 반환
            feature_vector = data_builder.build_features_from_logs(raw_logs)

            if feature_vector is None:
                feature_vector = np.random.rand(1, 10) # 10은 가정된 피처 수
            
            # 3. CTI 모델로 예측
            probabilities = self.cti_agent.predict(feature_vector) # (K,) 반환
            return probabilities

        except Exception as e:
            logger.error(f"CTIBridge 예측 실패: {e}. 랜덤 우선순위를 반환합니다.")
            return np.random.rand(4) # 4는 config.ACTION_DIM (가정)

if __name__ == '__main__':
    # 테스트용
    print(f"MTD_full_testbed ML 경로: {ML_PATH}")
    print(f"CTI 라이브러리 임포트 성공: {CTI_LIBS_IMPORTED}")
    
    bridge = CTIBridge()
    priorities = bridge.get_cti_priorities()
    
    print(f"\nCTI Bridge 테스트:")
    print(f"  CTI 모델 파일: {CTI_MODEL_PATH} (존재 여부: {os.path.exists(CTI_MODEL_PATH)})")
    print(f"  CTI 이벤트 파일: {EVENTS_FILE_PATH} (존재 여부: {os.path.exists(EVENTS_FILE_PATH)})")
    print(f"\n획득된 공격 우선순위 (Probabilities):")
    print(priorities)