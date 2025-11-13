MTD RL - ver_03

아키텍처 개선 (Architecture Overhaul)

이 버전(ver_03)은 ver_02의 한계를 극복하기 위해 제안된 [MTD RL 아키텍처 개선 로드맵]의 구현체입니다.
주요 변경 사항은 다음과 같습니다:

Brain-Scorer 통합 (로드맵 1단계):

ver_02의 rl_driven_deception_manager.py (Brain)와 mtd_full_testbed의 mtd_scoring.py (Scorer)가 분리되어 발생했던 '신용 할당 문제(CAP)'를 해결했습니다.

environment.py 내의 MTDEnv 클래스에 Scorer 로직을 통합하여, RL 에이전트가 자신의 행동에 대한 MTD 성능 지표(S_D, R_A, C_M)를 직접 관측하고 보상받도록 수정했습니다.

새로운 상태 공간 (S) (로드맵 2.2절):

ver_02의 8D 스냅샷 벡터로 인한 '부분 관측 문제(POMDP)'를 해결하기 위해 상태 공간을 11차원으로 확장했습니다.

S_vector = S_cumulative | S_threat | S_config

S_cumulative (3차원): 누적 MTD 성능 지표 [S_D, R_A, C_M]

S_threat (4차원): '공격 스캔 히트맵'에서 집계된 CTI 위협 상태 [총 스캔 수, 실제 타겟 스캔 수, 디코이 스캔 수, 신규 공격자 플래그]

S_config (4차원): 현재 MTD 설정 상태 [현재 전략 ID, 파라미터 1, 파라미터 2, 마지막 MTD 이후 경과 시간]

새로운 행동 공간 (A) (로드맵 2.1절):

정적 전략 선택에서 벗어나 'MTD 파라미터 동적 최적화'를 위해 '매개변수화된 행동 공간(PAMDP)'을 도입했습니다.

Action = (a_discrete, a_continuous)

a_discrete (4차원): MTD 전략 유형 선택 (0: IP 셔플, 1: 포트 호핑, 2: 디코이 활성화, 3: 조치 없음)

a_continuous (2차원): 선택된 전략에 사용할 연속적 파라미터 [param1, param2] (예: 셔플 강도, 호핑 반경). 0~1 사이 값으로 정규화됨.

새로운 보상 함수 (R) (로드맵 2.3절):

'Brain'과 'Scorer'의 목표를 일치시키기 위해 '델타(Delta) 보상 함수'를 도입했습니다.

R_t = (w_d * delta_S_D) + (w_a * delta_R_A) - (w_m * delta_C_M)

에이전트는 MTD 행동으로 인한 각 성능 지표의 '변화량'에 대해 즉각적인 보상(또는 페널티)을 받아, 행동과 장기적 목표(S_MTD) 간의 인과관계를 학습할 수 있습니다.

PPO 알고리즘 수정 (로드맵 3단계):

ppo.py의 Actor 네트워크가 PAMDP를 처리할 수 있도록 수정되었습니다.

Actor는 이제 2개의 헤드(Head)를 가집니다:

이산적 행동(전략)을 위한 Softmax (Categorical) 출력 헤드

연속적 행동(파라미터)을 위한 Sigmoid (Normal) 출력 헤드

손실 함수는 두 행동의 로그 확률(log-probability)과 엔트로피를 모두 합산하여 계산합니다.

관측 가능성(Observability) 확보 (로드맵 2단계):

environment.py 내에 ThreatMonitor와 MTDExecutor 시뮬레이션 클래스를 구현했습니다.

ThreatMonitor: iptables LOG 및 로그 파서의 역할을 시뮬레이션하여, Seeker의 스캔 활동(S_threat)과 기만 성공(S_D) 이벤트를 생성합니다.

MTDExecutor: iptables_mtd_controller의 역할을 시뮬레이션하여, PAMDP 행동을 받아 MTD 비용(C_M)과 서비스 다운타임(R_A) 이벤트를 생성합니다.

실행 방법 (How to Run)

필요한 라이브러리를 설치합니다. (wandb, torch, gym, numpy 등)

pip install torch gym numpy wandb


train_mtd_only.py 스크립트를 실행하여 ver_03 에이전트 학습을 시작합니다.

python poppypoppypiggy/mtd_rl/MTD_RL-5d2950e60b9d8b9a5bba7e03891fbdb559822ccf/ver_03/train_mtd_only.py


Weights & Biases (wandb.ai) 대시보드에서 mtd_rl_ver_03 프로젝트를 확인하여 학습 진행 상황(평균 보상, S_D, R_A, C_M 등)을 실시간으로 모니터링합니다. (로드맵 4단계)

향후 작업 (Future Work)

environment.py의 ThreatMonitor 및 MTDExecutor를 mtd_full_testbed의 실제 iptables 컨트롤러 및 syslog 파서와 연동하여 시뮬레이션이 아닌 실제 테스트베드에서 작동하도록 확장해야 합니다.