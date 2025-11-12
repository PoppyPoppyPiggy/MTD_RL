import wandb
import pandas as pd

# (선택) 로그인: 터미널에서 `wandb login` 했으면 생략 가능
# wandb.login(key="YOUR_WANDB_API_KEY")

api = wandb.Api()

# ✅ 올바른 경로 형식 (runs/ 없음, 맨 앞 슬래시 없음)
run = api.run("emforhsqhf29-/MTD-RL-Testbed-Trainer/ij0x9qqf")

# 전체 히스토리 -> Pandas DataFrame
hist = run.history()  # 큰 run이면 메모리 많이 먹을 수 있음
print(hist.head())    # 미리보기
print(hist.columns)   # 어떤 지표들이 있는지 확인

# CSV로 저장
hist.to_csv("run_history.csv", index=False)
print("Saved run_history.csv")
