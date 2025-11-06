#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Training Loop for MTD vs Seeker ARL Framework (v21)
- [FIX] `plot_training_dashboard` (10개) 및 `plot_realtime_metrics` (7개) 호출 복원
- [FIX] `visit_heatmap` 및 `attack_heatmap` 데이터 수집 로직 복원
"""

import os, sys, csv, math, json, time, argparse, random, pathlib, datetime as dt
from typing import Dict, Any, Tuple, List
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------
# [1] Import (Local Modules)
# -----------------
try:
    from config import Config
    import utils
    import cti_bridge
    from environment import MTDSeekerEnvTorch
    from ppo import PPO, Buffer
    from metrics import RealTimeMetrics
    # [FIX] 2개의 플로팅 함수 모두 import
    from plotting import plot_training_dashboard, plot_realtime_metrics
except ImportError as e:
    print(f"Import Error: {e}")
    print("Ensure all .py files (config.py, environment.py, etc.) are in the same directory.")
    sys.exit(1)

# -----------------
# [2] Config & Globals
# -----------------
config = Config() # Load config
SHOW = ("--show" in sys.argv or "--show" in os.environ.get("ARGV", ""))

# -----------------
# [3] Epoch Metrics & Summary (원본 v21과 동일하게 복원)
# -----------------
def calculate_metrics_from_infos(all_infos, total_steps) -> Dict[str, float]:
    """ (v21) 롤아웃 동안 수집된 모든 info 딕셔너리를 집계 """
    if not all_infos:
        print("Warning: calculate_metrics_from_infos received empty list.")
        return {} # Return empty dict

    # 1. 모든 info에서 텐서들을 스택
    t = {}
    keys_to_stack = [k for k in all_infos[0] if isinstance(all_infos[0][k], torch.Tensor)]
    
    for k in keys_to_stack:
        try:
            valid_tensors = [d[k] for d in all_infos if k in d and d[k] is not None]
            if not valid_tensors:
                continue
            t[k] = torch.stack(valid_tensors)
        except RuntimeError as e:
            # print(f"Warning: Error stacking key '{k}' (shape mismatch?): {e}. Skipping.")
            continue
        except Exception as e:
            # print(f"Warning: Generic error processing key '{k}': {e}. Skipping.")
            continue

    # Helper for safe aggregation (sum or mean)
    def safe_sum(key):
        return t[key].sum().item() if key in t else 0.0
    def safe_mean(key, default_val=0.0):
        # 텐서가 비어있을 수 있음
        tensor = t.get(key)
        if tensor is None or tensor.numel() == 0:
            return default_val
        return tensor.mean().item()

    # 2. 값 계산
    num_exploits = safe_sum("exploits")
    num_exploit_successes = safe_sum("exploit_successes")
    num_exploit_blocks = safe_sum("exploit_blocks")
    num_breaches = safe_sum("breaches")
    num_breach_successes = safe_sum("breach_successes")
    num_breach_blocks = safe_sum("breach_blocks")
    num_decoys = safe_sum("decoys")
    num_scans = safe_sum("scans")
    num_founds = safe_sum("founds")
    
    total_blocks = num_exploit_blocks + num_breach_blocks + num_decoys
    total_cost = safe_sum("cost")

    # 3. 딕셔너리 반환 (원본 v21 기준)
    metrics_epoch = {
        "R_succ": 1.0 - (num_breach_successes / num_breaches) if num_breaches > 0 else 1.0,
        "C_def": total_cost / len(all_infos) if all_infos else 0.0, # Cost per step
        "Cost_per_Block": total_cost / total_blocks if total_blocks > 0 else float('inf'),
        
        "D_bits": 0.0, # D_bits는 히트맵에서 별도 계산
        "S_shuffle": safe_mean("did_shuffle") * math.log2(config.NUM_ENDPOINTS), # S
        "eta_dec": (num_decoys / num_exploits) if num_exploits > 0 else 0.0,
        
        "exposure_mean": safe_mean("exposure"),
        "dwell_mean": safe_mean("dwell"),
        
        "r_exploit_success": (num_exploit_successes / num_exploits) if num_exploits > 0 else 0.0,
        "r_exploit_block": (num_exploit_blocks / num_exploits) if num_exploits > 0 else 0.0,
        "r_breach_success": (num_breach_successes / num_breaches) if num_breaches > 0 else 0.0,
        "r_breach_block": (num_breach_blocks / num_breaches) if num_breaches > 0 else 0.0,
        "r_find": (num_founds / num_scans) if num_scans > 0 else 0.0,
        
        "r_known": safe_mean("known"),
        "r_exploited": safe_mean("exploited"),
        
        "r_exploit_attempt": num_exploits / total_steps if total_steps > 0 else 0.0,
        "r_scan": num_scans / total_steps if total_steps > 0 else 0.0,

        "mean_TTF": safe_sum("exposure_at_found") / num_founds if num_founds > 0 else 0.0,
        "mean_TTEB": safe_sum("exposure_at_exploit_block") / num_exploit_blocks if num_exploit_blocks > 0 else 0.0,
        "mean_TTBr": safe_sum("exposure_at_breach_success") / num_breach_successes if num_breach_successes > 0 else 0.0,
        
        "r_cti": safe_mean("cti_event_triggered"),

        "ip_cd_mean": safe_mean("ip_cd", config.DYN_PARAMS["ip_cd"]["base"]),
        "decoy_ratio_mean": safe_mean("decoy_ratio", config.DYN_PARAMS["decoy_ratio"]["base"]),
        "bl_level_mean": safe_mean("bl_level", config.DYN_PARAMS["bl_level"]["base"]),
        "attack_bias_mean": safe_mean("attack_bias", config.SEEKER_PARAMS["attack_bias"]["base"]),
        "scan_effort_mean": safe_mean("scan_effort", config.SEEKER_PARAMS["scan_effort"]["base"]),
    }
    
    # D_bits (Diversity)
    if "service_id" in t:
        service_ids = t["service_id"].flatten().cpu().numpy()
        if service_ids.size > 0:
            visit_counts = np.bincount(service_ids, minlength=config.NUM_ENDPOINTS)
            visit_sum = visit_counts.sum()
            if visit_sum > 0:
                probs = visit_counts[visit_counts > 0] / visit_sum
                metrics_epoch["D_bits"] = float(-np.sum(probs * np.log2(probs)))

    return metrics_epoch


def save_training_summary(outdir: pathlib.Path, history: Dict[str, list], args: argparse.Namespace):
    """학습 종료 후 최종 요약 JSON 저장 (원본 v21 기준)"""
    
    def tail_avg(key: str, last_n_pct=0.1):
        arr = [v for v in history.get(key, []) if v is not None and not math.isnan(v)]
        if not arr: return 0.0
        n = max(1, int(len(arr) * last_n_pct)) # 마지막 10% 평균
        return float(np.mean(arr[-n:]))

    # Epoch 메트릭 (Epoch_ 접두사 사용)
    metrics = {
        "R_succ_breach_stop": tail_avg("Epoch_R_succ"),
        "C_def": tail_avg("Epoch_C_def"),
        "D_bits": tail_avg("Epoch_D_bits"),
        "eta_dec": tail_avg("Epoch_eta_dec"),
        "r_exploit_success": tail_avg("Epoch_Exploit_Success"),
        "r_breach_success": tail_avg("Epoch_Breach_Success"),
        "r_known": tail_avg("Epoch_Known"),
        "r_exploited": tail_avg("Epoch_Exploited"),
        "mean_TTF": tail_avg("Epoch_TTF"),
        "mean_TTEB": tail_avg("Epoch_TTEB"),
        "mean_TTBr": tail_avg("Epoch_TTBr"),
    }
    # Policy 메트릭 (Policy_ 접두사 사용)
    def_policy = {
        "ip_cd": tail_avg("Policy_ip_cd"),
        "decoy_ratio": tail_avg("Policy_decoy_ratio"),
        "bl_level": tail_avg("Policy_bl_level"),
    }
    seek_policy = {
        "attack_bias": tail_avg("Policy_attack_bias"),
        "scan_effort": tail_avg("Policy_scan_effort"),
    }

    # 간단한 진단 (원본 v21 기준)
    status = "Training Complete"
    diagnosis = "Check plots for detailed analysis."
    recommendation = "Run more updates for convergence."
    
    r_succ_final = metrics.get("R_succ_breach_stop", 0.0)
    c_def_final = metrics.get("C_def", 0.0)
    ip_cd_final = def_policy.get("ip_cd", 0.0)
    bl_level_final = def_policy.get("bl_level", 0.0)

    if r_succ_final > 0.9 and c_def_final < 0.4:
        status = "Good Convergence (Cost-Efficient)"
    elif ip_cd_final < 15 and bl_level_final < 2.0:
        status = "Shuffle-Dominant Strategy"
    elif bl_level_final > 3.0 and ip_cd_final > 40:
        status = "Block-Dominant Strategy"
    elif r_succ_final < 0.8:
        status = "Potentially Suboptimal"

    summary = {
        "run_parameters": vars(args),
        "final_metrics_avg_10pct": metrics,
        "final_policy_defender": def_policy,
        "final_policy_seeker": seek_policy if args.seeker_mode == 'arl' else config.SEEKER_BEHAVIOR_LEVELS[args.seeker_level]["name"],
        "interpretation": {"status": status, "diagnosis": diagnosis, "recommendation": recommendation},
        "config_rewards": {k: v for k, v in vars(config).items() if k.startswith(('REW_', 'PENALTY_'))},
        "config_costs": {k: v for k, v in vars(config).items() if k.startswith('COST_')},
    }

    def convert_nan_to_null(obj):
        if isinstance(obj, dict): return {k: convert_nan_to_null(v) for k, v in obj.items()}
        elif isinstance(obj, list): return [convert_nan_to_null(elem) for elem in obj]
        elif isinstance(obj, float) and math.isnan(obj): return None
        return obj

    try:
        with open(outdir / "models" / "training_summary.json", "w", encoding="utf-8") as f:
            json.dump(convert_nan_to_null(summary), f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error writing training summary JSON: {e}")

# -----------------
# [4] Training Loop
# -----------------
def train(args):
    """ 메인 학습 함수 """
    
    # --- 1. 초기 설정 ---
    utils.set_seed(args.seed)
    outdir_str = f"M{args.mtd_mode}{args.mtd_level}_S{args.seeker_mode}{args.seeker_level}"
    outdir = utils.make_outdir(level=f"{config.LEVEL}_{outdir_str}", args=args)
    
    # --- 2. 환경 및 에이전트 초기화 ---
    env = MTDSeekerEnvTorch(
        n_envs=config.N_ENVS,
        device=config.DEVICE,
        mtd_mode=args.mtd_mode,
        seeker_mode=args.seeker_mode,
        mtd_level=args.mtd_level,
        seeker_level=args.seeker_level
    )
    
    agent_def, buf_def = None, None
    if args.mtd_mode == 'arl':
        agent_def = PPO(env.state_dim_def, config.MTD_META_ACTION_DIM, config.DEVICE, args.lr, config.EPS_CLIP, args.entropy_coef)
        buf_def = Buffer(config.N_ENVS, config.ROLLOUT_STEPS, env.state_dim_def, config.DEVICE)

    agent_seek, buf_seek = None, None
    if args.seeker_mode == 'arl':
        agent_seek = PPO(env.state_dim_seek, config.SEEKER_META_ACTION_DIM, config.DEVICE, args.lr, config.EPS_CLIP, args.entropy_coef)
        buf_seek = Buffer(config.N_ENVS, config.ROLLOUT_STEPS, env.state_dim_seek, config.DEVICE)

    # --- 3. 로깅 및 메트릭 초기화 ---
    rt = RealTimeMetrics(
        num_ips=config.NUM_IPS,
        ports=config.SCENARIO["COMMON_PORTS"],
        n_envs=config.N_ENVS,
        step_sec=config.TIME_STEP_SEC,
        time_bin_sec=config.TIME_BIN_SEC
    )
    
    # [FIX] CSV 헤더 (원본 v21 기준)
    header = [
        "Update", "Reward_Def_Mean", "Reward_Seek_Mean",
        "PolicyLoss_Def", "ValueLoss_Def", "Entropy_Def",
        "PolicyLoss_Seek", "ValueLoss_Seek", "Entropy_Seek",
        # Epoch Metrics
        "Epoch_R_succ", "Epoch_C_def", "Epoch_D_bits", "Epoch_S_shuffle", "Epoch_eta_dec",
        "Epoch_Exploit_Success", "Epoch_Breach_Success", "Epoch_Known", "Epoch_Exploited",
        "Epoch_TTF", "Epoch_TTEB", "Epoch_TTBr", "Epoch_CTI_Rate",
        # Policy
        "Policy_ip_cd", "Policy_decoy_ratio", "Policy_bl_level",
        "Policy_attack_bias", "Policy_scan_effort",
    ]
    try:
        with open(outdir / "training_log.csv", "w", newline="", encoding='utf-8') as f:
            csv.writer(f).writerow(header)
    except IOError as e:
        print(f"Error: Could not write to training log file {outdir/'training_log.csv'}: {e}")
        return
        
    hist = {k: [] for k in header}
    
    # [FIX] 히트맵 데이터 초기화 (원본 v21 기준)
    visit_heatmap = np.zeros((config.NUM_IPS, len(config.SCENARIO["COMMON_PORTS"])))
    attack_heatmap = np.zeros_like(visit_heatmap)

    if args.internal_cti:
        cti_window_steps = max(1, int(config.CTI_INTERNAL_WINDOW_STEPS / config.TIME_STEP_SEC))
        recent_breach_successes_history = deque(maxlen=cti_window_steps)
        print(f"[CTI Internal] Enabled: Trigger if {config.CTI_INTERNAL_SUCCESS_COUNT} breaches in {cti_window_steps} steps.")
    else:
        recent_breach_successes_history = None

    print(f"####### ARL 학습 시작 #######")
    print(f"Level: {config.LEVEL}")
    print(f"Scenario: MTD({args.mtd_mode}, L{args.mtd_level}) vs Seeker({args.seeker_mode}, L{args.seeker_level})")
    print(f"Device: {config.DEVICE}, Updates: {args.updates}, Seed: {args.seed}")
    print(f"결과 디렉토리: {outdir}")
    
    start_time = time.time()
    obs_def, obs_seek = env.reset()

    # --- 4. [TRAINING LOOP] ---
    for update in range(1, args.updates + 1):
        infos = []
        
        # --- 4a. [ROLLOUT] ---
        for _ in range(config.ROLLOUT_STEPS):
            cti_triggered_external = torch.zeros(config.N_ENVS, dtype=torch.bool, device=config.DEVICE)
            cti_triggered_internal = torch.zeros(config.N_ENVS, dtype=torch.bool, device=config.DEVICE)
            
            if args.cti:
                evt = cti_bridge.read_cti_event(config.CTI_EVENT_PATH)
                if evt:
                    cp = cti_bridge.event_to_cti_params(evt)
                    off = cti_bridge.policy_offset_from_attack(evt)
                    if cp: env.apply_cti(cp)
                    if off: cti_triggered_external = env.apply_policy_offset(off)
            
            with torch.no_grad():
                a_def, lp_def, v_def = (agent_def.policy.act(obs_def)) if agent_def else (None, None, None)
                a_seek, lp_seek, v_seek = (agent_seek.policy.act(obs_seek)) if agent_seek else (None, None, None)
            
            next_obs_def, next_obs_seek, r_def, r_seek, done, info = env.step(a_def, a_seek)

            if recent_breach_successes_history is not None:
                current_breach_successes = info["breach_successes"].sum().item()
                recent_breach_successes_history.append(current_breach_successes)
                if len(recent_breach_successes_history) == recent_breach_successes_history.maxlen:
                    if sum(recent_breach_successes_history) >= config.CTI_INTERNAL_SUCCESS_COUNT:
                        cti_triggered_internal = env.apply_policy_offset(config.CTI_INTERNAL_RESPONSE)
                        recent_breach_successes_history.clear()
            
            info["cti_event_triggered"] = (cti_triggered_external | cti_triggered_internal).float()
            infos.append(info)
            
            if agent_def: buf_def.add(obs_def, a_def, r_def, done, lp_def, v_def)
            if agent_seek: buf_seek.add(obs_seek, a_seek, r_seek, done, lp_seek, v_seek)
                
            obs_def, obs_seek = next_obs_def, next_obs_seek

            svc_np = info["service_id"].detach().view(-1).cpu().numpy()
            rt.ingest_step(info, svc_np)
        
        # --- 4b. [PPO UPDATE] ---
        losses_def_vals = [0.0, 0.0, 0.0]
        losses_seek_vals = [0.0, 0.0, 0.0]
        
        if agent_def:
            with torch.no_grad(): _, last_v_def = agent_def.policy(next_obs_def)
            ret_def, adv_def = buf_def.compute_returns_and_advantages(last_v_def, config.GAMMA, config.GAE_LAMBDA)
            losses_def_vals = agent_def.update(buf_def, ret_def, adv_def, config.K_EPOCHS, config.MAX_GRAD_NORM)

        if agent_seek:
            with torch.no_grad(): _, last_v_seek = agent_seek.policy(next_obs_seek)
            ret_seek, adv_seek = buf_seek.compute_returns_and_advantages(last_v_seek, config.GAMMA, config.GAE_LAMBDA)
            losses_seek_vals = agent_seek.update(buf_seek, ret_seek, adv_seek, config.K_EPOCHS, config.MAX_GRAD_NORM)

        # --- 4c. [LOGGING] ---
        # 롤아웃 데이터 집계 (Epoch Metrics 계산)
        metrics_epoch = calculate_metrics_from_infos(infos, config.N_ENVS * config.ROLLOUT_STEPS)
        
        # [FIX] 히트맵 데이터 누적 (원본 v21 기준)
        if infos:
            t = {k: torch.stack([d[k] for d in infos if k in d]) for k in infos[0] if isinstance(infos[0][k], torch.Tensor)}
            if "service_id" in t:
                service_ids = t["service_id"].flatten().cpu().numpy()
                visit_counts = np.bincount(service_ids, minlength=config.NUM_ENDPOINTS)
                if visit_counts.shape[0] == visit_heatmap.size:
                    visit_heatmap += visit_counts.reshape(visit_heatmap.shape)

            if "exploits" in t and "service_id" in t:
                exploit_mask = t["exploits"].flatten().cpu().numpy().astype(bool)
                if exploit_mask.shape == service_ids.shape:
                    exploit_loc = service_ids[exploit_mask]
                    if len(exploit_loc) > 0:
                        exploit_counts = np.bincount(exploit_loc, minlength=config.NUM_ENDPOINTS)
                        if exploit_counts.shape[0] == attack_heatmap.size:
                            attack_heatmap += exploit_counts.reshape(attack_heatmap.shape)

        # CSV 저장을 위한 row 구성
        row = {
            "Update": update,
            "Reward_Def_Mean": buf_def.rewards.mean().item() if buf_def else 0.0,
            "Reward_Seek_Mean": buf_seek.rewards.mean().item() if buf_seek else 0.0,
            
            "PolicyLoss_Def": losses_def_vals[0],
            "ValueLoss_Def": losses_def_vals[1],
            "Entropy_Def": losses_def_vals[2],
            "PolicyLoss_Seek": losses_seek_vals[0],
            "ValueLoss_Seek": losses_seek_vals[1],
            "Entropy_Seek": losses_seek_vals[2],
            
            # Epoch Metrics
            "Epoch_R_succ": metrics_epoch["R_succ"],
            "Epoch_C_def": metrics_epoch["C_def"],
            "Epoch_D_bits": metrics_epoch["D_bits"],
            "Epoch_S_shuffle": metrics_epoch["S_shuffle"],
            "Epoch_eta_dec": metrics_epoch["eta_dec"],
            "Epoch_Exploit_Success": metrics_epoch["r_exploit_success"],
            "Epoch_Breach_Success": metrics_epoch["r_breach_success"],
            "Epoch_Known": metrics_epoch["r_known"],
            "Epoch_Exploited": metrics_epoch["r_exploited"],
            "Epoch_TTF": metrics_epoch["mean_TTF"],
            "Epoch_TTEB": metrics_epoch["mean_TTEB"],
            "Epoch_TTBr": metrics_epoch["mean_TTBr"],
            "Epoch_CTI_Rate": metrics_epoch["r_cti"],
            
            # Policy
            "Policy_ip_cd": metrics_epoch["ip_cd_mean"],
            "Policy_decoy_ratio": metrics_epoch["decoy_ratio_mean"],
            "Policy_bl_level": metrics_epoch["bl_level_mean"],
            "Policy_attack_bias": metrics_epoch["attack_bias_mean"],
            "Policy_scan_effort": metrics_epoch["scan_effort_mean"],
        }
        
        row_safe = {k: (v if (v is not None and not (isinstance(v, float) and math.isnan(v))) else "") for k, v in row.items()}
        try:
             with open(outdir / "training_log.csv", "a", newline="", encoding='utf-8') as f:
                 csv.writer(f).writerow([row_safe.get(k,"") for k in header])
        except IOError as e:
            print(f"Warning: Could not write to training log file at update {update}: {e}")
            
        for k in header:
            hist[k].append(row.get(k, float('nan')))

        # 주기적 출력
        if update % 20 == 0 or update == args.updates:
            elapsed = time.time() - start_time
            print(f"Upd {update:4d}/{args.updates} | "
                  f"R_Def: {row['Reward_Def_Mean']:.3f} | "
                  f"R_Seek: {row['Reward_Seek_Mean']:.3f} | "
                  f"Epoch BreachSucc: {metrics_epoch['r_breach_success']:.2%} | "
                  f"Epoch TTEB: {metrics_epoch['mean_TTEB']:.1f}s | "
                  f"Elapsed: {elapsed:.1f}s")

    # --- 5. [TRAINING LOOP END] ---
    print("\nTraining complete. Saving final models and plots...")
    
    if agent_def: torch.save(agent_def.policy.state_dict(), outdir / "models" / "defender_policy.pth")
    if agent_seek: torch.save(agent_seek.policy.state_dict(), outdir / "models" / "seeker_policy.pth")

    save_training_summary(outdir, hist, args)

    # --- 6. [FINAL PLOTTING] ---
    # CSV (RT) 저장
    rt.save_csv(outdir / "figs-rt") # realtime_metrics.csv 저장
    
    # [FIX] 2개의 플로팅 함수 모두 호출
    print("Generating final training dashboard (figs/)...")
    plot_training_dashboard(
        hist, 
        visit_heatmap, 
        attack_heatmap, 
        outdir,
        port_names=config.SCENARIO["COMMON_PORTS"]
    )
    
    print("Generating final real-time metrics (figs-rt/)...")
    plot_realtime_metrics(outdir) # <-- [!!] 수정된 부분 [!!]
    
    print(f"\n학습 완료. 결과: {outdir}")

# -----------------
# [5] Validation Functions
# (검증 모드는 그래프를 생성하지 않고 텍스트만 출력합니다)
# -----------------
def validate_static(args):
    """정적 MTD 레벨(L0~L5) vs 지정된 Seeker"""
    utils.set_seed(args.seed)
    seeker_level_val = args.seeker_level_validate
    seeker_name = config.SEEKER_BEHAVIOR_LEVELS[seeker_level_val]['name']
    outdir_str = f"VALIDATE_STATIC_vs_S{seeker_level_val}"
    outdir = utils.make_outdir(level=f"{config.LEVEL}_{outdir_str}", args=args)
    
    results = {}

    # 1. ARL Policy (Trained) vs Seeker
    try:
        env_arl = MTDSeekerEnvTorch(n_envs=config.N_ENVS, device=config.DEVICE, mtd_mode='arl', seeker_mode='static_behavior', seeker_level=seeker_level_val)
        def_pol = PPO(env_arl.state_dim_def, config.MTD_META_ACTION_DIM, config.DEVICE, 0, 0, 0).policy
        def_pol.load_state_dict(torch.load(args.load_policy, map_location=config.DEVICE))
        def_pol.eval()
        
        infos, obs_def, _ = [], *env_arl.reset()
        for _ in range(config.VALIDATION_STEPS):
            with torch.no_grad(): a_def,_,_ = def_pol.act(obs_def)
            obs_def, _, _, _, _, info = env_arl.step(a_def, None)
            info["cti_event_triggered"] = torch.zeros(config.N_ENVS, dtype=torch.float, device=config.DEVICE)
            infos.append(info)
        results[f'Learned ARL (vs {seeker_name})'] = calculate_metrics_from_infos(infos, config.N_ENVS*config.VALIDATION_STEPS)
    except FileNotFoundError:
        print(f"Error: Policy file not found at {args.load_policy}")
        return
    except Exception as e:
        print(f"Error loading ARL policy: {e}")
        return

    # 2. Static MTD Levels (L0-L5) vs Seeker
    for lvl, p in config.STATIC_MTD_LEVELS.items():
        env_s = MTDSeekerEnvTorch(n_envs=config.N_ENVS, device=config.DEVICE, mtd_mode='static', mtd_level=lvl, seeker_mode='static_behavior', seeker_level=seeker_level_val)
        infos_static, _, _ = [], *env_s.reset()
        for _ in range(config.VALIDATION_STEPS):
            _, _, _, _, _, info_static = env_s.step(None, None)
            info_static["cti_event_triggered"] = torch.zeros(config.N_ENVS, dtype=torch.float, device=config.DEVICE)
            infos_static.append(info_static)
        results[p['name']] = calculate_metrics_from_infos(infos_static, config.N_ENVS*config.VALIDATION_STEPS)


    print(f"\n===== [정적 정책 비교 (vs Seeker {seeker_name})] =====")
    print(f"{'Policy':28s} | {'BrchSucc':>8s} | {'ExpSucc':>8s} | {'FindRate':>8s} | {'TTEB(s)':>7s} | {'TTF(s)':>7s} | {'D_bits':>7s} | {'S_shuf':>7s}")
    print("-" * 90)
    
    for name, m in results.items():
        if not m: continue
        print(f"{name:28s} | "
              f"{m.get('r_breach_success', 0.0):>8.2%} | "
              f"{m.get('r_exploit_success', 0.0):>8.2%} | "
              f"{m.get('r_find', 0.0):>8.2%} | "
              f"{m.get('mean_TTEB', 0.0):>7.1f} | "
              f"{m.get('mean_TTF', 0.0):>7.1f} | "
              f"{m.get('D_bits', 0.0):>7.2f} | "
              f"{m.get('S_shuffle', 0.0):>7.2f}")

def validate_policy(args):
    """ ARL Policy (v1) vs ARL Policy (v2) 쇼다운 """
    utils.set_seed(args.seed)
    print("\n####### 정책 쇼다운 #######")
    env = MTDSeekerEnvTorch(n_envs=config.N_ENVS, device=config.DEVICE, mtd_mode='arl', seeker_mode='arl')
    
    def load_policy_safe(path, state_dim, action_dim, device):
        try:
            policy = PPO(state_dim, action_dim, device, 0, 0, 0).policy
            policy.load_state_dict(torch.load(path, map_location=device))
            policy.eval()
            return policy
        except FileNotFoundError:
            print(f"Error: Policy file not found at {path}")
            return None
        except Exception as e:
            print(f"Error loading policy state dict from {path}: {e}")
            return None

    def_v1 = load_policy_safe(args.load_def_v1, env.state_dim_def, config.MTD_META_ACTION_DIM, config.DEVICE)
    seek_v1 = load_policy_safe(args.load_seek_v1, env.state_dim_seek, config.SEEKER_META_ACTION_DIM, config.DEVICE)
    def_v2 = load_policy_safe(args.load_def_v2, env.state_dim_def, config.MTD_META_ACTION_DIM, config.DEVICE)
    seek_v2 = load_policy_safe(args.load_seek_v2, env.state_dim_seek, config.SEEKER_META_ACTION_DIM, config.DEVICE)

    scenarios = {
        "Def(v1) vs Seek(v1)": (def_v1, seek_v1),
        "Def(v2) vs Seek(v2)": (def_v2, seek_v2),
        "Def(v2) vs Seek(v1)": (def_v2, seek_v1),
        "Def(v1) vs Seek(v2)": (def_v1, seek_v2),
    }
    results = {}
    
    for name, (dpol, spol) in scenarios.items():
        if not dpol or not spol:
            print(f"Skipping scenario '{name}' due to missing policy.")
            continue
            
        infos, obs_def, obs_seek = [], *env.reset()
        for _ in range(config.VALIDATION_STEPS):
            with torch.no_grad():
                a_def,_,_ = dpol.act(obs_def)
                a_seek,_,_ = spol.act(obs_seek)
            obs_def, obs_seek, _, _, _, info = env.step(a_def, a_seek)
            info["cti_event_triggered"] = torch.zeros(config.N_ENVS, dtype=torch.float, device=config.DEVICE)
            infos.append(info)
        results[name] = calculate_metrics_from_infos(infos, config.N_ENVS*config.VALIDATION_STEPS)

    print("\n===== [쇼다운 결과] =====")
    print(f"{'Scenario':24s} | {'BrchSucc':>8s} | {'ExpSucc':>8s} | {'FindRate':>8s} | {'TTEB(s)':>7s} | {'TTF(s)':>7s}")
    print("-" * 75)
    for name, m in results.items():
        if not m: continue
        print(f"{name:24s} | "
              f"{m.get('r_breach_success', 0.0):>8.2%} | "
              f"{m.get('r_exploit_success', 0.0):>8.2%} | "
              f"{m.get('r_find', 0.0):>8.2%} | "
              f"{m.get('mean_TTEB', 0.0):>7.1f} | "
              f"{m.get('mean_TTF', 0.0):>7.1f}")

# -----------------
# [6] Main Execution
# -----------------
if __name__ == '__main__':
    p = argparse.ArgumentParser(description="MTD vs Seeker ARL Framework (v21)")
    
    # --- 기본 실행 모드 ---
    p.add_argument("--mode", type=str, default="train", choices=['train', 'validate-static', 'validate-policy'])
    p.add_argument("--level", type=str, default=config.LEVEL, help="Experiment level name (for results folder).")
    p.add_argument("--seed", type=int, default=config.SEED)

    # --- 학습 파라미터 ---
    g_train = p.add_argument_group('Training')
    g_train.add_argument("--updates", type=int, default=config.META_UPDATES, help="Total number of training updates.")
    g_train.add_argument("--lr", type=float, default=config.LR, help="Learning rate.")
    g_train.add_argument("--entropy-coef", type=float, default=config.ENTROPY_COEF, help="Entropy bonus coefficient.")
    
    # --- 시나리오 파라미터 ---
    g_scen = p.add_argument_group('Scenario')
    g_scen.add_argument("--mtd-mode", type=str, default="arl", choices=['arl', 'static'])
    g_scen.add_argument("--mtd-level", type=int, default=0, choices=list(config.STATIC_MTD_LEVELS.keys()))
    g_scen.add_argument("--seeker-mode", type=str, default="arl", choices=['arl', 'static_behavior'])
    g_scen.add_argument("--seeker-level", type=int, default=3, choices=list(config.SEEKER_BEHAVIOR_LEVELS.keys()))
    
    # --- CTI 파라미터 ---
    g_cti = p.add_argument_group('CTI')
    g_cti.add_argument("--cti", action="store_true", default=False, help="Enable external CTI file reading.")
    g_cti.add_argument("--internal-cti", action="store_true", default=config.CTI_INTERNAL_TRIGGER, help="Enable internal CTI trigger.")

    # --- 검증 파라미터 ---
    g_val = p.add_argument_group('Validation')
    g_val.add_argument("--load-policy", type=str, default="results/models/defender_policy.pth", help="Path to defender_policy.pth for validation.")
    g_val.add_argument("--seeker-level-validate", type=int, default=0, choices=list(config.SEEKER_BEHAVIOR_LEVELS.keys()), help="Seeker level to validate against (for validate-static).")
    
    g_show = p.add_argument_group('Showdown')
    g_show.add_argument("--load-def-v1", type=str, help="Path to defender_policy.pth (v1).")
    g_show.add_argument("--load-seek-v1", type=str, help="Path to seeker_policy.pth (v1).")
    g_show.add_argument("--load-def-v2", type=str, help="Path to defender_policy.pth (v2).")
    g_show.add_argument("--load-seek-v2", type=str, help="Path to seeker_policy.pth (v2).")

    args = p.parse_args()

    # --- 모드에 따라 config 객체 값 수정 ---
    config.LEVEL = args.level
    config.SEED = args.seed
    config.META_UPDATES = args.updates
    config.LR = args.lr
    config.ENTROPY_COEF = args.entropy_coef
    config.CTI_INTERNAL_TRIGGER = args.internal_cti
    
    # --- 메인 함수 실행 ---
    if args.mode == 'train':
        # MTD/Seeker 모드 결정 (argparse가 static/arl을 처리)
        # train() 함수가 args를 받아 내부에서 처리
        train(args)
    elif args.mode == 'validate-static':
        if not os.path.exists(args.load_policy):
            p.error(f"--load-policy file not found: {args.load_policy}")
        validate_static(args)
    elif args.mode == 'validate-policy':
        if not all([args.load_def_v1, args.load_seek_v1, args.load_def_v2, args.load_seek_v2]):
            p.error("--load-def-v1, --load-seek-v1, --load-def-v2, --load-seek-v2 all 4 are required.")
        validate_policy(args)