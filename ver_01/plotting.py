#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting Utilities for MTD vs Seeker ARL Framework (v21)
- [UPGRADE] 원본(v21)의 모든 플로팅 기능 (17개 그래프) 복원
- `plot_training_dashboard`: figs/ (10개) 생성 (학습 곡선, 파라미터, 히트맵)
- `plot_realtime_metrics`: figs-rt/ (7개) 생성 (실시간 집계)
"""

from typing import Dict, List, Tuple
import numpy as np
import pathlib
import csv
import math
import matplotlib
matplotlib.use("Agg") # Non-GUI backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------
# 1. Real-time Metrics Plotting (figs-rt/)
# ---------------------------------

def _plot_helper_rt(x, ys, labels, title, ylabel, path: pathlib.Path, y2s=None, y2labels=None, y2label=None):
    """Internal plotting helper function for RealTimeMetrics"""
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    if x is None or len(x) == 0: # 데이터가 없으면 빈 그래프
        ax.set_title(title); ax.set_xlabel("Time (sec)"); ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--")
        plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()
        return

    for y, l in zip(ys, labels):
        ax.plot(x, y, label=l)
    ax.set_title(title); ax.set_xlabel("Time (sec)"); ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--"); ax.legend(loc='upper left')

    if y2s:
        ax2 = ax.twinx()
        for y, l in zip(y2s, y2labels):
            ax2.plot(x, y, label=l, linestyle='--', alpha=0.8)
        ax2.set_ylabel(y2label or ylabel); ax2.legend(loc='upper right')

    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()

def _load_csv(filepath: pathlib.Path) -> Tuple[Dict[str, np.ndarray], List[str]] | Tuple[None, None]:
    """CSV 파일을 읽어 딕셔너리로 반환"""
    if not filepath.exists():
        print(f"Warning: Plotting failed, file not found: {filepath}")
        return None, None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            data = [row for row in reader if row] # 빈 줄 무시
        
        if not data:
            print(f"Warning: No data found in CSV: {filepath}")
            return None, None

        data = np.array(data, dtype=float).T
        return {h: data[i] for i, h in enumerate(header)}, header
    except Exception as e:
        print(f"Error loading CSV {filepath}: {e}")
        return None, None

def plot_realtime_metrics(outdir: pathlib.Path):
    """
    realtime_metrics.csv를 읽어 7개의 RT 그래프 생성 (figs-rt/)
    (원본 RealTimeMetrics.save_pngs()의 로직)
    """
    print(f"Generating Real-time Metrics plots in {outdir / 'figs-rt'}...")
    outdir_png = outdir / "figs-rt"
    filepath = outdir_png / "realtime_metrics.csv"
    data, header = _load_csv(filepath)
    if data is None:
        print("Skipping Real-time metrics plotting (data not found).")
        return

    x = data.get("time_sec", [])
    if len(x) == 0:
        print("Skipping Real-time metrics plotting (no timestamps found).")
        return

    # DRS
    _plot_helper_rt(x, [data["D_bits"]], ["D"], "Diversity (D)", "bits", outdir_png / "rt_01_Diversity.png")
    _plot_helper_rt(x, [data["R"]], ["R"], "Redundancy (R)", "R", outdir_png / "rt_02_Redundancy.png")
    _plot_helper_rt(x, [data["S"]], ["S"], "Shuffle (S)", "S", outdir_png / "rt_03_Shuffle.png")

    # Performance (v19)
    _plot_helper_rt(
        x,
        [data["r_exploit_attempt"], data["r_exploit_success"], data["r_exploit_block"], data["eta_dec"]],
        ["ExploitAttempt/s", "ExploitSuccessRate", "ExploitBlockRate", "Decoy(η_dec)"],
        "MTD vs Seeker — L1 Performance (Exploit)", "rate", outdir_png / "rt_04_MTD_vs_Seeker_L1.png"
    )
    _plot_helper_rt(
        x,
        [data["r_breach_attempt"], data["r_breach_success"], data["r_breach_block"]],
        ["BreachAttempt/s", "BreachSuccessRate", "BreachBlockRate"],
        "MTD vs Seeker — L2 Performance (Breach)", "rate", outdir_png / "rt_04_MTD_vs_Seeker_L2.png"
    )
    _plot_helper_rt(
        x,
        [data["r_scan"], data["r_find"]],
        ["Scan/s", "Find ratio"],
        "Seeker Scanning — Performance", "rate", outdir_png / "rt_05_Seeker_Scan_Find.png"
    )

    # Intuitive Time-to-Event (v19)
    _plot_helper_rt(
        x,
        [data["mean_TTF"], data["mean_TTEB"], data["mean_TTBr"]],
        ["Mean Time-to-Find (TTF)", "Mean Time-to-Exploit-Block (TTEB)", "Mean Time-to-Breach (TTBr)"],
        "MTD Intuitive Metrics — Time-to-Event", "Mean Steps (sec)", outdir_png / "rt_06_MTD_Time_to_Event.png"
    )

    # CTI & Policy Response (ip_cd_mean이 없을 경우 대비)
    if "ip_cd_mean" in data:
         _plot_helper_rt(
            x,
            [data["r_cti"]], ["CTI Event Rate"],
            "CTI Events & Policy Response", "Rate", outdir_png / "rt_07_CTI_and_Policy.png",
            y2s=[data["ip_cd_mean"]], y2labels=["ip_cd (Shuffle Freq)"], y2label="ip_cd"
        )

# ---------------------------------
# 2. Training Dashboard Plotting (figs/)
# ---------------------------------

def _simple_plot(x, ys, labels, title, ylabel, path, y2s=None, y2labels=None, y2label=None):
    """Internal plotting helper for Training Dashboard"""
    plt.figure(figsize=(10,6))
    ax = plt.gca()
    has_data = False
    if x is not None and len(x) > 0:
        for y,l in zip(ys,labels):
            # NaN 값 필터링
            valid_indices = [i for i, val in enumerate(y) if val is not None and not math.isnan(val)]
            if valid_indices:
                ax.plot([x[i] for i in valid_indices], [y[i] for i in valid_indices], label=l)
                has_data = True
        
        ax.set_title(title); ax.set_xlabel('Update'); ax.set_ylabel(ylabel); ax.grid(True, linestyle='--')
        if has_data: ax.legend(loc='upper left')

        if y2s and has_data:
            ax2 = ax.twinx()
            has_y2_data = False
            for y,l in zip(y2s,y2labels):
                valid_indices = [i for i, val in enumerate(y) if val is not None and not math.isnan(val)]
                if valid_indices:
                    ax2.plot([x[i] for i in valid_indices], [y[i] for i in valid_indices], label=l, linestyle='--', alpha=0.8)
                    has_y2_data = True
            if has_y2_data:
                ax2.set_ylabel(y2label or ylabel); ax2.legend(loc='upper right')
    else:
         ax.set_title(title); ax.set_xlabel('Update'); ax.set_ylabel(ylabel); ax.grid(True, linestyle='--')
    
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()

def plot_training_dashboard(history: Dict[str, list], 
                            visit_heatmap: np.ndarray, 
                            attack_heatmap: np.ndarray, 
                            outdir: pathlib.Path, 
                            port_names: List[str]):
    """
    학습 로그(history)를 기반으로 10개의 학습 곡선 그래프 생성 (figs/)
    (원본 plot_and_save()의 로직)
    """
    print(f"Generating Training Dashboard plots in {outdir / 'figs'}...")
    updates = history.get("Update", [])
    if not updates: 
        print("Skipping Training Dashboard plotting (no updates found).")
        return

    figs_dir = outdir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Rewards
    _simple_plot(updates, [history.get('Reward_Def_Mean', []), history.get('Reward_Seek_Mean', [])], ["Def(Mean)", "Seek(Mean)"],
                   "1. Agent Rewards", "Mean Reward", figs_dir / "01_rewards.png")

    # Plot 2: PPO Loss (Def)
    _simple_plot(updates, 
                 [history.get('PolicyLoss_Def', []), history.get('ValueLoss_Def', [])], 
                 ["PolicyLoss", "ValueLoss"],
                 "2. Defender PPO Loss", "Loss", figs_dir / "02_ppo_loss_def.png")

    # Plot 3: Tradeoff (R_succ vs C_def)
    # (Note: R_succ/C_def는 원본의 calculate_metrics_from_infos에서 계산되어야 함. train.py가 이를 history에 추가해야 함)
    r_succ_data = history.get('Epoch_R_succ', []) # train.py가 이 이름으로 저장한다고 가정
    c_def_data = history.get('Epoch_C_def', [])   # train.py가 이 이름으로 저장한다고 가정
    
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    if updates and r_succ_data and c_def_data:
        ax2 = ax.twinx()
        valid_indices_r = [i for i, v in enumerate(r_succ_data) if v is not None and not math.isnan(v)]
        valid_indices_c = [i for i, v in enumerate(c_def_data) if v is not None and not math.isnan(v)]
        
        if valid_indices_r:
            ax.plot([updates[i] for i in valid_indices_r], [r_succ_data[i] for i in valid_indices_r], label='R_succ (BreachStop)', linewidth=2, color='blue')
            ax.legend(loc='upper left')
        if valid_indices_c:
            ax2.plot([updates[i] for i in valid_indices_c], [c_def_data[i] for i in valid_indices_c], label='C_def (Cost)', linewidth=2, linestyle='--', color='red')
            ax2.legend(loc='upper right')

        ax.set_title('3. Performance vs Cost'); ax.set_xlabel('Update'); ax.set_ylabel('R_succ'); ax2.set_ylabel('C_def')
        ax.grid(True, linestyle='--')
    else:
        ax.set_title('3. Performance vs Cost (No Data)'); ax.set_xlabel('Update')
    plt.tight_layout(); plt.savefig(figs_dir / "03_tradeoff.png", dpi=200); plt.close()


    # Plot 4: Seeker Success (Multi-Stage)
    _simple_plot(updates, [history.get('Epoch_Exploit_Success', []), history.get('Epoch_Breach_Success', [])],
                   ["Exploit Success Rate", "Breach Success Rate"],
                   "4. Seeker Success (Multi-Stage)", "Rate", figs_dir / "04_seeker_multistage_success.png")

    # Plot 5: LPC
    _simple_plot(updates, [history.get('Epoch_Known', []), history.get('Epoch_Exploited', [])],
                   ["r_known (Found)", "r_exploited (Access)"],
                   "5. LPC: Knowledge vs Access", "Rate", figs_dir / "05_lpc_knowledge_access.png")

    # Plot 6: Seeker Params
    _simple_plot(updates, [history.get('Policy_scan_effort', []), history.get('Policy_attack_bias', [])],
                   ["scan_effort", "attack_bias (0=Stealth, 1=Loud)"],
                   "6. Seeker Policy Params", "Value", figs_dir / "06_seeker_params.png")

    # Plot 7: MTD Params & CTI
    _simple_plot(updates,
                   [history.get('Policy_ip_cd', []), history.get('Policy_decoy_ratio', []), history.get('Policy_bl_level', [])],
                   ["ip_cd (Shuffle)", "decoy_ratio", "bl_level (Block)"],
                   "7. MTD Policy Params & CTI", "Value", figs_dir / "07_mtd_params_cti.png",
                   y2s=[history.get('Epoch_CTI_Rate', [])], y2labels=["CTI Event Rate"], y2label="CTI Rate")

    # Plot 8: Time-to-Event (Epoch)
    _simple_plot(updates,
                   [history.get('Epoch_TTF', []), history.get('Epoch_TTEB', []), history.get('Epoch_TTBr', [])],
                   ["Mean TTF", "Mean TTEB", "Mean TTBr"],
                   "8. Epoch: Time-to-Event", "Mean Steps (sec)", figs_dir / "08_time_to_event.png")

    # Plot 9 & 10: Heatmaps
    for name, data in [("09_visit_heatmap.png", visit_heatmap), ("10_exploit_heatmap.png", attack_heatmap)]:
        plt.figure(figsize=(12, 5))
        if data is not None and data.shape[0] > 0 and data.shape[1] > 0:
            im = plt.imshow(data, aspect='auto', interpolation='nearest', cmap='viridis')
            plt.title(name[:-4]); plt.xlabel('Port Index'); plt.ylabel('IP Index')
            plt.xticks(np.arange(data.shape[1]), [str(p) for p in port_names][:data.shape[1]], rotation=45, ha="right")
            plt.yticks(np.arange(data.shape[0]))
            if data.sum() > 0: plt.colorbar(im, label='count')
        else:
            plt.title(f"{name[:-4]} (No data)"); plt.xlabel('Port Index'); plt.ylabel('IP Index')
        plt.tight_layout(); plt.savefig(figs_dir / name, dpi=200); plt.close()