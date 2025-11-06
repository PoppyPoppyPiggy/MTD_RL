#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time Metrics (v21 Refactor)
- v21: Plotting logic removed, only CSV saving
"""

import numpy as np
import torch
import math
import pathlib
from typing import Dict, List

# config.py 파일에서 config 인스턴스 임포트
from config import config

class RealTimeMetrics:
    """
    v19: TTEB(Time-to-Exploit-Block) 등 다단계 지표 반영
    v17: TTF, TTBr, r_cti 추가
    v21: Plotting logic removed
    """
    def __init__(self, num_ips:int, ports:List[int], n_envs:int, step_sec:float=1.0, time_bin_sec:int=5):
        self.num_ips = int(num_ips)
        self.ports = list(sorted(ports))
        self.num_ports = len(self.ports)
        self.num_endpoints = self.num_ips * self.num_ports
        self.n_envs = float(n_envs)
        self.step_sec = float(step_sec)
        self.bin_sec = int(time_bin_sec)

        # 현재 윈도우 누적(raw, env 합계)
        self._cur = {
            # (v19)
            "exploits_raw": 0.0, "exploit_successes_raw": 0.0, "exploit_blocks_raw": 0.0,
            "breaches_raw": 0.0, "breach_successes_raw": 0.0, "breach_blocks_raw": 0.0,
            "decoys_raw":0.0, "scans_raw":0.0, "founds_raw":0.0,
            "did_shuffle_raw":0.0, "cti_events_raw":0.0,
            # time-to-event (sum of exposure_steps)
            "exposure_at_found_sum":0.0,
            "exposure_at_exploit_block_sum":0.0,
            "exposure_at_breach_success_sum":0.0,
        }
        self._svc_hist = np.zeros(self.num_endpoints, dtype=np.int64)  # service visit histogram (for D)
        self._port_seen = set()  # distinct port idx (for R)
        self._bin_elapsed = 0.0  # seconds elapsed inside current bin
        self._last_ip_cd = 0.0 # for CTI plot

        # 결과 시계열
        self.ts = []
        self.series = {k: [] for k in [
            "r_scan","r_find",
            "r_exploit_attempt", "r_exploit_success", "r_exploit_block", "eta_dec", # Perf (Exploit)
            "r_breach_attempt", "r_breach_success", "r_breach_block", # Perf (Breach)
            "D_bits","R","S", # DRS
            "mean_TTF", "mean_TTEB", "mean_TTBr", # Time-to-Event
            "r_cti", "ip_cd_mean" # CTI & Policy
        ]}
        self.header = [
            "time_sec","r_scan","r_find",
            "r_exploit_attempt", "r_exploit_success", "r_exploit_block", "eta_dec",
            "r_breach_attempt", "r_breach_success", "r_breach_block",
            "D_bits","R","S",
            "mean_TTF", "mean_TTEB", "mean_TTBr",
            "r_cti", "ip_cd_mean"
        ]

    # ---------- 내부 도우미 ----------
    def _entropy_bits(self, hist: np.ndarray) -> float:
        s = hist.sum()
        if s <= 0: return 0.0
        p = hist[hist > 0] / s
        return float(-(p * np.log2(p)).sum())

    def _compute_D(self) -> float:
        return self._entropy_bits(self._svc_hist)

    def _compute_R(self) -> float:
        N_critical = 1.0
        N_backup = max(0.0, float(len(self._port_seen)) - N_critical)
        return N_backup / N_critical

    # ---------- 퍼블릭 API ----------
    def ingest_step(self, info: Dict[str, torch.Tensor], service_ids: np.ndarray):
        """현재 벡터 스텝(=step_sec 초) 정보를 원시 카운트로 누적 (v19)"""
        # raw 합계 (모든 env 합)
        self._cur["exploits_raw"]   += float(info["exploits"].sum().item())
        self._cur["exploit_successes_raw"] += float(info["exploit_successes"].sum().item())
        self._cur["exploit_blocks_raw"]    += float(info["exploit_blocks"].sum().item())
        self._cur["breaches_raw"]     += float(info["breaches"].sum().item())
        self._cur["breach_successes_raw"] += float(info["breach_successes"].sum().item())
        self._cur["breach_blocks_raw"]    += float(info["breach_blocks"].sum().item())
        self._cur["decoys_raw"]     += float(info["decoys"].sum().item())
        self._cur["scans_raw"]      += float(info["scans"].sum().item())
        self._cur["founds_raw"]     += float(info["founds"].sum().item())
        self._cur["did_shuffle_raw"] += float(info["did_shuffle"].sum().item())
        self._cur["cti_events_raw"] += float(info["cti_event_triggered"].sum().item()) # CTI

        # Time-to-Event (sum of exposure steps)
        self._cur["exposure_at_found_sum"] += float(info["exposure_at_found"].sum().item())
        self._cur["exposure_at_exploit_block_sum"] += float(info["exposure_at_exploit_block"].sum().item())
        self._cur["exposure_at_breach_success_sum"] += float(info["exposure_at_breach_success"].sum().item())
        
        self._last_ip_cd = float(info["ip_cd"].mean().item()) # policy snapshot

        # D: 서비스 방문 히스토그램 누적
        self._svc_hist += np.bincount(service_ids, minlength=self.num_endpoints)

        # R: 이 스텝에서 관찰된 포트 종류
        port_idx = service_ids % self.num_ports
        for p in np.unique(port_idx):
            self._port_seen.add(int(p))

        self._bin_elapsed += self.step_sec
        while self._bin_elapsed >= self.bin_sec:
            self._flush_bin()

    def _flush_bin(self):
        """TIME_BIN_SEC 마다 누적된 데이터를 평균/비율로 변환하여 저장"""
        dur = float(self.bin_sec)
        denom_env_sec = dur * self.n_envs  # (초 * env 수)

        # per-second(per env) rates
        scans_ps   = self._cur["scans_raw"]   / denom_env_sec
        cti_ps     = self._cur["cti_events_raw"] / denom_env_sec # CTI rate
        exploits_ps = self._cur["exploits_raw"] / denom_env_sec
        breaches_ps = self._cur["breaches_raw"] / denom_env_sec

        # counts
        exploits   = self._cur["exploits_raw"]
        exploit_successes = self._cur["exploit_successes_raw"]
        exploit_blocks = self._cur["exploit_blocks_raw"]
        breaches = self._cur["breaches_raw"]
        breach_successes = self._cur["breach_successes_raw"]
        breach_blocks = self._cur["breach_blocks_raw"]
        decoys     = self._cur["decoys_raw"]
        founds     = self._cur["founds_raw"]

        # ratios
        r_scan   = scans_ps
        r_cti    = cti_ps
        r_find   = (founds / self._cur["scans_raw"]) if self._cur["scans_raw"] > 0 else 0.0

        r_exploit_attempt = exploits_ps
        r_exploit_success = (exploit_successes / exploits) if exploits > 0 else 0.0
        r_exploit_block = (exploit_blocks / exploits) if exploits > 0 else 0.0
        eta_dec  = (decoys / exploits) if exploits > 0 else 0.0
        
        r_breach_attempt = breaches_ps
        r_breach_success = (breach_successes / breaches) if breaches > 0 else 0.0
        r_breach_block = (breach_blocks / breaches) if breaches > 0 else 0.0

        # Time-to-Event (mean steps) (v19)
        mean_TTF = (self._cur["exposure_at_found_sum"] / founds) if founds > 0 else 0.0
        mean_TTEB = (self._cur["exposure_at_exploit_block_sum"] / exploit_blocks) if exploit_blocks > 0 else 0.0 # TTB -> TTEB
        mean_TTBr= (self._cur["exposure_at_breach_success_sum"] / breach_successes) if breach_successes > 0 else 0.0

        # D/R/S
        D_bits = self._compute_D()
        f_shuffle = self._cur["did_shuffle_raw"] / denom_env_sec
        S_val = float(f_shuffle * math.log2(self.num_endpoints))
        R_val = self._compute_R()

        t_end = (self.ts[-1] + dur) if self.ts else dur
        self.ts.append(t_end)
        
        # self.series에 값 추가
        for key, value in [
            ("r_scan", r_scan), ("r_find", r_find),
            ("r_exploit_attempt", r_exploit_attempt), ("r_exploit_success", r_exploit_success),
            ("r_exploit_block", r_exploit_block), ("eta_dec", eta_dec),
            ("r_breach_attempt", r_breach_attempt), ("r_breach_success", r_breach_success),
            ("r_breach_block", r_breach_block),
            ("D_bits", D_bits), ("R", R_val), ("S", S_val),
            ("mean_TTF", mean_TTF), ("mean_TTEB", mean_TTEB), ("mean_TTBr", mean_TTBr),
            ("r_cti", r_cti), ("ip_cd_mean", self._last_ip_cd)
        ]:
            self.series[key].append(value)

        # reset bin
        for k in list(self._cur.keys()):
            self._cur[k] = 0.0
        self._svc_hist[:] = 0
        self._port_seen.clear()
        self._bin_elapsed -= self.bin_sec  # carry remainder

    # ---------- Export ----------
    def save_csv(self, outdir: pathlib.Path):
        """수집된 실시간 지표를 CSV 파일로 저장"""
        outdir.mkdir(parents=True, exist_ok=True)
        rows = [self.header]
        for i, t in enumerate(self.ts):
            row = [t] + [self.series[key][i] for key in self.header if key != "time_sec"]
            rows.append(row)
        
        filepath = outdir / "realtime_metrics.csv"
        try:
            with open(filepath, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(rows)
        except Exception as e:
            print(f"Error saving realtime_metrics.csv: {e}")