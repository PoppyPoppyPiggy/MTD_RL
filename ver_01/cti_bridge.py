#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTI Event Bridge for MTD vs Seeker ARL Framework
"""

import json
import pathlib
import numpy as np

# CTI 공격 ID별 MTD 정책 대응 맵
ATTACK_TO_POLICY = {
    # comm-link flooding / slowloris 등 -> 셔플 주기 단축, 디코이↑
    2: {"ip_cd_mult": 0.7, "decoy_add": 0.10, "bl_add": 1.0},
    29: {"ip_cd_mult": 0.8, "decoy_add": 0.15, "bl_add": 0.0},
    39: {"ip_cd_mult": 0.7, "decoy_add": 0.15, "bl_add": 0.0},
    40: {"ip_cd_mult": 0.7, "decoy_add": 0.10, "bl_add": 0.0},
    # MAVLink injection류 -> BL↑, 셔플 소폭 단축, 디코이 소폭↑
    3: {"ip_cd_mult": 0.8, "decoy_add": 0.05, "bl_add": 1.0},
    30: {"ip_cd_mult": 0.9, "decoy_add": 0.05, "bl_add": 1.0},
    31: {"ip_cd_mult": 0.9, "decoy_add": 0.05, "bl_add": 0.5},
    32: {"ip_cd_mult": 0.9, "decoy_add": 0.00, "bl_add": 0.5},
    # wifi deauth 등 -> BL↑
    48: {"ip_cd_mult": 0.95, "decoy_add": 0.00, "bl_add": 2.0},
    "_default": {"ip_cd_mult": 0.9, "decoy_add": 0.02, "bl_add": 0.5},
}

def read_cti_event(path: str):
    """CTI 이벤트 JSON 파일 읽기"""
    p = pathlib.Path(path)
    if not p.exists(): return None
    try:
        # 파일 읽기 오류 방지
        with p.open('r', encoding='utf-8') as f:
            content = f.read()
            # 비어있는 파일 처리
            if not content:
                return None
            return json.loads(content)
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {path}")
        return None
    except Exception as e:
        print(f"Warning: Error reading CTI event file {path}: {e}")
        return None


def event_to_cti_params(evt, default_exploit_p=0.9):
    """ CTI 이벤트를 환경 파라미터로 변환 (v19) """
    if not evt: return None
    return {
        "exploit_prob": float(evt.get("attack_prob", default_exploit_p)), # attack_prob -> exploit_prob
        "target_ips": np.array(evt.get("target_ips", []), dtype=int),
        "target_ports": np.array(evt.get("target_ports", []), dtype=int)
    }

def policy_offset_from_attack(evt):
    """CTI 이벤트에 따른 MTD 정책 오프셋 계산"""
    if not evt: return None
    prof = ATTACK_TO_POLICY.get(int(evt.get("attack_id", -1)), ATTACK_TO_POLICY["_default"])
    sev = float(evt.get("severity", 0.5))
    return {
        "ip_cd_mult": max(0.5, min(1.0, prof["ip_cd_mult"]**sev)),
        "decoy_add": max(0.0, prof["decoy_add"] * sev),
        "bl_add": prof["bl_add"] * sev
    }