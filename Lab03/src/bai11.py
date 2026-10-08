"""Bai 11 - Policy cố định"""
import os, sys
os.environ.setdefault("MPLBACKEND", "Agg")            # luu hinh, khong mo cua so
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from mc_utils import *
from experiments import *


def main():
    print("Ý nghĩa action:", ACTION_NAMES)
    for state in [(21, 5, False), (20, 10, False), (19, 10, False), (12, 6, False), (15, 3, True)]:
        action = stick_on_20_policy(state)
        print(f"state={state} -> action {action} ({ACTION_NAMES[action]})")


if __name__ == "__main__":
    main()
