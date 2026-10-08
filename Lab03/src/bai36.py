"""Bai 36 - Monte Carlo Agent hoàn chỉnh"""
import os, sys
os.environ.setdefault("MPLBACKEND", "Agg")            # luu hinh, khong mo cua so
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from mc_utils import *
from experiments import *
from main import run_mini_project

def main():
    results = run_mini_project()


if __name__ == "__main__":
    main()
