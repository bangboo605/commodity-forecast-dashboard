# -*- coding: utf-8 -*-
"""금 주요변수 6개의 설정(앱 CSV의 열 순서). 새 변수는 VariableConfig 한 줄을 추가하면 된다."""
from .selection import VariableConfig

CSV = "data/gold/gold_major_variables.csv"
GOLD = {c.code: c for c in [
    VariableConfig("X1_l1", CSV, "X1_l1", app_column="X1"),
    VariableConfig("X2", CSV, "X2"),
    VariableConfig("X3_l1", CSV, "X3_l1", app_column="X3"),
    VariableConfig("X4", CSV, "X4"),
    VariableConfig("X5_l1", CSV, "X5_l1", app_column="X5"),
    VariableConfig("X6_l1", CSV, "X6_l1", app_column="X6"),
]}
