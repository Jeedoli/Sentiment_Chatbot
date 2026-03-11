"""
tests/test_preprocess.py
──────────────────
`load_local_csv` 함수와 전처리 도우미에 대한 단위 테스트.
"""

import os
import zipfile

import pandas as pd
import pytest

from scripts.preprocess import load_local_csv


def make_csv(path, rows=None):
    if rows is None:
        rows = [
            {"text": "좋아요", "label": 2},
            {"text": "별로예요", "label": 0},
        ]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_load_csv(tmp_path):
    csv = tmp_path / "a.csv"
    make_csv(csv)
    df = load_local_csv(str(csv))
    assert list(df.columns) == ["text", "label"]
    assert len(df) == 2


def test_load_excel(tmp_path):
    excel = tmp_path / "a.xlsx"
    df_orig = pd.DataFrame({"text": ["x"], "label": [1]})
    df_orig.to_excel(excel, index=False)
    df = load_local_csv(str(excel))
    assert df.iloc[0]["text"] == "x"
    assert df.iloc[0]["label"] == 1


def test_load_directory(tmp_path):
    d = tmp_path / "dir"
    d.mkdir()
    make_csv(d / "one.csv")
    make_csv(d / "two.csv")
    df = load_local_csv(str(d))
    # 두 개의 파일이 합쳐져야 함
    assert len(df) == 4


def test_load_zip(tmp_path):
    csv = tmp_path / "inner.csv"
    make_csv(csv)
    z = tmp_path / "stuff.zip"
    with zipfile.ZipFile(z, "w") as zp:
        zp.write(csv, arcname="inner.csv")
    df = load_local_csv(str(z))
    assert len(df) == 2
