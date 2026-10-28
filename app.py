"""Streamlit UI: update draws, manual entry, run the pipeline, view tickets/backtest/curves.

Run:
    C:/ProgramData/miniconda3/envs/Predict/python.exe -m streamlit run app.py
"""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from data.loader import MAIN_COLS, draw_date, load_draws
from data.updater import append_draws
from main import main

DATA = os.environ.get("LOTTO_DATA", "lotto.xlsx")

st.set_page_config(page_title="Lotto ML", layout="wide")
st.title("로또 6/45 단독 모델")

with st.sidebar:
    epochs = st.number_input("epochs", 1, 500, 50, key="epochs")
    tickets = st.number_input("티켓 수", 1, 50, 5, key="tickets")
    window = st.number_input("window (회차)", 10, 520, 52, key="window")
    auto_update = st.checkbox("자동 갱신", True, key="auto_update")

try:
    df = load_draws(DATA)
except (OSError, ValueError) as e:
    st.error(f"{DATA}: {e}")
    st.stop()
last = df.iloc[-1]
st.subheader(f"마지막 회차: {last['draw']}회 ({draw_date(int(last['draw']))})")
st.markdown(" ".join(f"`{last[c]}`" for c in MAIN_COLS) + f" + 보너스 `{last['bonus']}`")

with st.expander("당첨번호 수동 입력"):
    with st.form("add"):
        draw = st.number_input("회차", 1, value=int(last["draw"]) + 1, key="add_draw")
        cols = st.columns(7)
        # No defaults: a stray click must not save a plausible-looking fake draw.
        nums = [cols[i].number_input(f"n{i+1}", 1, 45, None, key=f"add_n{i+1}") for i in range(6)]
        bonus = cols[6].number_input("보너스", 1, 45, None, key="add_bonus")
        if st.form_submit_button("추가", key="add_submit"):
            if None in (*nums, bonus):
                st.error("번호 6개와 보너스를 모두 입력하세요.")
            else:
                try:
                    append_draws(DATA, [{"draw": draw, **{f"n{i+1}": v for i, v in enumerate(nums)}, "bonus": bonus}])
                    st.success(f"{draw}회 추가됨")
                    st.rerun()
                except (ValueError, OSError) as e:
                    st.error(str(e))

if st.button("실행", key="run", type="primary"):
    # UI is AR-only; SEDD/HGNN stay CLI-only (--model) for comparison.
    argv = ["--data", DATA, "--model", "ar", "--epochs", str(epochs), "--tickets", str(tickets),
            "--window", str(window)] + ([] if auto_update else ["--no-update"])
    with st.spinner("갱신 + 학습 + 백테스트 중..."):
        st.session_state["result"] = main(argv)

res = st.session_state.get("result")
if res:
    if res["update_warning"]:
        st.warning(res["update_warning"])
    else:
        st.info(f"새 회차 {res['added']}개 추가 (마지막 {res['last_draw']}회)")
    st.caption(f"sample_seed = {res['sample_seed']}")

    st.subheader(f"{res['last_draw'] + 1}회 추천 티켓")
    st.dataframe(pd.DataFrame(
        [{"model": n, "ticket": i + 1, **{f"n{j+1}": b for j, b in enumerate(t)}}
         for n, r in res["models"].items() for i, t in enumerate(r["next_draw"])]
    ), hide_index=True)

    st.subheader("백테스트 (테스트 구간)")
    keys = ["mean_matches", "baseline_mean_matches", "rate_ge3", "baseline_rate_ge3", "p_value", "unique_ratio"]
    st.dataframe(pd.DataFrame({n: {k: r["backtest"][k] for k in keys} for n, r in res["models"].items()}).T)

    st.subheader("학습 곡선")
    for n, r in res["models"].items():
        st.markdown(f"**{n}**")
        st.line_chart(pd.DataFrame(r["history"]))

st.caption("로또 추첨은 독립 시행입니다. 기준선 대비 p_value가 작지 않으면 무작위 선택과 차이가 없습니다.")
