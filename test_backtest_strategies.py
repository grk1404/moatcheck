"""Offline regression tests for the Back Test page (synthetic prices, no network).

Run:  python test_backtest_strategies.py     (or: python -m pytest test_backtest_strategies.py -q)
"""
import numpy as np
import pandas as pd

import backtest_strategies as bs
from backtest import run_backtest, _compute_sharpe

RNG = np.random.default_rng(7)
N = 1500
IDX = pd.bdate_range("2018-01-01", periods=N)


def _prices():
    r = RNG.normal(0.0008, 0.035, N)
    close = pd.Series(30 * np.cumprod(1 + r), index=IDX)
    open_ = close.shift(1).fillna(close.iloc[0]) * (1 + RNG.normal(0, 0.004, N))
    qqq = pd.Series(100 * np.cumprod(1 + r / 3), index=IDX)
    return pd.DataFrame({"open": open_, "close": close, "signal_close": qqq,
                         "qqq_open": qqq.shift(1).fillna(qqq.iloc[0]), "qqq_close": qqq})


P = _prices()
START = IDX[900]
W = P[P.index >= START]


def _sig(name, prices=P, start=START, **kw):
    return bs.STRATEGIES[name](prices, trade_start=start, swing_pct=0.15, leverage=3.0, **kw)


def test_no_weight_before_trade_start():
    for name in bs.STRATEGIES:
        s = _sig(name)
        assert (s.weight[s.weight.index < START] == 0).all(), name


def test_weights_in_unit_interval():
    for name in bs.STRATEGIES:
        w = _sig(name).weight
        assert w.between(0, 1).all() and w.notna().all(), name


def test_trade_state_independent_of_warmup_length():
    """Strategies whose only state is trade state must give identical results from START
    whether 100 or 800 bars of history precede it."""
    for name in ("Swing - since last exit", "One Percent A Week (TASC^)", "Buy and Hold (baseline)"):
        a = _sig(name, P.iloc[800:]).weight
        b = _sig(name, P.iloc[100:]).weight
        pd.testing.assert_series_equal(a[a.index >= START], b[b.index >= START], check_names=False, obj=name)


def test_swing_does_not_start_long_from_warmup():
    for name in ("Swing - since last exit", "Regime-Aware Swing"):
        w = _sig(name).weight
        first = w[w > 0].index
        assert len(first) == 0 or first[0] > START, name


def test_swing_variants_use_traded_tickers_open_for_entry():
    """Entry price is the traded ticker's open the bar after the signal (all swing variants alike)."""
    thr = 0.15
    for name in ("Swing - since last exit", "Swing - 20-day window", "Swing - 60-day window"):
        w = _sig(name).weight.to_numpy()
        o, c = P["open"].to_numpy(), P["close"].to_numpy()
        i0 = int(np.argmax(w > 0))
        entry = o[i0 + 1]
        j = i0 + 1
        while j < len(w) - 1 and w[j] > 0:
            assert c[j - 1] < entry * (1 + thr) or j == i0 + 1, name
            j += 1


def test_engine_matches_independent_equity_for_binary_weights():
    for name in ("Buy and Hold (baseline)", "Swing - since last exit", "Regime-Aware Swing"):
        s = _sig(name)
        r = run_backtest(W, s.weight, starting_cash=10_000.0)
        pos = s.weight.reindex(W.index).shift(1).fillna(0).to_numpy()
        pp = np.r_[0, pos[:-1]]
        o, c = W["open"].to_numpy(), W["close"].to_numpy()
        pc = np.r_[c[0], c[:-1]]
        f = np.where(pos == 1, np.where(pp == 1, c / pc, c / o), np.where(pp == 1, o / pc, 1.0))
        assert np.allclose(10_000.0 * np.cumprod(f), r.equity_curve.to_numpy(), rtol=1e-9), name


def test_engine_fractional_weight_math():
    """Constant 50% target from bar 1's open: equity ~ 0.5 invested, no rebalance cost at 0 bps
    except drift correction. Check day-1 value exactly."""
    w = pd.Series(0.5, index=W.index)
    r = run_backtest(W, w, starting_cash=10_000.0)
    o1, c1 = W["open"].iloc[1], W["close"].iloc[1]
    expect = 5_000.0 + 5_000.0 * c1 / o1
    assert abs(r.equity_curve.iloc[1] - expect) < 1e-6
    assert abs(r.equity_curve.iloc[0] - 10_000.0) < 1e-9


def test_costs_reduce_equity_and_zero_cost_is_free():
    s = _sig("Swing - since last exit")
    r0 = run_backtest(W, s.weight)
    r1 = run_backtest(W, s.weight, cost_bps=10.0)
    assert r1.equity_curve.iloc[-1] < r0.equity_curve.iloc[-1]
    flat = run_backtest(W, pd.Series(0.0, index=W.index), cost_bps=50.0)
    assert flat.equity_curve.iloc[-1] == 10_000.0


def test_exit_at_close_sells_at_close_not_next_open():
    w = pd.Series(0.0, index=W.index)
    w.iloc[3:6] = 1.0            # hold bars 4..6 (fills at open of bar 4)
    ex = pd.Series(False, index=W.index)
    ex.iloc[5] = True            # force sell at bar 5's close
    r = run_backtest(W, w, exit_at_close=ex)
    t = r.trades.iloc[0]
    assert t["exit_date"] == W.index[5]
    assert abs(t["exit_price"] - W["close"].iloc[5]) < 1e-9
    assert abs(t["entry_price"] - W["open"].iloc[4]) < 1e-9
    assert r.position.iloc[5] == 0


def test_one_percent_week_never_holds_over_weekend():
    s = _sig("One Percent A Week (TASC^)")
    r = run_backtest(W, s.weight, exit_at_close=s.exit_at_close)
    iso = W.index.isocalendar()
    wk = (iso.year * 100 + iso.week).to_numpy()
    last_of_week = np.r_[wk[1:] != wk[:-1], True]
    held = r.position.to_numpy() > 0
    assert not (held & last_of_week).any(), "position carried past a week's last close"


def test_one_percent_week_uses_traded_ticker_not_qqq():
    t = P.copy()
    t["signal_close"] = t["signal_close"].iloc[::-1].to_numpy()   # scramble the signal series
    a = bs.signal_1pct_week(P, trade_start=START).weight
    b = bs.signal_1pct_week(t, trade_start=START).weight
    pd.testing.assert_series_equal(a, b)


def test_sleeves_ramp_starts_from_cash_and_steps_at_most_10pts():
    w = _sig("Sleeves (traded live)").weight
    assert w[w.index < START].eq(0).all()
    d = w.diff().abs().dropna()
    assert d.max() <= 0.10 + 1e-12
    assert w[w.index >= START].iloc[0] <= 0.10 + 1e-12


def test_sleeves_matches_sleeves_strategy_module_when_started_early():
    import sleeves_strategy as sl
    full = sl.build_sleeve_targets(P["signal_close"])["tqqq"]
    mine = bs.signal_sleeves(P, trade_start=None, leverage=3.0).weight
    both = full.index
    # trade_start=None smooths from the first bar, the module from its first valid target: both ramp
    # from 0 under the same rules, so after the ramp settles they must agree.
    assert np.allclose(full.iloc[300:], mine.reindex(both).iloc[300:], atol=1e-9)


def test_dial_matches_live_vote_functions():
    import tqqq_sqqq_strategies as live
    w = _sig("7-Step Dial (live rules)").weight
    i = IDX.get_loc(IDX[1200])
    c = P["signal_close"].iloc[: i + 1]
    s50, sreg = c.rolling(50).mean().iloc[-1], c.rolling(live.REGIME_LONG_MA).mean().iloc[-1]
    mid, sd = c.rolling(20).mean().iloc[-1], c.rolling(20).std().iloc[-1]
    up, lo = mid + 2 * sd, mid - 2 * sd
    px = c.iloc[-1]
    regime = "BULL" if px > s50 > sreg else "BEAR" if px < s50 < sreg else "NEUTRAL"
    ind = {"price": px, "sma_50": s50, "sma_regime": sreg, "regime": regime, "dist_from_sma50": (px - s50) / s50,
           "bb_width": (up - lo) / mid, "bb_position": (px - lo) / (up - lo) * 100}
    votes = [live.STRATEGIES[k]({"_indicators": ind})["target_tqqq"] for k in range(7)]
    dev = sum(votes) / 7 - 0.5
    mult = live.SETTINGS["high_vol_multiplier"] if (regime != "NEUTRAL" and ind["bb_width"] > 0.10) else live.SETTINGS["low_vol_multiplier"]
    expect = live.dial_to_allocation(max(0, min(1, 0.5 + dev * mult)))["tqqq"]
    assert abs(w.iloc[i] - expect) < 1e-12


def test_benchmarks_enter_on_same_bar_as_strategies():
    r = run_backtest(W, pd.Series(1.0, index=W.index))
    assert abs(r.ticker_buy_hold_return - r.total_return) < 1e-9
    assert abs(r.benchmark_ticker.iloc[-1] - r.equity_curve.iloc[-1]) < 1e-6
    expect_q = W["qqq_close"].iloc[-1] / W["qqq_open"].iloc[1] - 1
    assert abs(r.qqq_buy_hold_return - expect_q) < 1e-9


def test_buy_hold_cagr_formula():
    r = run_backtest(W, pd.Series(1.0, index=W.index))
    yrs = (W.index[-1] - W.index[0]).days / 365.25
    expect = (W["close"].iloc[-1] / W["open"].iloc[1]) ** (1 / yrs) - 1  # first bar is cash; buy fills at bar 1's open
    assert abs(r.cagr - expect) < 1e-9


def test_sharpe_is_mean_over_std_annualized():
    r = pd.Series(RNG.normal(0.001, 0.02, 500))
    assert abs(_compute_sharpe(r) - np.sqrt(252) * r.mean() / r.std()) < 1e-12
    assert _compute_sharpe(pd.Series(dtype=float)) == 0.0
    assert _compute_sharpe(pd.Series([0.01] * 10)) == 0.0


def test_roundtrip_trade_log_pnl_adds_up():
    s = _sig("Swing - since last exit")
    r = run_backtest(W, s.weight, starting_cash=10_000.0)
    if r.trades.empty:
        return
    # all-in/all-out at 0 cost: compounded trade returns reproduce total return
    comp = float(np.prod(1 + r.trades["return_pct"]) - 1)
    assert abs(comp - r.total_return) < 1e-9


if __name__ == "__main__":
    import sys
    fails = 0
    for k, v in sorted(globals().items()):
        if k.startswith("test_") and callable(v):
            try:
                v()
                print("PASS", k)
            except Exception as e:
                fails += 1
                print("FAIL", k, "->", repr(e)[:300])
    sys.exit(1 if fails else 0)
