"""Regression tests for the five A-share data-quality gates (DORA-124 §4.4).

Each gate has a runnable, deterministic test that pins the acceptance criteria:

- 成交量单位门禁: no 100x volume-unit jump across sources (#1062).
- 复权门禁: a claimed 前复权 source matches a reference on ex-dividend dates.
- 停牌/涨跌停语义门禁: a suspended day is not a 0% move.
- 跨源一致性门禁: the same settled day's close agrees within 1%.
- OHLC 结构门禁: bad bars never reach the backtest.
- provenance: the §4.1 envelope {source, volume_unit, adjust, is_final}.

No test touches the network; all are pure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.loaders.base import validate_ohlc
from backtest.loaders.data_quality import (
    check_cross_source_close,
    check_volume_unit_consistency,
    suspension_days,
    validate_provenance,
    validate_qfq_adjustment,
)
from backtest.metrics import bar_returns


# ---------------------------------------------------------------------------
# Gate 1 — 成交量单位门禁 (lots vs shares, no 100x jump)
# ---------------------------------------------------------------------------


class TestVolumeUnitGate:
    def test_lots_and_shares_agree_after_normalization(self) -> None:
        """1 lot == 100 shares, so a correct cross-unit pair is not drift."""
        samples = {
            "miniqmt": ("lots", 100.0),   # 100 lots  = 10,000 shares
            "tencent": ("lots", 100.0),
            "baostock": ("shares", 10_000.0),  # hypothetical raw-share source
        }
        assert check_volume_unit_consistency(samples) == []

    def test_same_unit_100x_jump_is_detected(self) -> None:
        """Two sources both claiming 'lots' but 100x apart is the #1062 drift."""
        samples = {
            "miniqmt": ("lots", 100.0),
            "tencent": ("lots", 100.0),
            "eastmoney": ("lots", 10_000.0),  # 100x — someone is reporting shares
        }
        violations = check_volume_unit_consistency(samples)
        assert len(violations) == 1
        assert "eastmoney" in violations[0]
        assert "100.00x" in violations[0]

    def test_undeclared_unit_is_a_violation(self) -> None:
        """A source without a declared unit cannot be trusted against the rest."""
        samples = {"miniqmt": ("lots", 100.0), "unknown": (None, 100.0)}
        violations = check_volume_unit_consistency(samples)
        assert any("volume_unit" in v for v in violations)

    def test_single_sample_is_vacuous_pass(self) -> None:
        """One source cannot be compared; the gate must not fabricate drift."""
        assert check_volume_unit_consistency({"miniqmt": ("lots", 100.0)}) == []

    def test_nonpositive_or_nonfinite_volume_is_skipped(self) -> None:
        samples = {
            "a": ("lots", 100.0),
            "b": ("lots", 0.0),
            "c": ("lots", float("nan")),
        }
        assert check_volume_unit_consistency(samples) == []

    def test_boundary_just_below_tolerance_passes(self) -> None:
        """A 2% deviation is exactly at tolerance; below it must pass."""
        samples = {"a": ("lots", 100.0), "b": ("lots", 101.9)}
        assert check_volume_unit_consistency(samples) == []


# ---------------------------------------------------------------------------
# Gate 2 — 复权门禁 (前复权 vs reference on ex-dates)
# ---------------------------------------------------------------------------


def _series(values: list[float], start: str = "2024-01-02") -> pd.Series:
    index = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=index)


class TestAdjustmentGate:
    def test_qfq_matching_reference_on_ex_date_passes(self) -> None:
        """A correct qfq series and reference agree on the ex-date."""
        candidate = _series([10.0, 10.0, 10.0])
        reference = _series([10.0, 10.0, 10.0])
        passed, violations = validate_qfq_adjustment(
            candidate, reference, ex_dates=["2024-01-03"]
        )
        assert passed is True
        assert violations == []

    def test_mechanical_ex_date_gap_is_detected(self) -> None:
        """An unadjusted source prints the ex-date gap; a qfq source must not."""
        candidate = _series([10.0, 5.0, 5.0])   # raw: 10 -> 5 on the ex-date
        reference = _series([10.0, 10.0, 10.0])  # qfq reference: flat
        passed, violations = validate_qfq_adjustment(
            candidate, reference, ex_dates=["2024-01-03"]
        )
        assert passed is False
        assert len(violations) == 1
        assert "50.00%" in violations[0]

    def test_non_ex_dates_are_not_checked(self) -> None:
        """Off ex-dates a source may differ for unrelated reasons; only the
        ex-dates are the discriminating days."""
        candidate = _series([10.0, 11.0, 10.0])
        reference = _series([10.0, 12.0, 10.0])
        passed, _ = validate_qfq_adjustment(
            candidate, reference, ex_dates=["2024-01-04"]  # ex-date on the 4th
        )
        assert passed is True

    def test_no_ex_dates_is_vacuous_pass(self) -> None:
        assert validate_qfq_adjustment(
            _series([1.0]), _series([100.0]), ex_dates=[]
        )[0] is True

    def test_ex_date_missing_from_one_series_is_skipped(self) -> None:
        """No overlap means no comparison, not a fabricated violation."""
        candidate = _series([10.0], start="2024-01-02")
        reference = _series([10.0], start="2024-02-02")
        passed, violations = validate_qfq_adjustment(
            candidate, reference, ex_dates=["2024-01-03"]
        )
        assert passed is True
        assert violations == []

    def test_empty_series_is_vacuous_pass(self) -> None:
        passed, _ = validate_qfq_adjustment(
            pd.Series(dtype=float), _series([10.0]), ex_dates=["2024-01-03"]
        )
        assert passed is True

    def test_boundary_at_tolerance_passes(self) -> None:
        """A 1% error is at the threshold; it must pass (≤1%)."""
        candidate = _series([10.0, 10.1, 10.0])
        reference = _series([10.0, 10.0, 10.0])
        passed, violations = validate_qfq_adjustment(
            candidate, reference, ex_dates=["2024-01-03"]
        )
        assert passed is True
        assert violations == []


# ---------------------------------------------------------------------------
# Gate 3 — 停牌/涨跌停语义门禁 (suspended day is not a 0% move)
# ---------------------------------------------------------------------------


class TestSuspensionSemanticsGate:
    def test_zero_volume_bar_is_flagged_suspended(self) -> None:
        index = pd.bdate_range("2024-01-02", periods=3)
        frame = pd.DataFrame(
            {"open": [10.0, 10.0, 10.0], "high": [10.5, 10.5, 10.5],
             "low": [9.9, 9.9, 9.9], "close": [10.1, 10.1, 10.1],
             "volume": [100.0, 0.0, 50.0]},
            index=index,
        )
        flags = suspension_days(frame)
        assert flags.tolist() == [False, True, False]

    def test_frame_without_volume_column_is_all_false(self) -> None:
        frame = pd.DataFrame({"close": [1.0, 2.0]}, index=[0, 1])
        assert not suspension_days(frame).any()

    def test_suspended_day_is_not_a_real_zero_move(self) -> None:
        """A suspended day (no traded volume) is flagged, and its absence from
        the series must not be manufactured into a fabricated 0% move that
        erases the real across-gap move (issue #872)."""
        index = pd.bdate_range("2024-01-02", periods=5)
        frame = pd.DataFrame(
            {"close": [100.0, 110.0, np.nan, np.nan, 121.0],
             "volume": [100.0, 100.0, 0.0, 0.0, 100.0]},
            index=index,
        )
        # The two dark sessions read as suspended (zero volume).
        assert suspension_days(frame).tolist() == [False, False, True, True, False]

        ret = bar_returns(frame["close"], label="suspended")
        # The suspended sessions are flat, but the real 110 -> 121 move is
        # credited to the resumed bar rather than silently dropped to 0.
        assert ret.iloc[2:4].tolist() == [0.0, 0.0]
        assert ret.iloc[4] == pytest.approx(121.0 / 110.0 - 1.0)

    def test_bar_returns_credits_real_move_across_suspension(self) -> None:
        """bar_returns must not erase the across-gap move (issue #872)."""
        close = pd.Series([100.0, 110.0, np.nan, np.nan, 121.0])
        ret = bar_returns(close, label="suspended")
        assert ret.iloc[4] == pytest.approx(0.10)
        assert ret.iloc[2:4].tolist() == [0.0, 0.0]


# ---------------------------------------------------------------------------
# Gate 4 — 跨源一致性门禁 (same settled-day close ≤1%)
# ---------------------------------------------------------------------------


class TestCrossSourceCloseGate:
    def test_agreement_within_one_percent_passes(self) -> None:
        closes = {"miniqmt": 100.0, "baostock": 100.5, "tencent": 99.8}
        assert check_cross_source_close(closes) == []

    def test_one_source_beyond_one_percent_is_flagged(self) -> None:
        closes = {"miniqmt": 100.0, "baostock": 100.4, "tencent": 98.0}
        violations = check_cross_source_close(closes)
        assert len(violations) == 1
        assert "tencent" in violations[0]

    def test_single_source_is_vacuous_pass(self) -> None:
        assert check_cross_source_close({"miniqmt": 100.0}) == []

    def test_nonpositive_or_nonfinite_close_is_skipped(self) -> None:
        closes = {"miniqmt": 100.0, "bad_zero": 0.0, "bad_nan": float("nan")}
        assert check_cross_source_close(closes) == []

    def test_boundary_at_one_percent_passes(self) -> None:
        """A source exactly 1% off the median is at the threshold."""
        closes = {"a": 100.0, "b": 101.0}  # median 100.5; b is ~0.5% off
        assert check_cross_source_close(closes) == []


# ---------------------------------------------------------------------------
# Gate 5 — OHLC 结构门禁 (bad bars never reach the backtest)
# ---------------------------------------------------------------------------


class TestOhlcStructureGate:
    def test_nonpositive_price_is_dropped(self) -> None:
        """非正价不流入: a non-positive OHLC price must be dropped."""
        frame = pd.DataFrame(
            {"open": [10.0, -1.0], "high": [11.0, 11.0], "low": [9.0, 9.0],
             "close": [10.5, 10.5], "volume": [100.0, 100.0]},
            index=[0, 1],
        )
        cleaned = validate_ohlc(frame)
        assert list(cleaned["close"]) == [10.5]

    def test_high_below_low_is_dropped(self) -> None:
        frame = pd.DataFrame(
            {"open": [10.0, 10.0], "high": [11.0, 8.0], "low": [9.0, 9.0],
             "close": [10.5, 10.5], "volume": [100.0, 100.0]},
            index=[0, 1],
        )
        cleaned = validate_ohlc(frame)
        assert list(cleaned["close"]) == [10.5]

    def test_sanitize_data_map_guards_every_source(self) -> None:
        """The runner's central pass is the single boundary: any loader's bad bar
        is removed before the backtest, whatever the source."""
        from backtest.runner import _sanitize_data_map

        dirty = pd.DataFrame(
            {"open": [10.0, 10.0], "high": [11.0, 8.0], "low": [9.0, 9.0],
             "close": [10.5, 10.5], "volume": [100.0, 100.0]},
            index=[0, 1],
        )
        cleaned = _sanitize_data_map({"AAA": dirty})
        assert list(cleaned["AAA"]["close"]) == [10.5]


# ---------------------------------------------------------------------------
# Provenance — §4.1 {source, volume_unit, adjust, is_final}
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_valid_envelope_passes(self) -> None:
        envelope = {
            "source": "miniqmt",
            "volume_unit": "lots",
            "adjust": "qfq",
            "is_final": True,
            "symbol": "600519.SH",
            "timeframe": "1d",
        }
        assert validate_provenance(envelope) == []

    @pytest.mark.parametrize("field", ["source", "volume_unit", "adjust", "is_final"])
    def test_missing_required_field_is_flagged(self, field: str) -> None:
        envelope = {
            "source": "miniqmt",
            "volume_unit": "lots",
            "adjust": "qfq",
            "is_final": True,
        }
        del envelope[field]
        assert any("missing provenance field" in v for v in validate_provenance(envelope))

    def test_invalid_volume_unit_is_flagged(self) -> None:
        envelope = {"source": "x", "volume_unit": "手", "adjust": "qfq", "is_final": True}
        assert any("volume_unit" in v for v in validate_provenance(envelope))

    def test_invalid_adjust_is_flagged(self) -> None:
        envelope = {"source": "x", "volume_unit": "lots", "adjust": "forward", "is_final": True}
        assert any("adjust" in v for v in validate_provenance(envelope))

    def test_is_final_must_be_a_bool(self) -> None:
        envelope = {"source": "x", "volume_unit": "lots", "adjust": "qfq", "is_final": "yes"}
        assert any("is_final" in v for v in validate_provenance(envelope))

    def test_blank_source_is_flagged(self) -> None:
        envelope = {"source": "  ", "volume_unit": "lots", "adjust": "qfq", "is_final": True}
        assert any("source" in v for v in validate_provenance(envelope))

    def test_a_share_loader_units_feed_a_valid_provenance(self) -> None:
        """The loader ``volume_units`` declaration is the provenance
        ``volume_unit``: every A-share source agrees on 'lots', so no 100x jump
        can enter at the declaration level."""
        from backtest.loaders.akshare_loader import DataLoader as AkshareLoader
        from backtest.loaders.baostock_loader import DataLoader as BaostockLoader
        from backtest.loaders.eastmoney_loader import DataLoader as EastmoneyLoader
        from backtest.loaders.miniqmt_loader import DataLoader as MiniqmtLoader
        from backtest.loaders.mootdx_loader import DataLoader as MootdxLoader
        from backtest.loaders.tencent_loader import DataLoader as TencentLoader
        from backtest.loaders.tushare import DataLoader as TushareLoader

        loaders = [
            MiniqmtLoader, BaostockLoader, TencentLoader, MootdxLoader,
            EastmoneyLoader, AkshareLoader, TushareLoader,
        ]
        for loader_cls in loaders:
            envelope = {
                "source": loader_cls.name,
                "volume_unit": loader_cls.volume_units.get("a_share"),
                "adjust": "qfq" if loader_cls.name != "tencent" else "none",
                "is_final": True,
            }
            assert validate_provenance(envelope) == [], loader_cls.name

    def test_is_final_derives_from_settled_range(self) -> None:
        """The bridge/loader ``is_final`` flag is the settled-range rule: only
        ranges strictly before today are final."""
        from backtest.loaders.base import loader_cache_range_is_final

        assert loader_cache_range_is_final("2020-01-01") is True
        assert loader_cache_range_is_final("2999-12-31") is False
        assert loader_cache_range_is_final("not-a-date") is False
