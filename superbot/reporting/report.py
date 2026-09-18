"""
Report generation for SuperBot.
"""
from __future__ import annotations

import hashlib
import html
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from superbot.backtest.engine import BacktestResult

logger = logging.getLogger(__name__)

SUPERBOT_VERSION = "0.1.0"


@dataclass
class ReportMetadata:
    run_id: str
    generated_at: str
    superbot_version: str
    data_hash: str
    notes: str = ""

    @classmethod
    def create(cls, run_id: str, data_hash: str = "", notes: str = "") -> "ReportMetadata":
        return cls(
            run_id=run_id,
            generated_at=datetime.now(timezone.utc).isoformat(),
            superbot_version=SUPERBOT_VERSION,
            data_hash=data_hash,
            notes=notes,
        )


@dataclass
class MetricsBlock:
    n_bars: int = 0
    n_trades: int = 0
    n_wins: int = 0
    n_losses: int = 0
    win_rate: float = 0.0
    total_r: float = 0.0
    mean_r: float = 0.0
    median_r: float = 0.0
    std_r: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_dd_r: float = 0.0
    profit_factor: float = 0.0
    pnl_susdt: float = 0.0
    pnl_pct: float = 0.0
    fees_paid: float = 0.0
    exit_reasons: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_bars": self.n_bars,
            "n_trades": self.n_trades,
            "n_wins": self.n_wins,
            "n_losses": self.n_losses,
            "win_rate": self.win_rate,
            "total_r": self.total_r,
            "mean_r": self.mean_r,
            "median_r": self.median_r,
            "std_r": self.std_r,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "max_dd_r": self.max_dd_r,
            "profit_factor": self.profit_factor,
            "pnl_susdt": self.pnl_susdt,
            "pnl_pct": self.pnl_pct,
            "fees_paid": self.fees_paid,
            "exit_reasons": self.exit_reasons,
        }


def extract_metrics(result: BacktestResult) -> MetricsBlock:
    m = MetricsBlock()
    m.n_bars = result.n_signals
    m.pnl_susdt = result.pnl_susdt
    m.pnl_pct = result.pnl_pct

    fills = result.fills
    if not fills:
        return m

    trades = _extract_trades(fills)
    m.n_trades = len(trades)
    if not trades:
        return m

    r_values = np.array([t["r_multiple"] for t in trades], dtype=np.float64)

    m.n_wins = int((r_values > 0).sum())
    m.n_losses = int((r_values < 0).sum())
    m.win_rate = m.n_wins / m.n_trades if m.n_trades else 0.0
    m.total_r = float(r_values.sum())
    m.mean_r = float(r_values.mean())
    m.median_r = float(np.median(r_values))
    m.std_r = float(r_values.std(ddof=1)) if r_values.size > 1 else 0.0
    m.sharpe = _sharpe(r_values)
    m.sortino = _sortino(r_values)
    m.max_dd_r = _max_dd(r_values)
    m.profit_factor = _profit_factor(r_values)

    reasons: dict[str, int] = {}
    for t in trades:
        r = t["exit_reason"]
        reasons[r] = reasons.get(r, 0) + 1
    m.exit_reasons = dict(sorted(reasons.items(), key=lambda kv: -kv[1]))

    m.fees_paid = float(sum(f.fee for f in fills))
    return m


def _extract_trades(fills: list) -> list[dict[str, Any]]:
    """
    Group fills into complete trades.

    A trade is finalized when we see a TERMINAL exit event:
        SL, BE, TRAIL, TIMEEXIT, THESIS_EXIT, EXTERNAL_CLOSE, SLIPPAGE_EXIT, TP

    Partial exits (TP1, TP2) do NOT close the trade; they just update the
    exit_reason/exit_ts. Any remaining open trade at the end is closed with
    exit_reason='OPEN_AT_END'.
    """
    TERMINAL = {
        "SL", "BE", "TRAIL", "TIMEEXIT", "THESIS_EXIT",
        "EXTERNAL_CLOSE", "SLIPPAGE_EXIT", "TP",
    }
    PARTIAL = {"TP1", "TP2"}

    trades: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    legs: list[tuple[float, float]] = []

    for fill in fills:
        if fill.reason == "ENTRY":
            # Close any still-open trade before opening a new one
            if current is not None and legs:
                current["exit_legs"] = legs
                current["exit_reason"] = current.get("exit_reason") or "OPEN_AT_END"
                current["exit_ts"] = current.get("exit_ts") or fills[-1].timestamp_utc
                current["r_multiple"] = _compute_r(current, legs)
                trades.append(current)
            current = {
                "entry_ts": fill.timestamp_utc,
                "entry_price": fill.price,
                "side": fill.position_side,
                "qty": fill.quantity,
                "exit_ts": None,
                "exit_reason": "",
                "duration_bars": 0,
            }
            legs = []
        elif current is not None:
            legs.append((fill.price, fill.quantity))
            if fill.reason in PARTIAL:
                # Partial close: update exit metadata but do not finalize
                current["exit_ts"] = fill.timestamp_utc
                current["exit_reason"] = fill.reason
            elif fill.reason in TERMINAL:
                current["exit_ts"] = fill.timestamp_utc
                current["exit_reason"] = fill.reason
                current["exit_legs"] = legs
                current["r_multiple"] = _compute_r(current, legs)
                trades.append(current)
                current = None
                legs = []

    if current is not None and legs:
        current["exit_ts"] = current.get("exit_ts") or fills[-1].timestamp_utc
        current["exit_reason"] = current.get("exit_reason") or "OPEN_AT_END"
        current["exit_legs"] = legs
        current["r_multiple"] = _compute_r(current, legs)
        trades.append(current)

    for t in trades:
        if t["exit_ts"] is not None and t["entry_ts"] is not None:
            delta = pd.Timestamp(t["exit_ts"]) - pd.Timestamp(t["entry_ts"])
            t["duration_bars"] = int(delta.total_seconds() // (15 * 60))
    return trades


def _compute_r(trade: dict[str, Any], legs: list[tuple[float, float]]) -> float:
    if not legs:
        return 0.0
    entry = trade["entry_price"]
    side = trade["side"]
    total_qty = sum(q for _, q in legs)
    if total_qty <= 0:
        return 0.0
    avg_exit = sum(p * q for p, q in legs) / total_qty
    if side == "LONG":
        return (avg_exit - entry) / entry * 100.0
    return (entry - avg_exit) / entry * 100.0


def _sharpe(r: np.ndarray) -> float:
    if r.size < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(len(r)))


def _sortino(r: np.ndarray) -> float:
    if r.size < 2:
        return 0.0
    neg = r[r < 0]
    if neg.size == 0:
        return float(r.mean() * np.sqrt(len(r))) if r.mean() > 0 else 0.0
    if neg.size < 2:
        return 0.0
    dstd = neg.std(ddof=1)
    if dstd == 0 or not np.isfinite(dstd):
        return 0.0
    return float(r.mean() / dstd * np.sqrt(len(r)))


def _max_dd(r: np.ndarray) -> float:
    if r.size == 0:
        return 0.0
    cum = np.cumsum(r)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    return float(dd.max()) if dd.size else 0.0


def _profit_factor(r: np.ndarray) -> float:
    if r.size == 0:
        return 0.0
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    if losses <= 0:
        return float(gains) if gains > 0 else 0.0
    return float(gains / losses)


def compute_data_hash(*dfs: pd.DataFrame) -> str:
    h = hashlib.sha256()
    for df in dfs:
        h.update(str(df.shape).encode())
        h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return h.hexdigest()[:16]


CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f5f7fa; color: #2c3e50; margin: 0; padding: 24px; line-height: 1.6; }
.container { max-width: 1200px; margin: 0 auto; }
h1 { color: #1a252f; border-bottom: 3px solid #2E86DE; padding-bottom: 8px; }
h2 { color: #2c3e50; margin-top: 32px; border-left: 4px solid #2E86DE; padding-left: 12px; }
.meta { background: #fff; padding: 16px; border-radius: 8px;
    margin: 16px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.08); font-size: 14px; color: #555; }
.meta code { background: #ecf0f1; padding: 2px 6px; border-radius: 4px; }
.metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px; margin: 16px 0; }
.metric-card { background: #fff; padding: 16px; border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
.metric-label { font-size: 12px; color: #7f8c8d; text-transform: uppercase; letter-spacing: 0.5px; }
.metric-value { font-size: 24px; font-weight: 600; color: #2c3e50; margin-top: 4px; }
.metric-value.pos { color: #27AE60; }
.metric-value.neg { color: #E74C3C; }
table { width: 100%; border-collapse: collapse; background: #fff;
    border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin: 16px 0; }
th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid #ecf0f1; }
th { background: #2E86DE; color: #fff; font-weight: 500; font-size: 13px; }
tr:hover { background: #f8f9fa; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
td.pos { color: #27AE60; font-weight: 500; }
td.neg { color: #E74C3C; font-weight: 500; }
.warn { background: #fff3cd; padding: 12px; border-left: 4px solid #f39c12; border-radius: 4px; margin: 12px 0; }
.footer { text-align: center; color: #95a5a6; margin-top: 40px; font-size: 12px; }
.pill { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: 500; }
.pill.good { background: #d4edda; color: #155724; }
.pill.bad { background: #f8d7da; color: #721c24; }
"""


def _metric_card(label: str, value: str, cls: str = "") -> str:
    return f'<div class="metric-card"><div class="metric-label">{html.escape(label)}</div><div class="metric-value {cls}">{html.escape(value)}</div></div>'


def _render_metrics_grid(m: MetricsBlock) -> str:
    cards = [
        _metric_card("Trades", str(m.n_trades)),
        _metric_card("Win Rate", f"{m.win_rate:.1%}", "pos" if m.win_rate >= 0.5 else "neg"),
        _metric_card("Total R", f"{m.total_r:+.2f}", "pos" if m.total_r > 0 else "neg"),
        _metric_card("Mean R", f"{m.mean_r:+.3f}", "pos" if m.mean_r > 0 else "neg"),
        _metric_card("Sortino", f"{m.sortino:+.2f}", "pos" if m.sortino > 0 else "neg"),
        _metric_card("Sharpe", f"{m.sharpe:+.2f}", "pos" if m.sharpe > 0 else "neg"),
        _metric_card("Max DD (R)", f"{m.max_dd_r:.2f}", "neg"),
        _metric_card("Profit Factor", f"{m.profit_factor:.2f}", "pos" if m.profit_factor >= 1.0 else "neg"),
        _metric_card("PnL (SUSDT)", f"{m.pnl_susdt:+.2f}", "pos" if m.pnl_susdt > 0 else "neg"),
        _metric_card("PnL %", f"{m.pnl_pct:+.2f}%", "pos" if m.pnl_pct > 0 else "neg"),
        _metric_card("Fees Paid", f"{m.fees_paid:.2f}"),
        _metric_card("Bars", str(m.n_bars)),
    ]
    return f'<div class="metrics-grid">{"".join(cards)}</div>'


def _render_exit_reasons_table(reasons: dict[str, int]) -> str:
    total = sum(reasons.values())
    rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td class='num'>{v}</td><td class='num'>{v/total*100:.1f}%</td></tr>"
        for k, v in reasons.items()
    )
    return f"<table><thead><tr><th>Reason</th><th>Count</th><th>%</th></tr></thead><tbody>{rows}</tbody></table>"


def _render_trades_table(trades: list[dict[str, Any]]) -> str:
    if not trades:
        return "<p>No trades</p>"
    rows = []
    for i, t in enumerate(trades):
        r = t["r_multiple"]
        cls = "pos" if r > 0 else "neg" if r < 0 else ""
        entry_ts = pd.Timestamp(t["entry_ts"]).strftime("%Y-%m-%d %H:%M") if t.get("entry_ts") else ""
        exit_ts = pd.Timestamp(t["exit_ts"]).strftime("%Y-%m-%d %H:%M") if t.get("exit_ts") else ""
        last_exit_price = t.get("exit_legs", [[0, 0]])[-1][0] if t.get("exit_legs") else 0
        rows.append(
            f"<tr><td>{i + 1}</td><td>{html.escape(entry_ts)}</td><td>{html.escape(exit_ts)}</td>"
            f"<td>{html.escape(t['side'])}</td><td class='num'>{t['entry_price']:.2f}</td>"
            f"<td class='num'>{last_exit_price:.2f}</td><td class='num {cls}'>{r:+.2f}</td>"
            f"<td>{html.escape(t.get('exit_reason', ''))}</td><td class='num'>{t.get('duration_bars', 0)}</td></tr>"
        )
    return f"""<table><thead><tr><th>#</th><th>Entry</th><th>Exit</th><th>Side</th>
    <th>Entry Px</th><th>Exit Px</th><th>R</th><th>Reason</th><th>Bars</th></tr></thead>
    <tbody>{"".join(rows)}</tbody></table>"""


def _render_metadata_block(meta: ReportMetadata) -> str:
    return f"""<div class="meta">
      <div><strong>Run ID:</strong> <code>{html.escape(meta.run_id)}</code></div>
      <div><strong>Generated:</strong> {html.escape(meta.generated_at)}</div>
      <div><strong>SuperBot:</strong> v{html.escape(meta.superbot_version)}</div>
      <div><strong>Data hash:</strong> <code>{html.escape(meta.data_hash)}</code></div>
    </div>"""


def _render_html(title: str, meta: ReportMetadata, sections: list[tuple[str, str]]) -> str:
    body = "\n".join(f"<h2>{html.escape(s_title)}</h2>\n{s_html}" for s_title, s_html in sections)
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}</style></head>
<body><div class="container"><h1>{html.escape(title)}</h1>
{_render_metadata_block(meta)}
{body}
<div class="footer">SuperBot v{html.escape(meta.superbot_version)} &middot; {html.escape(meta.generated_at)}</div>
</div></body></html>"""


def generate_backtest_report(
    result: BacktestResult,
    output_dir: str | Path,
    baseline_metrics: MetricsBlock | None = None,
    notes: str = "",
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = extract_metrics(result)
    trades = _extract_trades(result.fills)

    if result.signals:
        hash_df = pd.DataFrame({
            "bar_ts": [s.bar_ts for s in result.signals],
            "signal": [s.final_signal for s in result.signals],
        })
    else:
        hash_df = pd.DataFrame()
    data_hash = compute_data_hash(hash_df)

    meta = ReportMetadata.create(run_id=result.run_id, data_hash=data_hash, notes=notes)

    sections: list[tuple[str, str]] = []
    sections.append(("Metrics", _render_metrics_grid(metrics)))

    if metrics.exit_reasons:
        sections.append(("Exit Reasons", _render_exit_reasons_table(metrics.exit_reasons)))

    if trades:
        sorted_trades = sorted(trades, key=lambda t: -t["r_multiple"])
        best = sorted_trades[:10]
        worst = sorted_trades[-10:][::-1] if len(sorted_trades) > 10 else sorted_trades[::-1]
        sections.append(("Top 10 Best Trades", _render_trades_table(best)))
        sections.append(("Top 10 Worst Trades", _render_trades_table(worst)))

    paths: dict[str, Path] = {}

    html_path = output_dir / "report.html"
    html_path.write_text(_render_html(f"Backtest Report — {result.run_id}", meta, sections), encoding="utf-8")
    paths["html"] = html_path

    json_path = output_dir / "metrics.json"
    json_path.write_text(json.dumps({
        "metadata": {"run_id": meta.run_id, "generated_at": meta.generated_at,
                     "superbot_version": meta.superbot_version,
                     "data_hash": meta.data_hash, "notes": meta.notes},
        "metrics": metrics.to_dict(),
    }, indent=2, default=str), encoding="utf-8")
    paths["json"] = json_path

    md_path = output_dir / "report.md"
    md_path.write_text(_render_markdown(meta, metrics, trades), encoding="utf-8")
    paths["md"] = md_path

    logger.info("Saved backtest report to %s", output_dir)
    return paths


def _render_markdown(meta: ReportMetadata, m: MetricsBlock, trades: list[dict[str, Any]]) -> str:
    lines = [
        f"# Backtest Report — {meta.run_id}",
        "",
        f"- Generated: `{meta.generated_at}`",
        f"- Data hash: `{meta.data_hash}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Trades | {m.n_trades} |",
        f"| Win Rate | {m.win_rate:.1%} |",
        f"| Total R | {m.total_r:+.2f} |",
        f"| Sortino | {m.sortino:+.2f} |",
        f"| PnL (SUSDT) | {m.pnl_susdt:+.2f} |",
        f"| PnL % | {m.pnl_pct:+.2f}% |",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    import yaml
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    if len(sys.argv) < 2:
        print("Usage: python -m superbot.reporting.report <config.yaml>")
        sys.exit(1)
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    from superbot.backtest.data_loader import load_market_data
    from superbot.backtest.engine import BacktestEngine
    md = load_market_data(cfg["paths"]["data_csv"], use_cache=False)
    engine = BacktestEngine(cfg, md)
    result = engine.run()
    out_dir = Path("superbot_runs") / result.run_id
    paths = generate_backtest_report(result, out_dir)
    for k, v in paths.items():
        print(f"  {k}: {v}")