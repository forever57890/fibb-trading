"""
Structured + human-readable logging for fibb_15m_trader runs.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fmt_balance(bal: dict) -> str:
    if not bal or not bal.get("found"):
        return bal.get("note", "balance N/A") if bal else "balance N/A"
    return (
        f"wallet={bal.get('wallet_balance')} "
        f"avail={bal.get('available_balance')} "
        f"cross_un_pnl={bal.get('cross_un_pnl')}"
    )


def _fmt_positions(positions: List[dict]) -> str:
    if not positions:
        return "(flat)"
    parts = []
    for p in positions:
        parts.append(
            f"{p.get('position_side')} amt={p.get('position_amt')} "
            f"entry={p.get('entry_price')} uPnL={p.get('unrealized_pnl')}"
        )
    return "; ".join(parts)


def summarize_bar(bar: Optional[dict]) -> Dict[str, Any]:
    if not bar:
        return {}
    entries = bar.get("entries") or []
    opened = [e for e in entries if not e.get("skipped")]
    skipped_entries = [e for e in entries if e.get("skipped")]
    diag = bar.get("entry_diagnostics") or []
    touch_signals = [d for d in diag if d.get("status") == "touch_signal"]
    already_open = [d for d in diag if d.get("status") == "already_open"]
    no_touch = [d for d in diag if d.get("status") == "no_touch"]

    return {
        "bar_time": bar.get("bar_time"),
        "close": bar.get("close"),
        "skipped": bar.get("skipped"),
        "skip_reason": bar.get("reason"),
        "exit_count": len(bar.get("exits") or []),
        "entry_opened_count": len(opened),
        "entry_skipped_count": len(skipped_entries),
        "armed_stop_count": len(bar.get("armed_stops") or []),
        "open_legs_after": bar.get("open_legs_after"),
        "touch_signal_legs": [d.get("entry_id") for d in touch_signals],
        "already_open_legs": [d.get("entry_id") for d in already_open],
        "no_touch_count": len(no_touch),
        "headline": bar.get("headline"),
        "headline_zh": bar.get("headline_zh"),
    }


def build_bar_headline(bar: dict) -> tuple[str, str]:
    """English key + Traditional Chinese headline for the bar step."""
    if bar.get("skipped"):
        reason = bar.get("reason") or "unknown"
        if reason == "already_processed":
            return (
                "SKIPPED_ALREADY_PROCESSED",
                f"略過：K 線 {bar.get('bar_time', '?')} 已處理過",
            )
        if reason == "warmup":
            return ("SKIPPED_WARMUP", "略過：暖機不足，尚無有效通道")
        return ("SKIPPED", f"略過：{reason}")

    exits = bar.get("exits") or []
    entries = bar.get("entries") or []
    opened = [e for e in entries if not e.get("skipped")]
    diag = bar.get("entry_diagnostics") or []

    parts: List[str] = []
    if exits:
        ids = [e.get("entry_id") for e in exits]
        parts.append(f"平倉 {len(exits)} 筆：{', '.join(ids)}")
    if opened:
        ids = [e.get("entry_id") for e in opened]
        parts.append(f"開倉 {len(opened)} 筆：{', '.join(ids)}")
    for e in entries:
        if e.get("skipped"):
            parts.append(
                f"{e.get('entry_id')} 未開：{e.get('reason', 'skipped')}"
            )

    touch_unopened = [
        d
        for d in diag
        if d.get("status") == "touch_signal"
        and d.get("entry_id") not in {x.get("entry_id") for x in opened}
    ]
    for d in touch_unopened:
        extra = d.get("blocked_reason") or "未執行開倉"
        parts.append(f"{d.get('entry_id')} 觸軌但未開：{extra}")

    open_after = bar.get("open_legs_after") or []
    if not parts:
        if open_after:
            hold = bar.get("hold_diagnostics") or []
            if hold:
                parts.append(
                    f"持倉 {len(open_after)} leg（{', '.join(open_after)}）：本根未觸及出場"
                )
            else:
                parts.append(f"持倉 {len(open_after)} leg，本根無交易")
        else:
            touched = [d for d in diag if d.get("status") == "touch_signal"]
            if touched:
                parts.append("有觸軌訊號但未成功開倉（見下方 leg 明細）")
            else:
                parts.append("本根無觸軌進場；無持倉或持倉未出場")

    zh = "；".join(parts)
    action = "BAR_PROCESSED"
    if opened and exits:
        action = "CLOSED_AND_OPENED"
    elif opened:
        action = "OPENED"
    elif exits:
        action = "CLOSED"
    elif not open_after and not exits and not opened:
        action = "NO_TRADE"
    return (action, zh)


def build_action_detail(result: dict) -> Dict[str, Any]:
    action = result.get("action", "")
    detail: Dict[str, Any] = {"action": action}
    bar = result.get("bar") or {}

    if action == "SKIPPED":
        detail["reason"] = (bar.get("reason") or "bar_skipped")
        detail["bar_time"] = bar.get("bar_time")
        if bar.get("reason") == "already_processed":
            detail["reason_zh"] = "state.last_bar_time 與本根相同，cron 重複執行"
        elif bar.get("reason") == "warmup":
            detail["reason_zh"] = "bar_index < 1，通道尚未計算完成"
    elif action == "ERROR":
        detail["reason"] = result.get("error")
    else:
        bar_sum = summarize_bar(bar)
        detail["headline_zh"] = bar_sum.get("headline_zh")
        detail["bar_time"] = bar_sum.get("bar_time")
        detail["entry_opened_count"] = bar_sum.get("entry_opened_count")
        detail["exit_count"] = bar_sum.get("exit_count")
        if bar_sum.get("entry_opened_count", 0) == 0 and not bar.get("skipped"):
            detail["no_entry_reason_zh"] = _explain_no_entries(bar)
    return detail


def _explain_no_entries(bar: dict) -> str:
    diag = bar.get("entry_diagnostics") or []
    if not diag:
        return "無 leg 診斷資料"
    touch = [d for d in diag if d.get("status") == "touch_signal"]
    if touch:
        blocked = [
            f"{d.get('entry_id')}({d.get('blocked_reason') or '未知'})"
            for d in touch
        ]
        return "觸軌但未開：" + "、".join(blocked)
    already = [d["entry_id"] for d in diag if d.get("status") == "already_open"]
    if already and len(already) == len(diag):
        return "全部 leg 已有持倉：" + "、".join(already)
    if already:
        return "部分 leg 已持倉；其餘未觸軌（見 leg 明細）"
    return "六個 leg 皆未出現首次觸軌穿越（見 leg 明細）"


def _format_touch_row(d: dict) -> str:
    touch = d.get("touch") or {}
    if d.get("side") == "SHORT":
        return (
            f"  {d.get('entry_id')}: {d.get('reason')} | "
            f"high={touch.get('high')} prev_high={touch.get('prev_high')} "
            f"top={touch.get('band')} prev_top={touch.get('prev_band')}"
        )
    return (
        f"  {d.get('entry_id')}: {d.get('reason')} | "
        f"low={touch.get('low')} prev_low={touch.get('prev_low')} "
        f"bott={touch.get('band')} prev_bott={touch.get('prev_band')}"
    )


def build_run_summary_lines(result: dict) -> List[str]:
    lines = [
        "",
        "=" * 72,
        f"FiBB 15m Run | {result.get('run_at')} | action={result.get('action')}",
        "=" * 72,
    ]

    cfg = result.get("config") or {}
    p = cfg.get("params") or {}
    lines.append(
        f"Config: {cfg.get('symbol')} {cfg.get('interval')} dry_run={cfg.get('dry_run')} "
        f"length={p.get('length')} tp%={float(p.get('tp_pct', 0)) * 100} "
        f"deferred_sl={p.get('use_deferred_channel_sl')} "
        f"reprice_tp_to_basis={cfg.get('reprice_tp_to_basis')}"
    )

    state_before = result.get("state_before") or {}
    if state_before:
        legs = state_before.get("open_legs") or {}
        leg_ids = list(legs.keys()) if isinstance(legs, dict) else legs
        lines.append(
            f"State(before): last_bar={state_before.get('last_bar_time')} "
            f"open_legs={leg_ids or '(none)'} trades={state_before.get('trade_count', 0)}"
        )

    for phase_key, label in (("account_before", "Account@before"), ("account_after", "Account@after")):
        acc = result.get(phase_key)
        if not acc:
            continue
        lines.append(
            f"{label}: mark={acc.get('mark_price')} | USDT {_fmt_balance(acc.get('balance_usdt') or {})}"
        )
        lines.append(f"  Exchange positions: {_fmt_positions(acc.get('positions') or [])}")

    bar = result.get("bar") or {}
    bar_sum = summarize_bar(bar)
    if bar_sum.get("headline_zh"):
        lines.append(f"本根總結: {bar_sum.get('headline_zh')}")

    ad = result.get("action_detail") or {}
    if ad.get("reason_zh"):
        lines.append(f"Detail: {ad.get('reason_zh')}")
    elif ad.get("no_entry_reason_zh") and bar_sum.get("entry_opened_count", 0) == 0:
        lines.append(f"未開倉: {ad.get('no_entry_reason_zh')}")

    ch = result.get("channels_at_bar") or {}
    if ch:
        lines.append(
            f"Channels: basis={ch.get('basis')} "
            f"T1={ch.get('top1')} T2={ch.get('top2')} T3={ch.get('top3')} "
            f"B1={ch.get('bott1')} B2={ch.get('bott2')} B3={ch.get('bott3')}"
        )
    if bar.get("close") is not None:
        lines.append(f"Bar: time={bar.get('bar_time')} close={bar.get('close')}")

    for ex in bar.get("exits") or []:
        tr = ex.get("trade") or {}
        lines.append(
            f"  EXIT {ex.get('entry_id')}: {ex.get('reason')} @ {ex.get('exit_price')} "
            f"net_pnl={tr.get('net_pnl')}"
        )

    for ent in bar.get("entries") or []:
        if ent.get("skipped"):
            lines.append(f"  ENTRY skip {ent.get('entry_id')}: {ent.get('reason')}")
            continue
        ex = ent.get("exchange") or {}
        tp_algo = ent.get("tp_algo_id") or ex.get("tp_algo_id")
        tp_price = ex.get("take_profit_price")
        lines.append(
            f"  ENTRY {ent.get('entry_id')}: {ent.get('side')} qty={ent.get('qty')} "
            f"status={ex.get('status', ex)}"
            + (f" tp={tp_price}" if tp_price else "")
            + (f" tp_algo_id={tp_algo}" if tp_algo else " tp_algo_id=(none)")
        )

    if bar.get("tp_reprice_note"):
        lines.append(f"  {bar.get('tp_reprice_note')}")

    for rp in bar.get("tp_reprices") or []:
        ex = rp.get("exchange") or {}
        lines.append(
            f"  TP_REPRICE {rp.get('entry_id')}: {rp.get('old_tp')} -> {rp.get('new_tp')} "
            f"status={ex.get('status', ex)} "
            f"old_algo={rp.get('old_tp_algo_id')} new_algo={ex.get('tp_algo_id')}"
        )

    for hold in bar.get("hold_diagnostics") or []:
        lines.append(
            f"  HOLD {hold.get('entry_id')}: {hold.get('hold_reason')} "
            f"TP={hold.get('take_profit')} SL={hold.get('stop_loss')}"
        )

    lines.append("Leg 進場診斷:")
    for d in bar.get("entry_diagnostics") or []:
        lines.append(_format_touch_row(d))

    state_after = result.get("state_after") or {}
    if state_after:
        legs = state_after.get("open_legs") or {}
        leg_ids = list(legs.keys()) if isinstance(legs, dict) else legs
        lines.append(
            f"State(after): last_bar={state_after.get('last_bar_time')} "
            f"open_legs={leg_ids or '(none)'} realized_pnl={state_after.get('realized_pnl')}"
        )

    if result.get("error"):
        lines.append(f"ERROR: {result['error']}")

    lines.append("=" * 72)
    return lines


def enrich_run_record(result: dict) -> dict:
    bar = result.get("bar")
    if bar and not bar.get("skipped"):
        headline, headline_zh = build_bar_headline(bar)
        bar["headline"] = headline
        bar["headline_zh"] = headline_zh

    result.setdefault("logged_at", utc_now_iso())
    result["bar_summary"] = summarize_bar(bar)
    result["action_detail"] = build_action_detail(result)
    result["summary_lines"] = build_run_summary_lines(result)
    return result


def format_run_log_block(result: dict) -> str:
    lines = result.get("summary_lines") or build_run_summary_lines(result)
    block = "\n".join(lines)
    include_json = os.getenv("FIBB_LOG_JSON", "1") == "1"
    if include_json:
        block += "\n\n--- JSON ---\n"
        block += json.dumps(result, ensure_ascii=False, indent=2)
    return block + "\n\n"


def print_run_output(result: dict, *, include_json: bool | None = None) -> None:
    if include_json is None:
        include_json = os.getenv("FIBB_PRINT_JSON", "0") == "1"
    for line in result.get("summary_lines") or build_run_summary_lines(result):
        print(line)
    if include_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
