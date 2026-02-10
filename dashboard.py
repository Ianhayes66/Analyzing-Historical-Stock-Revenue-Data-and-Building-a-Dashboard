"""
Performance Dashboard
----------------------
Real-time web dashboard built with Plotly Dash showing:
  - Portfolio equity curve
  - Open positions with P&L
  - Strategy signals heatmap
  - Trade history table
  - Key performance metrics cards

Run standalone or launched from main.py.
"""

import logging

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dash_table, dcc, html
from dash.dependencies import Input, Output

import config
import data_fetcher
import indicators
from backtester import run_backtest
from engine import TradingEngine
from risk_manager import RiskManager

logger = logging.getLogger(__name__)

# ── Global engine instance (shared with the running bot) ────────────────
_engine: TradingEngine = None  # type: ignore


def create_app(engine: TradingEngine = None) -> dash.Dash:
    """Build and return the Dash application."""
    global _engine
    _engine = engine or TradingEngine()

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        title="Trading Bot Dashboard",
    )

    app.layout = dbc.Container(
        fluid=True,
        children=[
            # ── Header ──────────────────────────────────────────────────
            dbc.Row(
                dbc.Col(
                    html.H1(
                        "Trading Bot Dashboard",
                        className="text-center my-3",
                        style={"color": "#00d4ff"},
                    )
                )
            ),

            # ── KPI Cards ───────────────────────────────────────────────
            dbc.Row(id="kpi-cards", className="mb-3"),

            # ── Charts Row ──────────────────────────────────────────────
            dbc.Row([
                dbc.Col(dcc.Graph(id="equity-chart"), md=8),
                dbc.Col(dcc.Graph(id="allocation-chart"), md=4),
            ], className="mb-3"),

            # ── Signals & Positions ─────────────────────────────────────
            dbc.Row([
                dbc.Col([
                    html.H4("Open Positions", style={"color": "#00d4ff"}),
                    dash_table.DataTable(
                        id="positions-table",
                        style_table={"overflowX": "auto"},
                        style_header={
                            "backgroundColor": "#303030",
                            "color": "#00d4ff",
                            "fontWeight": "bold",
                        },
                        style_cell={
                            "backgroundColor": "#1e1e1e",
                            "color": "white",
                            "textAlign": "center",
                        },
                    ),
                ], md=6),
                dbc.Col([
                    html.H4("Latest Signals", style={"color": "#00d4ff"}),
                    dash_table.DataTable(
                        id="signals-table",
                        style_table={"overflowX": "auto"},
                        style_header={
                            "backgroundColor": "#303030",
                            "color": "#00d4ff",
                            "fontWeight": "bold",
                        },
                        style_cell={
                            "backgroundColor": "#1e1e1e",
                            "color": "white",
                            "textAlign": "center",
                        },
                        style_data_conditional=[
                            {"if": {"filter_query": '{direction} = "BUY"'},
                             "color": "#00ff88"},
                            {"if": {"filter_query": '{direction} = "SELL"'},
                             "color": "#ff4444"},
                        ],
                    ),
                ], md=6),
            ], className="mb-3"),

            # ── Trade History ───────────────────────────────────────────
            dbc.Row(
                dbc.Col([
                    html.H4("Recent Trade History", style={"color": "#00d4ff"}),
                    dash_table.DataTable(
                        id="trades-table",
                        style_table={"overflowX": "auto"},
                        style_header={
                            "backgroundColor": "#303030",
                            "color": "#00d4ff",
                            "fontWeight": "bold",
                        },
                        style_cell={
                            "backgroundColor": "#1e1e1e",
                            "color": "white",
                            "textAlign": "center",
                        },
                        style_data_conditional=[
                            {"if": {"filter_query": "{pnl} > 0"},
                             "color": "#00ff88"},
                            {"if": {"filter_query": "{pnl} < 0"},
                             "color": "#ff4444"},
                        ],
                    ),
                ])
            ),

            # ── Auto-refresh ────────────────────────────────────────────
            dcc.Interval(id="refresh-interval", interval=30_000, n_intervals=0),
        ],
    )

    # ── Callbacks ───────────────────────────────────────────────────────
    @app.callback(
        [
            Output("kpi-cards", "children"),
            Output("equity-chart", "figure"),
            Output("allocation-chart", "figure"),
            Output("positions-table", "data"),
            Output("positions-table", "columns"),
            Output("signals-table", "data"),
            Output("signals-table", "columns"),
            Output("trades-table", "data"),
            Output("trades-table", "columns"),
        ],
        [Input("refresh-interval", "n_intervals")],
    )
    def update_dashboard(_n):
        status = _engine.get_status()
        portfolio = status.get("portfolio", {})

        # ── KPI Cards ──────────────────────────────────────────────
        cards = _build_kpi_cards(portfolio)

        # ── Equity Chart ───────────────────────────────────────────
        equity_fig = _build_equity_chart(status)

        # ── Allocation Chart ───────────────────────────────────────
        alloc_fig = _build_allocation_chart(status)

        # ── Positions Table ────────────────────────────────────────
        pos_data, pos_cols = _build_positions_table(status)

        # ── Signals Table ──────────────────────────────────────────
        sig_data, sig_cols = _build_signals_table(status)

        # ── Trades Table ───────────────────────────────────────────
        trade_data, trade_cols = _build_trades_table(status)

        return (
            cards, equity_fig, alloc_fig,
            pos_data, pos_cols,
            sig_data, sig_cols,
            trade_data, trade_cols,
        )

    return app


# ── Component Builders ──────────────────────────────────────────────────

def _build_kpi_cards(portfolio: dict) -> list:
    pv = portfolio.get("portfolio_value", 0)
    ret = portfolio.get("return_pct", 0)
    wr = portfolio.get("win_rate", 0)
    trades = portfolio.get("total_trades", 0)
    pf = portfolio.get("profit_factor", 0)
    dd = portfolio.get("daily_pnl", 0)

    kpis = [
        ("Portfolio Value", f"${pv:,.2f}", "#00d4ff"),
        ("Total Return", f"{ret:.2%}", "#00ff88" if ret >= 0 else "#ff4444"),
        ("Win Rate", f"{wr:.0%}", "#00ff88" if wr >= 0.5 else "#ffaa00"),
        ("Total Trades", str(trades), "#00d4ff"),
        ("Profit Factor", f"{pf:.2f}", "#00ff88" if pf >= 1 else "#ff4444"),
        ("Daily P&L", f"${dd:,.2f}", "#00ff88" if dd >= 0 else "#ff4444"),
    ]

    return [
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.P(label, className="text-muted mb-1",
                           style={"fontSize": "0.85rem"}),
                    html.H4(value, style={"color": color}),
                ]),
                style={"backgroundColor": "#1e1e1e", "border": "1px solid #333"},
            ),
            md=2,
        )
        for label, value, color in kpis
    ]


def _build_equity_chart(status: dict) -> go.Figure:
    fig = go.Figure()
    # Use trade history to build a simple equity line
    trades = status.get("trade_history", [])
    if trades:
        cumulative_pnl = []
        running = config.INITIAL_CAPITAL
        for t in trades:
            running += t.get("pnl", 0)
            cumulative_pnl.append(running)
        fig.add_trace(go.Scatter(
            y=cumulative_pnl,
            mode="lines",
            name="Equity",
            line=dict(color="#00d4ff", width=2),
            fill="tozeroy",
            fillcolor="rgba(0, 212, 255, 0.1)",
        ))
    else:
        fig.add_trace(go.Scatter(
            y=[config.INITIAL_CAPITAL],
            mode="lines",
            name="Equity",
            line=dict(color="#00d4ff"),
        ))

    fig.update_layout(
        title="Equity Curve",
        template="plotly_dark",
        paper_bgcolor="#1e1e1e",
        plot_bgcolor="#1e1e1e",
        margin=dict(l=40, r=20, t=40, b=30),
    )
    return fig


def _build_allocation_chart(status: dict) -> go.Figure:
    positions = status.get("positions", {})
    if positions:
        labels = list(positions.keys())
        values = [abs(p.get("pnl", 0) + p["entry_price"] * p["shares"])
                  for p in positions.values()]
    else:
        labels = ["Cash"]
        values = [status.get("portfolio", {}).get("cash", config.INITIAL_CAPITAL)]

    # Always add cash
    cash = status.get("portfolio", {}).get("cash", 0)
    if positions:
        labels.append("Cash")
        values.append(cash)

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.5,
        marker=dict(colors=["#00d4ff", "#00ff88", "#ffaa00", "#ff4444",
                            "#aa44ff", "#44ffaa", "#ff8800", "#888888"]),
    ))
    fig.update_layout(
        title="Allocation",
        template="plotly_dark",
        paper_bgcolor="#1e1e1e",
        plot_bgcolor="#1e1e1e",
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=True,
        legend=dict(font=dict(size=10)),
    )
    return fig


def _build_positions_table(status: dict):
    positions = status.get("positions", {})
    data = []
    for ticker, p in positions.items():
        data.append({
            "ticker": ticker,
            "direction": p["direction"],
            "shares": p["shares"],
            "entry": f"${p['entry_price']:.2f}",
            "current": f"${p['current_price']:.2f}",
            "pnl": f"${p['pnl']:.2f}",
            "pnl_pct": f"{p['pnl_pct']:.1%}",
            "stop": f"${p['stop_loss']:.2f}",
        })
    columns = [{"name": c, "id": c} for c in
               ["ticker", "direction", "shares", "entry", "current",
                "pnl", "pnl_pct", "stop"]]
    return data, columns


def _build_signals_table(status: dict):
    signals = status.get("latest_signals", {})
    data = []
    for ticker, sigs in signals.items():
        for s in sigs:
            data.append({
                "ticker": ticker,
                "strategy": s["strategy"],
                "direction": s["direction"],
                "confidence": f"{s['confidence']:.0%}",
                "reason": s["reason"][:80],
            })
    columns = [{"name": c, "id": c} for c in
               ["ticker", "strategy", "direction", "confidence", "reason"]]
    return data, columns


def _build_trades_table(status: dict):
    trades = status.get("trade_history", [])
    data = []
    for t in reversed(trades[-20:]):
        data.append({
            "ticker": t["ticker"],
            "direction": t["direction"],
            "entry": f"${t['entry_price']:.2f}",
            "exit": f"${t['exit_price']:.2f}",
            "pnl": round(t["pnl"], 2),
            "pnl_pct": f"{t['pnl_pct']:.1%}",
            "reason": t.get("reason", ""),
        })
    columns = [{"name": c, "id": c} for c in
               ["ticker", "direction", "entry", "exit", "pnl", "pnl_pct", "reason"]]
    return data, columns


def run_dashboard(engine: TradingEngine = None, debug: bool = config.DASHBOARD_DEBUG):
    """Launch the dashboard server."""
    app = create_app(engine)
    logger.info(
        f"Dashboard running at http://{config.DASHBOARD_HOST}:{config.DASHBOARD_PORT}"
    )
    app.run(
        host=config.DASHBOARD_HOST,
        port=config.DASHBOARD_PORT,
        debug=debug,
    )
