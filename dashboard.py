"""
Polymarket Predictor Dashboard
--------------------------------
Real-time web dashboard showing:
  - Bankroll status and P&L
  - Top opportunities with divergence signals
  - Sentiment vs. price visualizations
  - Open bets tracker
  - Echo chamber alerts
  - Narrative momentum indicators

Dark-themed, auto-refreshing, designed for monitoring your $50 empire.
"""

import logging
from typing import Optional

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dash_table, dcc, html
from dash.dependencies import Input, Output

import config
from polymarket_predictor.bankroll_manager import BankrollManager
from polymarket_predictor.signal_generator import Opportunity, SignalGenerator

logger = logging.getLogger(__name__)

_generator: SignalGenerator = None  # type: ignore

COLORS = {
    "bg": "#0a0a1a",
    "card": "#12122a",
    "border": "#1e1e3a",
    "accent": "#00d4ff",
    "green": "#00ff88",
    "red": "#ff3366",
    "yellow": "#ffaa00",
    "purple": "#aa66ff",
    "text": "#e0e0e0",
    "muted": "#666688",
}


def create_app(generator: SignalGenerator = None) -> dash.Dash:
    global _generator
    _generator = generator or SignalGenerator()

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.CYBORG],
        title="Polymarket Predictor",
    )

    app.layout = dbc.Container(
        fluid=True,
        style={"backgroundColor": COLORS["bg"], "minHeight": "100vh", "padding": "20px"},
        children=[
            # Header
            dbc.Row(dbc.Col(html.Div([
                html.H1("POLYMARKET PREDICTOR",
                         style={"color": COLORS["accent"], "letterSpacing": "3px",
                                "fontWeight": "300", "marginBottom": "5px"}),
                html.P("Sentiment-Driven Prediction Market Intelligence",
                       style={"color": COLORS["muted"], "fontSize": "0.9rem"}),
            ], className="text-center my-3"))),

            # KPI Cards
            dbc.Row(id="kpi-row", className="mb-4"),

            # Main content
            dbc.Row([
                # Left: Opportunities
                dbc.Col([
                    _card("Top Opportunities", [
                        dash_table.DataTable(
                            id="opps-table",
                            style_table={"overflowX": "auto", "maxHeight": "500px",
                                         "overflowY": "auto"},
                            style_header={"backgroundColor": COLORS["card"],
                                          "color": COLORS["accent"],
                                          "fontWeight": "bold", "border": "none"},
                            style_cell={"backgroundColor": COLORS["bg"],
                                        "color": COLORS["text"], "textAlign": "center",
                                        "border": f"1px solid {COLORS['border']}",
                                        "fontSize": "0.85rem", "padding": "8px"},
                            style_data_conditional=[
                                {"if": {"filter_query": '{direction} = "BUY_YES"'},
                                 "color": COLORS["green"]},
                                {"if": {"filter_query": '{direction} = "BUY_NO"'},
                                 "color": COLORS["red"]},
                                {"if": {"state": "selected"},
                                 "backgroundColor": COLORS["border"]},
                            ],
                        ),
                    ]),
                ], md=8),

                # Right: Bankroll + Bets
                dbc.Col([
                    _card("Open Bets", [
                        dash_table.DataTable(
                            id="bets-table",
                            style_table={"overflowX": "auto"},
                            style_header={"backgroundColor": COLORS["card"],
                                          "color": COLORS["accent"],
                                          "fontWeight": "bold", "border": "none"},
                            style_cell={"backgroundColor": COLORS["bg"],
                                        "color": COLORS["text"], "textAlign": "center",
                                        "border": f"1px solid {COLORS['border']}",
                                        "fontSize": "0.85rem"},
                        ),
                    ]),
                    html.Br(),
                    _card("Sentiment vs Price", [
                        dcc.Graph(id="divergence-chart",
                                  config={"displayModeBar": False}),
                    ]),
                ], md=4),
            ], className="mb-4"),

            # Bottom row: Signal details + Echo chamber alerts
            dbc.Row([
                dbc.Col([
                    _card("Signal Breakdown", [
                        dcc.Graph(id="signal-radar",
                                  config={"displayModeBar": False}),
                    ]),
                ], md=6),
                dbc.Col([
                    _card("Echo Chamber Alerts", [
                        html.Div(id="echo-alerts"),
                    ]),
                ], md=6),
            ]),

            # Trade history
            dbc.Row(dbc.Col(
                _card("Trade History", [
                    dash_table.DataTable(
                        id="history-table",
                        style_table={"overflowX": "auto"},
                        style_header={"backgroundColor": COLORS["card"],
                                      "color": COLORS["accent"],
                                      "fontWeight": "bold", "border": "none"},
                        style_cell={"backgroundColor": COLORS["bg"],
                                    "color": COLORS["text"], "textAlign": "center",
                                    "border": f"1px solid {COLORS['border']}",
                                    "fontSize": "0.85rem"},
                        style_data_conditional=[
                            {"if": {"filter_query": "{pnl} > 0"}, "color": COLORS["green"]},
                            {"if": {"filter_query": "{pnl} < 0"}, "color": COLORS["red"]},
                        ],
                    ),
                ]),
            ), className="mb-4"),

            dcc.Interval(id="refresh", interval=config.REFRESH_INTERVAL_SEC * 1000,
                         n_intervals=0),
        ],
    )

    # ── Callbacks ───────────────────────────────────────────────────
    @app.callback(
        [
            Output("kpi-row", "children"),
            Output("opps-table", "data"),
            Output("opps-table", "columns"),
            Output("bets-table", "data"),
            Output("bets-table", "columns"),
            Output("divergence-chart", "figure"),
            Output("signal-radar", "figure"),
            Output("echo-alerts", "children"),
            Output("history-table", "data"),
            Output("history-table", "columns"),
        ],
        [Input("refresh", "n_intervals")],
    )
    def update(_n):
        opps = _generator.get_top_opportunities(20)
        bankroll = _generator.bankroll.get_summary()

        kpis = _build_kpis(bankroll, len(opps))
        opp_data, opp_cols = _build_opps_table(opps)
        bet_data, bet_cols = _build_bets_table(_generator.bankroll)
        div_fig = _build_divergence_chart(opps)
        radar_fig = _build_radar(opps)
        echo_alerts = _build_echo_alerts(opps)
        hist_data, hist_cols = _build_history(_generator.bankroll)

        return (kpis, opp_data, opp_cols, bet_data, bet_cols,
                div_fig, radar_fig, echo_alerts, hist_data, hist_cols)

    return app


# ── Component Helpers ───────────────────────────────────────────────

def _card(title: str, children: list) -> dbc.Card:
    return dbc.Card(
        [
            dbc.CardHeader(title, style={
                "backgroundColor": COLORS["card"],
                "color": COLORS["accent"],
                "borderBottom": f"1px solid {COLORS['border']}",
                "fontWeight": "600",
            }),
            dbc.CardBody(children, style={"backgroundColor": COLORS["bg"]}),
        ],
        style={"border": f"1px solid {COLORS['border']}",
               "borderRadius": "8px"},
    )


def _kpi_card(label: str, value: str, color: str) -> dbc.Col:
    return dbc.Col(
        dbc.Card(
            dbc.CardBody([
                html.P(label, className="mb-1",
                       style={"color": COLORS["muted"], "fontSize": "0.75rem",
                              "textTransform": "uppercase", "letterSpacing": "1px"}),
                html.H3(value, style={"color": color, "fontWeight": "300"}),
            ]),
            style={"backgroundColor": COLORS["card"],
                   "border": f"1px solid {COLORS['border']}",
                   "borderRadius": "8px"},
        ),
        md=2, className="mb-2",
    )


def _build_kpis(bankroll: dict, n_opps: int) -> list:
    pnl = bankroll["total_pnl"]
    roi = bankroll["roi_pct"]
    return [
        _kpi_card("Bankroll", f"${bankroll['bankroll']:.2f}",
                  COLORS["accent"]),
        _kpi_card("P&L", f"${pnl:+.2f}",
                  COLORS["green"] if pnl >= 0 else COLORS["red"]),
        _kpi_card("ROI", f"{roi:+.1f}%",
                  COLORS["green"] if roi >= 0 else COLORS["red"]),
        _kpi_card("Win Rate", f"{bankroll['win_rate']:.0f}%",
                  COLORS["green"] if bankroll["win_rate"] >= 50 else COLORS["yellow"]),
        _kpi_card("Open Bets", str(bankroll["open_bets"]),
                  COLORS["purple"]),
        _kpi_card("Opportunities", str(n_opps),
                  COLORS["accent"]),
    ]


def _build_opps_table(opps):
    data = []
    for o in opps[:20]:
        data.append({
            "score": f"{o.composite_score:.2f}",
            "direction": o.signal.direction,
            "edge": f"{o.signal.edge_estimate:+.1%}",
            "confidence": f"{o.signal.confidence:.0%}",
            "bet": f"${o.recommended_bet:.2f}" if o.recommended_bet > 0 else "-",
            "type": o.signal.signal_type,
            "price": f"{o.signal.market_price:.0%}",
            "question": o.market["question"][:50],
        })
    cols = [{"name": c.title(), "id": c} for c in
            ["score", "direction", "edge", "confidence", "bet", "type", "price", "question"]]
    return data, cols


def _build_bets_table(bankroll_mgr: BankrollManager):
    data = []
    for b in bankroll_mgr.state.open_bets:
        data.append({
            "direction": b.direction,
            "amount": f"${b.amount:.2f}",
            "entry": f"{b.entry_price:.2f}",
            "edge": f"{b.estimated_edge:+.1%}",
            "type": b.signal_type,
            "question": b.question[:40],
        })
    cols = [{"name": c.title(), "id": c} for c in
            ["direction", "amount", "entry", "edge", "type", "question"]]
    return data, cols


def _build_divergence_chart(opps) -> go.Figure:
    fig = go.Figure()
    if opps:
        top = opps[:10]
        questions = [o.market["question"][:25] for o in top]
        market_prices = [o.signal.market_price for o in top]
        sentiment_prices = [o.signal.sentiment_price for o in top]

        fig.add_trace(go.Bar(
            name="Market Price", x=questions, y=market_prices,
            marker_color=COLORS["accent"], opacity=0.7,
        ))
        fig.add_trace(go.Bar(
            name="Sentiment Est.", x=questions, y=sentiment_prices,
            marker_color=COLORS["purple"], opacity=0.7,
        ))

    fig.update_layout(
        barmode="group",
        template="plotly_dark",
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        margin=dict(l=30, r=10, t=10, b=60),
        height=280,
        legend=dict(orientation="h", y=1.1),
        font=dict(size=10),
    )
    return fig


def _build_radar(opps) -> go.Figure:
    fig = go.Figure()
    if opps:
        o = opps[0]
        categories = ["Divergence", "Velocity", "Echo Inv.", "Bayesian", "Volume"]
        values = [
            min(abs(o.signal.edge_estimate) * 5, 1),
            min(abs(o.narrative.momentum_score), 1),
            1 - o.echo.uniformity_score,
            o.signal.confidence,
            min(o.sentiment.sample_size / 50, 1),
        ]
        values.append(values[0])  # Close the polygon
        categories.append(categories[0])

        fig.add_trace(go.Scatterpolar(
            r=values, theta=categories, fill="toself",
            fillcolor=f"rgba(0, 212, 255, 0.15)",
            line=dict(color=COLORS["accent"]),
            name=o.market["question"][:30],
        ))

    fig.update_layout(
        polar=dict(
            bgcolor=COLORS["bg"],
            radialaxis=dict(visible=True, range=[0, 1],
                            gridcolor=COLORS["border"]),
            angularaxis=dict(gridcolor=COLORS["border"]),
        ),
        template="plotly_dark",
        paper_bgcolor=COLORS["bg"],
        margin=dict(l=40, r=40, t=20, b=20),
        height=300,
        showlegend=False,
        font=dict(size=10),
    )
    return fig


def _build_echo_alerts(opps) -> list:
    alerts = []
    for o in opps[:10]:
        if o.echo.is_echo_chamber:
            alerts.append(
                dbc.Alert(
                    [
                        html.Strong(f"ECHO CHAMBER: "),
                        html.Span(
                            f"{o.market['question'][:60]} — "
                            f"{o.echo.dominant_pct:.0%} agree on {o.echo.dominant_direction}. "
                            f"{'CONTRARIAN signal!' if o.echo.contrarian_signal else 'Signal dampened.'}"
                        ),
                    ],
                    color="warning" if not o.echo.contrarian_signal else "info",
                    className="mb-2",
                    style={"fontSize": "0.85rem"},
                )
            )

    if not alerts:
        alerts.append(html.P(
            "No echo chamber alerts. Sentiment diversity is healthy.",
            style={"color": COLORS["muted"], "fontStyle": "italic"},
        ))

    return alerts


def _build_history(bankroll_mgr: BankrollManager):
    data = []
    for b in reversed(bankroll_mgr.state.closed_bets[-20:]):
        data.append({
            "direction": b.direction,
            "amount": f"${b.amount:.2f}",
            "entry": f"{b.entry_price:.2f}",
            "exit": f"{b.exit_price:.2f}",
            "pnl": round(b.pnl, 2),
            "status": b.status,
            "question": b.question[:40],
        })
    cols = [{"name": c.title(), "id": c} for c in
            ["direction", "amount", "entry", "exit", "pnl", "status", "question"]]
    return data, cols


def run_dashboard(generator: SignalGenerator = None, debug: bool = False):
    app = create_app(generator)
    logger.info(f"Dashboard: http://{config.DASHBOARD_HOST}:{config.DASHBOARD_PORT}")
    app.run(
        host=config.DASHBOARD_HOST,
        port=config.DASHBOARD_PORT,
        debug=debug,
    )
