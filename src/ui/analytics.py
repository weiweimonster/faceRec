"""
Analytics Dashboard for Model Performance Metrics.

Provides visualizations and export functionality for comparing ranking model performance.
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime
from typing import Dict, Any

from src.ui.state import get_db
from src.evaluation.offline_metrics import OfflineEvaluator


def render_analytics_page():
    """Render the main analytics dashboard."""
    st.header("Model Performance Dashboard")

    db = get_db()

    # Get overall metrics
    overall = db.get_overall_ctr()
    model_summary = db.get_model_comparison_summary()

    # Top-level KPI cards
    st.subheader("Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Impressions", f"{overall['impressions']:,}")
    with col2:
        st.metric("Total Clicks", f"{overall['clicks']:,}")
    with col3:
        st.metric("Overall CTR", f"{overall['ctr']}%")
    with col4:
        st.metric("Sessions", f"{overall['unique_sessions']:,}")

    # Show date range if available
    date_range = model_summary.get("date_range", {})
    if date_range.get("start") and date_range.get("end"):
        st.caption(f"Data from {date_range['start'][:10]} to {date_range['end'][:10]}")

    st.divider()

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "Model Comparison",
        "CTR Trends",
        "Position Bias",
        "Export"
    ])

    with tab1:
        render_model_comparison(db, model_summary)

    with tab2:
        render_ctr_trends(db)

    with tab3:
        render_position_bias(db)

    with tab4:
        render_export(db, model_summary)


def render_model_comparison(db, model_summary: Dict[str, Any]):
    """Render the model comparison table."""
    st.subheader("Model Comparison")

    models = model_summary.get("models", {})

    if not models:
        st.info("No model data available yet. Start using the search feature to collect data.")
        return

    # Headline metric if lift is available
    ctr_lift = model_summary.get("ctr_lift")
    if ctr_lift is not None:
        if ctr_lift > 0:
            st.success(f"XGBoost shows **+{ctr_lift}%** CTR improvement over Heuristic baseline")
        elif ctr_lift < 0:
            st.warning(f"XGBoost shows **{ctr_lift}%** CTR compared to Heuristic baseline")

    # Build comparison table
    table_data = []
    for model_name, stats in models.items():
        table_data.append({
            "Model": model_name.title(),
            "Sessions": stats.get("sessions", 0),
            "Impressions": stats.get("impressions", 0),
            "Clicks": stats.get("clicks", 0),
            "CTR (%)": stats.get("ctr", 0),
            "Avg Click Position": stats.get("avg_click_position", "-")
        })

    if table_data:
        df = pd.DataFrame(table_data)
        st.dataframe(df, width='stretch', hide_index=True)

    # Offline evaluation metrics
    st.subheader("Offline Ranking Metrics")
    st.caption("Computed from click data as ground truth")

    evaluator = OfflineEvaluator(db)
    offline_metrics = evaluator.get_comparison_summary(k=5)

    offline_models = offline_metrics.get("models", {})
    if offline_models:
        offline_data = []
        for model_name, metrics in offline_models.items():
            offline_data.append({
                "Model": model_name.title(),
                "NDCG@5": metrics.get("ndcg@5", 0),
                "Precision@5": metrics.get("precision@5", 0),
                "Recall@5": metrics.get("recall@5", 0),
                "MRR": metrics.get("mrr", 0),
                "MAP": metrics.get("map", 0),
                "Sessions": metrics.get("sessions_evaluated", 0)
            })

        if offline_data:
            df_offline = pd.DataFrame(offline_data)
            st.dataframe(df_offline, width='stretch', hide_index=True)

        # Show lifts
        lifts = offline_metrics.get("lifts", {})
        if lifts:
            st.markdown("**Lift vs Heuristic:**")
            lift_cols = st.columns(len(lifts))
            for i, (metric, lift_val) in enumerate(lifts.items()):
                with lift_cols[i]:
                    label = metric.replace("_lift", "").upper()
                    delta_color = "normal" if lift_val >= 0 else "inverse"
                    st.metric(label, f"{lift_val:+.1f}%", delta_color=delta_color)
    else:
        st.info("Not enough click data for offline evaluation. Need sessions with at least one click.")


def render_ctr_trends(db):
    """Render CTR trends over time."""
    st.subheader("CTR Trends Over Time")

    ctr_data = db.get_ctr_by_date()

    if not ctr_data:
        st.info("No trend data available yet. Use the search feature over multiple days.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(ctr_data)

    if df.empty:
        st.info("No trend data available.")
        return

    # Pivot for line chart
    try:
        df_pivot = df.pivot(index='date', columns='model', values='ctr').reset_index()
        df_pivot = df_pivot.fillna(0)

        st.line_chart(df_pivot, x='date', y=[c for c in df_pivot.columns if c != 'date'])
    except Exception as e:
        st.warning(f"Could not create trend chart: {e}")
        st.dataframe(df)

    # Show raw data in expander
    with st.expander("View Raw Data"):
        st.dataframe(df, width='stretch', hide_index=True)


def render_position_bias(db):
    """Render position bias analysis."""
    st.subheader("Position Bias Analysis")
    st.caption("CTR decay by result position - shows how click-through rate decreases with lower positions")

    position_data = db.get_ctr_by_position(limit=20)

    if not position_data:
        st.info("No position data available yet.")
        return

    df = pd.DataFrame(position_data)

    if df.empty:
        st.info("No position data available.")
        return

    # Bar chart
    st.bar_chart(df, x='position', y='ctr')

    # Summary stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Position 1 CTR", f"{position_data[0]['ctr'] if position_data else 0}%")
    with col2:
        if len(position_data) > 4:
            st.metric("Position 5 CTR", f"{position_data[4]['ctr']}%")

    # Raw data
    with st.expander("View Raw Data"):
        st.dataframe(df, width='stretch', hide_index=True)


def render_export(db, model_summary: Dict[str, Any]):
    """Render export options."""
    st.subheader("Export Data")

    # Build comprehensive export data
    overall = db.get_overall_ctr()
    position_data = db.get_ctr_by_position()

    evaluator = OfflineEvaluator(db)
    offline_metrics = evaluator.get_comparison_summary(k=5)

    export_data = {
        "generated_at": datetime.now().isoformat(),
        "overall_metrics": overall,
        "model_comparison": model_summary,
        "offline_metrics": offline_metrics,
        "position_bias": position_data
    }

    col1, col2 = st.columns(2)

    with col1:
        # JSON Export
        json_str = json.dumps(export_data, indent=2, default=str)
        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name=f"ranking_metrics_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )

    with col2:
        # CSV Export (model comparison)
        models = model_summary.get("models", {})
        if models:
            csv_data = []
            for model_name, stats in models.items():
                csv_data.append({
                    "model": model_name,
                    **stats
                })
            df = pd.DataFrame(csv_data)
            csv_str = df.to_csv(index=False)

            st.download_button(
                label="Download CSV",
                data=csv_str,
                file_name=f"model_comparison_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

    # Copy-paste summary
    st.subheader("Portfolio Summary")
    st.caption("Copy this text for your portfolio or README")

    models = model_summary.get("models", {})
    xgb = models.get("xgboost", {})
    heur = models.get("heuristic", {})

    ctr_lift = model_summary.get("ctr_lift", 0)
    ndcg_lift = offline_metrics.get("lifts", {}).get("ndcg@5_lift", 0)

    summary_text = f"""## Ranking Model Performance

I implemented a learning-to-rank system using XGBoost trained on user click data.
Compared to a hand-tuned heuristic baseline, the ML model achieved:

- **{ctr_lift:+.1f}% CTR improvement** ({xgb.get('ctr', 0)}% vs {heur.get('ctr', 0)}%)
- **{ndcg_lift:+.1f}% NDCG@5 improvement**
- Based on {overall['unique_sessions']} search sessions and {overall['impressions']:,} photo impressions

### Key Metrics
| Metric | XGBoost | Heuristic | Lift |
|--------|---------|-----------|------|
| CTR | {xgb.get('ctr', 0)}% | {heur.get('ctr', 0)}% | {ctr_lift:+.1f}% |
| Avg Click Position | {xgb.get('avg_click_position', '-')} | {heur.get('avg_click_position', '-')} | {model_summary.get('position_lift', 0):+.1f}% |
| Sessions | {xgb.get('sessions', 0)} | {heur.get('sessions', 0)} | - |
"""

    st.code(summary_text, language="markdown")

    # Preview rendered
    with st.expander("Preview Rendered"):
        st.markdown(summary_text)
