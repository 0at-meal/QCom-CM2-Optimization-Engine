import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)

from src.pricing import PricingEngine
from src.features import load_data

st.set_page_config(page_title="QCom Pricing Optimization", layout="wide")

@st.cache_resource
def load_engine():
    model_path = os.path.join(parent_dir, 'models', 'conversion_model.pkl')
    return PricingEngine(model_path=model_path)

@st.cache_data
def run_evaluation_cached():
    df = load_data(os.path.join(parent_dir, 'data', 'qcom_pune_dataset.csv'))

    test_df = df.sample(500, random_state=42)
    engine = load_engine()
    
    results = []
    
    for _, row in test_df.iterrows():
        req = row.to_dict()
        try:
            opt = engine.optimize_fee(req)
            
            actual_fee = row['delivery_fee_charged']
            hist_features = engine._prepare_features(req, [actual_fee])
            hist_prob = engine.model.predict_proba(hist_features)[:, 1][0]
            
            margin = row['basket_margin']
            cost = row['delivery_cost_potential'] if not pd.isna(row['delivery_cost_potential']) else 60
            
            hist_cm2 = hist_prob * (margin + actual_fee - cost)
            
            results.append({
                'order_id': row['order_id'],
                'hist_fee': actual_fee,
                'opt_fee': opt['optimal_fee'],
                'hist_cm2': hist_cm2,
                'opt_cm2': opt['expected_cm2'],
                'hist_prob': hist_prob,
                'opt_prob': opt['expected_conversion'],
                'traffic_level': row['traffic_level'],
                'basket_value': row['basket_value']
            })
        except Exception as e:
            continue
            
    return pd.DataFrame(results)

st.title("QCom Dynamic Delivery Fee Optimization Dashboard")

st.sidebar.header("Configuration")
reload_data = st.sidebar.button("Reload Evaluation Data")

if 'eval_data' not in st.session_state or reload_data:
    with st.spinner("Running Evaluation (500 samples)..."):
        st.session_state.eval_data = run_evaluation_cached()

df_res = st.session_state.eval_data

st.subheader("Key Performance Indicators (Simulated Uplift)")
col1, col2, col3, col4 = st.columns(4)

total_hist_cm2 = df_res['hist_cm2'].sum()
total_opt_cm2 = df_res['opt_cm2'].sum()
uplift_pct = (total_opt_cm2 - total_hist_cm2) / total_hist_cm2 * 100

avg_hist_conv = df_res['hist_prob'].mean() * 100
avg_opt_conv = df_res['opt_prob'].mean() * 100
conv_impact = avg_opt_conv - avg_hist_conv

avg_hist_fee = df_res['hist_fee'].mean()
avg_opt_fee = df_res['opt_fee'].mean()

with col1:
    st.metric("Total Expected CM2", f"₹{total_opt_cm2:,.0f}", f"{uplift_pct:.1f}%")
with col2:
    st.metric("Avg Conversion Rate", f"{avg_opt_conv:.1f}%", f"{conv_impact:.2f} pp")
with col3:
    st.metric("Avg Delivery Fee", f"₹{avg_opt_fee:.1f}", f"₹{avg_opt_fee - avg_hist_fee:.1f}")
with col4:
    st.metric("Sample Size", f"{len(df_res)}")

st.markdown("---")
st.subheader("Analysis")

c1, c2 = st.columns(2)

with c1:
    st.markdown("### Fee Distribution Comparison")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=df_res['hist_fee'], name='Historical Fee', opacity=0.75))
    fig_hist.add_trace(go.Histogram(x=df_res['opt_fee'], name='Optimized Fee', opacity=0.75))
    fig_hist.update_layout(barmode='overlay', xaxis_title="Delivery Fee (₹)", yaxis_title="Count")
    st.plotly_chart(fig_hist, use_container_width=True)

with c2:
    st.markdown("### CM2 vs Conversion Probability")
    fig_scatter = px.scatter(
        df_res, x='opt_prob', y='opt_cm2', color='traffic_level',
        hover_data=['basket_value', 'hist_fee', 'opt_fee'],
        title="Optimized Outcome Space"
    )
    fig_scatter.update_layout(xaxis_title="Predicted Conversion Prob", yaxis_title="Expected CM2 (₹)")
    st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")
st.subheader("Price Simulator")
st.markdown("Test the pricing engine with custom inputs.")

with st.form("sim_form"):
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        b_val = st.number_input("Basket Value", 100, 5000, 500)
        b_margin = st.number_input("Basket Margin", 0, 1000, 100)
        b_weight = st.number_input("Weight (kg)", 0.1, 50.0, 2.0)
    with sc2:
        dist = st.number_input("Distance (km)", 0.1, 20.0, 3.0)
        est_time = st.number_input("Est. Time (min)", 5, 120, 25)
        traffic = st.selectbox("Traffic Level", ['low', 'medium', 'high'])
    with sc3:
        sens = st.slider("Price Sensitivity", 0.0, 1.0, 0.5)
        base_prob = st.slider("Base Conversion Prob", 0.0, 1.0, 0.8)
        cost_est = st.number_input("Est. Delivery Cost", 0, 200, 40)
        
    submitted = st.form_submit_button("Optimize Fee")
    
    if submitted:
        req = {
            'basket_value': b_val, 'basket_margin': b_margin, 'basket_weight_kg': b_weight,
            'num_items': 5, # Placeholder
            'distance_km': dist, 'estimated_delivery_time_min': est_time,
            'hour_of_day': 18, 'day_of_week': 0, 'traffic_level': traffic,
            'price_sensitivity_score': sens, 'delivery_cost_potential': cost_est,
            'conversion_prob_stage1': base_prob
        }
        
        engine = load_engine()
        res = engine.optimize_fee(req)
        
        st.success(f"Recommended Fee: ₹{res['optimal_fee']}")

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Exp. Conversion", f"{res['expected_conversion']*100:.1f}%")
        metric_col2.metric("Exp. CM2", f"₹{res['expected_cm2']:.2f}")
        metric_col3.metric("Baseline Prob", f"{res['baseline_conversion']*100:.1f}%")

        if 'candidates' in res and res['candidates']:
            cand_df = pd.DataFrame(res['candidates'])
            st.dataframe(cand_df.style.highlight_max(subset=['expected_cm2'], color='lightgreen'))
