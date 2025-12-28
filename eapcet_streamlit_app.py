import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Import the model (assumes eapcet_rank_model.py is in same directory)
try:
    from eapcet_rank_model import EAPCETRankPredictor
except ImportError:
    st.error("❌ Cannot import EAPCETRankPredictor. Ensure eapcet_rank_model.py is in the same directory.")
    st.stop()

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="EAPCET Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS FOR CLEAN UI
# ============================================================================

st.markdown("""
<style>
    .main { background-color: #ffffff; }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #2c3e50;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 14px;
        color: #7f8c8d;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-subtitle { font-size: 14px; color: #95a5a6; margin-top: 5px; }
    .perf-below-avg { color: #e74c3c; font-weight: 600; }
    .perf-average { color: #f39c12; font-weight: 600; }
    .perf-good { color: #3498db; font-weight: 600; }
    .perf-excellent { color: #27ae60; font-weight: 600; }
    .section-header {
        font-size: 18px;
        font-weight: 600;
        color: #2c3e50;
        margin: 30px 0 15px 0;
        border-bottom: 2px solid #ecf0f1;
        padding-bottom: 10px;
    }
    .disclaimer {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
        margin-top: 30px;
        font-size: 14px;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL
# ============================================================================

@st.cache_resource
def load_predictor():
    try:
        predictor = EAPCETRankPredictor()
        predictor.load_model("eapcet_rank_model.pkl")
        return predictor
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_performance_color(label: str) -> str:
    color_map = {
        "Below Average": "perf-below-avg",
        "Average": "perf-average",
        "Good": "perf-good",
        "Excellent": "perf-excellent",
        "Not Qualified": "perf-below-avg"
    }
    return color_map.get(label, "perf-average")

def render_metric_card(label: str, value: str, subtitle: str = ""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {f'<div class="metric-subtitle">{subtitle}</div>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.title("📊 EAPCET Performance Analytics Dashboard")
    st.markdown("**Internal Exam Simulation & Analytics System**")
    st.caption("TS EAPCET 2025: 207,190 appeared | 151,779 qualified (73.26%)")
    st.divider()
    
    predictor = load_predictor()
    
    with st.sidebar:
        st.header("📝 Enter Exam Details")
        score = st.number_input(
            "Total Score (0-160)",
            min_value=0.0,
            max_value=160.0,
            value=40.0,
            step=0.01,
            help="Enter your total EAPCET score"
        )
        exam_year = st.selectbox("Exam Year", options=[2025, 2024, 2023, 2022, 2021], index=0)
        attempted = st.number_input("Questions Attempted", min_value=0, max_value=160, value=160, step=1)
        correct = st.number_input("Correct Answers", min_value=0, max_value=attempted, value=int(score), step=1)
        
        st.divider()
        analyze_button = st.button("🔍 Analyze Performance", type="primary", use_container_width=True)
    
    if analyze_button or 'last_prediction' in st.session_state:
        if analyze_button:
            with st.spinner("Analyzing your performance..."):
                result = predictor.predict_rank(score, exam_year, attempted, correct)
                st.session_state.last_prediction = result
        else:
            result = st.session_state.last_prediction
        
        st.markdown('<div class="section-header">Performance Summary</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Qualification Logic for display
        is_qualified = score >= 40.0
        display_rank = f"Rank: {result['ai_rank']:,}" if is_qualified else "Not Qualified"
        display_percentile = f"Percentile: {result['percentile']:.2f}%" if is_qualified else "Below Threshold"
        display_perf_label = result['performance_label'] if is_qualified else "Not Qualified"
        display_helper = result['helper_text'] if is_qualified else "Score is below the 40-mark qualifying threshold."

        with col1:
            render_metric_card("Score", f"{score:.2f} / 160", f"({result['score_percentage']:.1f}%)")
        
        with col2:
            render_metric_card("AI Rank & Percentile", display_rank, display_percentile)
        
        with col3:
            render_metric_card("Accuracy", f"{result['accuracy']:.1f}%", f"{correct} correct out of {attempted} attempted")
        
        with col4:
            perf_color = get_performance_color(display_perf_label)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Performance</div>
                <div class="metric-value {perf_color}">{display_perf_label}</div>
                <div class="metric-subtitle">{display_helper}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Performance Comparison Section
        st.markdown('<div class="section-header">Performance Comparison</div>', unsafe_allow_html=True)
        st.markdown("**Your Score**")
        st.progress(score / 160, text=f"{score:.2f} / 160")
        st.markdown("**National Average**")
        st.progress(51/160, text="51 / 160 (32.0%)")
        st.markdown("**Topper Score**")
        st.progress(1.0, text="160 / 160 (100.0%)")

        # Detailed Insights (Only show rank range if qualified)
        st.markdown('<div class="section-header">Detailed Insights</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            if is_qualified:
                st.metric("Rank Range", f"{result['ai_rank']:,}", delta=f"±{(result['rank_upper_bound'] - result['rank_lower_bound']) // 2:,}", delta_color="off")
                st.caption(f"Range: {result['rank_lower_bound']:,} - {result['rank_upper_bound']:,}")
            else:
                st.metric("Qualification", "Fail", delta="Below 40", delta_color="inverse")
                st.caption("Minimum 40 marks required for General/OBC rank.")
        with col2:
            st.metric("Attempt Rate", f"{(attempted/160)*100:.1f}%", f"{attempted} / 160")
        with col3:
            st.metric("Wrong Answers", attempted - correct, "No negative marking")

        st.markdown("""
        <div class="disclaimer">
            ⚠️ <strong>Disclaimer:</strong> General/OBC candidates require 40/160 (25%) marks to qualify for a rank. 
            SC/ST candidates have no minimum marks requirement. Rankings shown are estimates.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("👈 Enter your exam details in the sidebar and click 'Analyze Performance' to start.")

if __name__ == "__main__":
    main()