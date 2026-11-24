"""
app/pages/3_Association_Recommendation.py
=========================================
Association Rules & Product Recommendation Dashboard
"""

import streamlit as st
import os
import pandas as pd
from src.association.recommendations import recommend_products

st.set_page_config(page_title="Product Recommendation System", page_icon="🛒", layout="wide")

st.title("🛒 Product Recommendation System")
st.markdown("Discover frequent market basket associations and get real-time cross-sell recommendations.")

tab1, tab2 = st.tabs(["🛍️ Interactive Recommender (New Cart Input)", "📈 Association Rules & Metrics"])

# Pre-populated popular items for demo selection
SAMPLE_ITEMS = [
    "WHITE HANGING HEART T-LIGHT HOLDER",
    "RED HANGING HEART T-LIGHT HOLDER",
    "HAND WARMER UNION JACK",
    "HAND WARMER RED POLKA DOT",
    "KNITTED UNION FLAG HOT WATER BOTTLE",
    "RED WOOLLY HOTTIE WHITE HEART.",
    "WHITE METAL LANTERN",
    "CREAM CUPID HEARTS COAT HANGER",
    "WOOD BLACK BOARD ANT WHITE FINISH",
    "COLOUR GLASS T-LIGHT HOLDER HANGING"
]

with tab1:
    st.subheader("Simulate Customer Shopping Cart")
    selected = st.multiselect(
        "Select one or more items currently in the customer's cart:",
        options=SAMPLE_ITEMS,
        default=["WHITE HANGING HEART T-LIGHT HOLDER"]
    )
    
    top_n = st.slider("Number of recommendations to return:", 1, 10, 5)
    
    if st.button("Generate Product Recommendations", use_container_width=True):
        if not selected:
            st.warning("Please select at least one item to receive recommendations.")
        else:
            recs = recommend_products(selected, top_n=top_n)
            st.divider()
            if recs:
                st.success(f"🎯 Found {len(recs)} Recommended Products (Ranked by Lift & Confidence):")
                for idx, r in enumerate(recs, 1):
                    with st.container():
                        col_x, col_y, col_z = st.columns([3, 1, 1])
                        with col_x:
                            st.markdown(f"**{idx}. {r['recommended_item']}**")
                            st.caption(f"Rule: {r['based_on_rule']}")
                        with col_y:
                            st.metric("Lift Score", f"{r['lift']:.2f}")
                        with col_z:
                            st.metric("Confidence", f"{r['confidence']:.1%}")
                        st.divider()
            else:
                st.info("No strong association rules matched the selected combination.")

with tab2:
    st.subheader("Mined Association Rules Catalog")
    rules_path = "models/association/association_rules.csv"
    if os.path.exists(rules_path):
        rules_df = pd.read_csv(rules_path)
        st.dataframe(rules_df[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']], use_container_width=True)
    else:
        st.info("Mined rules table will be available here after rule extraction (Day 4).")
        
    st.markdown("**Rules Scatter Plot (Support vs Confidence colored by Lift)**")
    plot_path = "results/association/rules_scatter_plot.png"
    if os.path.exists(plot_path):
        st.image(plot_path)
    else:
        st.caption("Waiting for rules_scatter_plot.png")
