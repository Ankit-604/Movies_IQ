import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_ind, chi2_contingency
import ast

# --- Page Config ---
st.set_page_config(page_title="🎬 MovieIQ Dashboard", layout="wide")

# --- Global Professional Visualization Styling ---
# Ensures all generated matplotlib/seaborn charts match the clean light corporate theme
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available() else 'default')
plt.rcParams.update({
    'figure.facecolor': '#F8FAFC',   # Matches application background
    'axes.facecolor': '#FFFFFF',     # Clean white plot backgrounds
    'text.color': '#1E293B',         # Charcoal labels
    'axes.labelcolor': '#1E293B',
    'xtick.color': '#64748B',        # Muted gray ticks
    'ytick.color': '#64748B',
    'axes.edgecolor': '#E2E8F0',     # Subtle border lines
    'grid.color': '#F1F5F9'          # Ultra-soft grid lines
})

# Professional Muted Palette Colors
COLOR_SUCCESS = "#0F766E"  # Deep teal for success
COLOR_FAILURE = "#94A3B8"  # Professional slate gray for failure/non-success
PALETTE_MUTED = [COLOR_FAILURE, COLOR_SUCCESS]

# --- Sidebar About Section ---
with st.sidebar.expander("ℹ️ About this app", expanded=True):
    st.markdown("""
    **MovieIQ** leverages advanced predictive modeling to determine the commercial viability of film concepts.
    
    **Core Dimensions Evaluated:**
    * Budget & Execution Scale
    * Algorithmic Popularity Metrics
    * Target Runtime Window
    * Historical Audience Reception
    """)

# --- App Title ---
st.title("🎬 MovieIQ — Film Predictive Analytics")
st.markdown("---")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload corporate movie asset ledger (CSV Format)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # --- Preprocessing ---
    df = df[["budget", "revenue", "popularity", "runtime", "vote_average", "title", "genres"]]
    df = df[df["budget"] > 0]
    df = df[df["revenue"] > 0]
    df.dropna(inplace=True)
    df["success"] = (df["revenue"] > df["budget"]).astype(int)
    df["main_genre"] = df["genres"].apply(lambda x: ast.literal_eval(x)[0]['name'] if x != '[]' else "Unknown")

    # --- Top-Level KPIs ---
    kp1, kp2, kp3 = st.columns(3)
    kp1.metric("🎞️ Operational Data Volume", f"{len(df):,}")
    kp2.metric("✅ Benchmark Success Rate", f"{df['success'].mean()*100:.1f}%")
    kp3.metric("🎬 Monitored Genre Segments", df['main_genre'].nunique())
    
    st.markdown("---")

    # --- Sidebar Filters ---
    st.sidebar.header("🔍 Filter Options")
    selected_genres = st.sidebar.multiselect("Select Genre(s)", options=df["main_genre"].unique(), default=df["main_genre"].unique())
    min_votes = st.sidebar.slider("Minimum Vote Average", 0.0, 10.0, 5.0)

    # --- Apply Filters ---
    filtered_df = df[(df["main_genre"].isin(selected_genres)) & (df["vote_average"] >= min_votes)]

    # --- Layout Grid Split for Overview ---
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("🎯 Active Dataset Sample")
        st.dataframe(filtered_df.grid_head(10), use_container_width=True)

    with col_right:
        st.subheader("📊 Numerical Feature Distribution")
        st.dataframe(filtered_df.describe().T[['mean', 'std', 'min', 'max']], use_container_width=True)

    st.markdown("---")

    # --- Charts Section ---
    st.subheader("📈 Exploratory Data Visualizations")
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("**Capital Investment vs. Returns**")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(
            data=filtered_df, 
            x="budget", 
            y="revenue", 
            hue="success", 
            palette=PALETTE_MUTED, 
            alpha=0.7, 
            ax=ax1
        )
        ax1.set_xlabel("Production Budget ($)")
        ax1.set_ylabel("Gross Revenue ($)")
        ax1.legend(title="Outcome Status", labels=["Failure/Loss", "Profitable Success"])
        st.pyplot(fig1)

    with col_chart2:
        st.markdown("**Metric Performance Metrics Across Outcome Classes**")
        avg_metrics = filtered_df.groupby("success")[["popularity", "runtime", "vote_average"]].mean().reset_index()
        avg_metrics = pd.melt(avg_metrics, id_vars="success", var_name="Metric", value_name="Average Value")

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.barplot(
            data=avg_metrics, 
            x="Metric", 
            y="Average Value", 
            hue="success", 
            palette=PALETTE_MUTED, 
            ax=ax2
        )
        ax2.set_xlabel("Key Performance Indicators")
        ax2.set_ylabel("Normalized Group Mean")
        ax2.legend(title="Outcome Status", labels=["Failure/Loss", "Profitable Success"])
        st.pyplot(fig2)

    st.markdown("---")

    # --- Statistical Variance Testing ---
    st.subheader("🧐 Statistical Significance Diagnostics")
    
    t_stat, p_val = ttest_ind(filtered_df[filtered_df["success"] == 1]["vote_average"],
                              filtered_df[filtered_df["success"] == 0]["vote_average"])
    
    contingency = pd.crosstab(filtered_df["main_genre"], filtered_df["success"])
    chi2, chi_p, dof, expected = chi2_contingency(contingency)
    
    stat_col1, stat_col2 = st.columns(2)
    stat_col1.info(f"**T-Test Evaluation (Audience Rating Variation):**\n* T-Statistic: `{t_stat:.3f}`\n* Variance Significance (p-value): `{p_val:.4f}`")
    stat_col2.info(f"**Chi-Square Evaluation (Genre Distribution Control):**\n* Chi2 Score: `{chi2:.3f}`\n* Structural Significance (p-value): `{chi_p:.4f}`")

    st.markdown("---")

    # --- ML Engine Section ---
    st.subheader("🤖 Machine Learning Performance Engine")

    features = filtered_df[['budget', 'popularity', 'runtime', 'vote_average']]
    target = filtered_df['success']

    if len(filtered_df) > 10:
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

        model = RandomForestClassifier(random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        ml_left, ml_right = st.columns([1, 1])
        
        with ml_left:
            st.metric("Model Predictive Accuracy", f"{model.score(X_test, y_test):.2%}")
            st.markdown("**Core Classification Performance Analytics**")
            st.text(classification_report(y_test, y_pred))
            
        with ml_right:
            st.markdown("**Model Confusion Error Bounds**")
            conf_matrix = confusion_matrix(y_test, y_pred)
            fig3, ax3 = plt.subplots(figsize=(5, 3.5))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", cbar=False, ax=ax3)
            ax3.set_xlabel("Predicted Label Target")
            ax3.set_ylabel("Ground Truth Target")
            st.pyplot(fig3)
    else:
        st.warning("Insufficient data elements filtered to run classification algorithms.")

    st.markdown("---")

    # --- Live Testing Interface ---
    st.subheader("🎬 Real-Time Simulation Deck")

    with st.form("prediction_form"):
        st.markdown("Provide conceptual input metrics below to simulate target market profitability forecasts:")
        
        sim_c1, sim_c2 = st.columns(2)
        input_budget = sim_c1.number_input("Target Capital Investment (USD)", min_value=1000, max_value=500000000, value=10000000, step=1000000)
        input_popularity = sim_c2.slider("Estimated Concept Popularity Score", 0.0, 100.0, 10.0)
        input_runtime = sim_c1.slider("Target Runtime (Minutes)", 30, 300, 120)
        input_vote_average = sim_c2.slider("Audience Score Benchmark Projection", 0.0, 10.0, 6.5)

        submit = st.form_submit_button("Run Strategic Viability Inference")

    if submit:
        input_data = pd.DataFrame({
            "budget": [input_budget],
            "popularity": [input_popularity],
            "runtime": [input_runtime],
            "vote_average": [input_vote_average]
        })

        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0][prediction]

        if prediction == 1:
            st.success(f"🌟 **Forecast Target Reached:** Project predicted as commercially viable (**Successful**) with a {prediction_proba:.1%} confidence validation.")
        else:
            st.error(f"🚨 **Risk Boundary Triggered:** Project predicted to fall below capital break-even constraints (**Unsuccessful**) with a {prediction_proba:.1%} confidence confirmation.")

    # --- Clean Data Export Deck ---
    st.markdown("---")
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Export Refined Asset Ledger (CSV)", data=csv, file_name="filtered_movies.csv", mime="text/csv")

else:
    st.info("💡 Complete application initialization by uploading the corporate dataset asset.")
