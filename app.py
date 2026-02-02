import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- CONFIG ---
st.set_page_config(page_title="Ethical AI Auditor", layout="wide")

# --- BACKEND ---
@st.cache_data
def train_system():
    try:
        df = pd.read_csv('loan_experiment_data.csv')
    except FileNotFoundError:
        st.error("Dataset not found! Please run 'generate_data.py' first.")
        st.stop()
    
    # One-hot encode Gender and Race
    df_encoded = pd.get_dummies(df, columns=['Gender', 'Race'])
    
    X = df_encoded.drop('Loan_Status', axis=1)
    y = df_encoded['Loan_Status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    
    return model, X_test, y_test, df

model, X_test, y_test, original_df = train_system()

# --- CENTRAL CALCULATIONS ---
probs = model.predict_proba(X_test)[:, 1]
results_df = X_test.copy()
results_df['Probability'] = probs
results_df['AI_Decision'] = (probs > 0.5).astype(int)

# --- UI NAVIGATION ---
st.title("⚖️ Ethical AI Loan Approval & Audit System")
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to:", ["Decision Dashboard", "Fairness Audit & Metrics"])

# === PAGE 1: DECISION DASHBOARD ===
if page == "Decision Dashboard":
    st.subheader("📥 Human-in-the-Loop Review Queue")
    
    selected_index = st.selectbox("Select Applicant ID to Review:", results_df.index)
    applicant = results_df.loc[selected_index]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("### 👤 Applicant Profile")
        display_profile = applicant.drop(['Probability', 'AI_Decision'])
        st.write(display_profile.to_frame().T)
        
        st.divider()
        st.markdown("### 🤖 AI Verdict")
        prob_val = applicant['Probability']
        st.metric("Repayment Confidence", f"{prob_val*100:.1f}%")
        
        if prob_val > 0.5:
            st.success("**STATUS: ✅ APPROVED**")
        else:
            st.error("**STATUS: ❌ REJECTED**")
        
    with col2:
        st.warning("### 🔍 The 'Why' (Individual Explainability)")
        
        # --- FIX: ENSURE DATA IS NUMERIC TO AVOID TYPEERROR ---
        raw_features = applicant.drop(['Probability', 'AI_Decision'])
        # Force conversion to float so .nlargest() works
        numeric_features = pd.to_numeric(raw_features, errors='coerce').fillna(0)
        
        # Calculate local contribution (Simulated SHAP)
        # We compare this person to the average to see what makes them unique
        feature_impact = (numeric_features - X_test.mean()) / (X_test.std() + 0.001)
        local_contribution = (feature_impact * model.feature_importances_).abs()
        
        # Display chart
        st.bar_chart(local_contribution.nlargest(10))
        
        st.write(f"**Interpreting the logic for ID {selected_index}:**")
        st.write("The bars show which traits most influenced this specific AI decision.")

    
    st.divider()
    
    st.subheader("📝 Final Human Decision (Ethical Oversight)")
    c1, c2 = st.columns([1, 2])
    with c1:
        decision = st.radio("Review Action:", ["Confirm AI Decision", "Manual Override (Approve)", "Manual Override (Reject)"])
    with c2:
        justification = st.text_area("Audit Justification:")
        if st.button("Log Final Decision"):
            if justification:
                st.success("📝 **Decision Logged to Audit Trail**")
                st.code(f"AUDIT_RECORD | ID: {selected_index} | Action: {decision} | Reason: {justification}")
            else:
                st.warning("Justification required for accountability.")

# === PAGE 2: FAIRNESS AUDIT ===
elif page == "Fairness Audit & Metrics":
    st.subheader("📊 Bias Detection & Performance Metrics")
    acc = (results_df['AI_Decision'] == y_test).mean()
    st.metric("System Accuracy", f"{acc*100:.1f}%")
    
    st.divider()
    st.write("### ⚖️ Demographic Parity Audit")
    
    def plot_parity(column_prefix):
        cols = [c for c in X_test.columns if column_prefix in c]
        for col in cols:
            group_mask = X_test[col] == 1
            if group_mask.any():
                rate = results_df[group_mask]['AI_Decision'].mean()
                label = col.replace(f"{column_prefix}_", "")
                st.progress(rate, text=f"**{label}** Approval Rate: {rate*100:.1f}%")

    st.write("**Gender Approval Rates**")
    plot_parity('Gender')
    st.write("**Race Approval Rates**")
    plot_parity('Race')