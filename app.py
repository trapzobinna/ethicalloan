import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- CONFIG ---
st.set_page_config(page_title="Ethical AI Auditor", layout="wide")

# --- MEMORY INITIALIZATION ---
if 'audit_log' not in st.session_state:
    st.session_state.audit_log = []

# --- BACKEND ---
@st.cache_data
def train_system():
    try:
        df = pd.read_csv('loan_experiment_data.csv')
    except FileNotFoundError:
        st.error("Dataset not found! Please run 'generate_data.py' first.")
        st.stop()
    
    df_encoded = pd.get_dummies(df, columns=['Gender', 'Race'])
    X = df_encoded.drop('Loan_Status', axis=1)
    y = df_encoded['Loan_Status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    
    return model, X_test, y_test, df

model, X_test, y_test, original_df = train_system()

# --- CENTRAL CALCULATIONS ---
probs = model.predict_proba(X_test)[:, 1]
baseline_prob = probs.mean() 

results_df = X_test.copy()
results_df['Probability'] = probs
results_df['AI_Decision_Value'] = (probs > 0.5).astype(int)
results_df['AI_Verdict'] = results_df['AI_Decision_Value'].map({1: "APPROVED", 0: "REJECTED"})

# --- UI NAVIGATION & GLOBAL STATS ---
st.title("⚖️ Ethical AI Loan Approval & Audit System")
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to:", ["Decision Dashboard", "Fairness Audit & Metrics", "Session Audit Trail"])

# New Global Stats Section so you always know what the "Average" is
st.sidebar.divider()
st.sidebar.subheader("📈 Global Audit Stats")
st.sidebar.metric("System Baseline (Avg)", f"{baseline_prob*100:.1f}%")
st.sidebar.caption("This is the 'Average' confidence across all applicants. Your delta (e.g. -67%) is relative to this number.")

# === PAGE 1: DECISION DASHBOARD ===
if page == "Decision Dashboard":
    st.subheader("📥 Human-in-the-Loop Review Queue")
    
    selected_index = st.selectbox("Select Applicant ID to Review:", results_df.index)
    applicant = results_df.loc[selected_index]

    prob_val = applicant['Probability']
    diff = prob_val - baseline_prob
    ai_verdict_str = applicant['AI_Verdict']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("### 👤 Applicant Profile")
        display_profile = applicant.drop(['Probability', 'AI_Decision_Value', 'AI_Verdict'])
        st.write(display_profile.to_frame().T)
        
        st.divider()
        st.markdown("### 🤖 AI Verdict")
        
        # This delta explains the -67% clearly now
        st.metric("Repayment Confidence", f"{prob_val*100:.1f}%", delta=f"{diff*100:.1f}% vs Average")
        
        if prob_val > 0.5:
            st.success(f"**STATUS: ✅ {ai_verdict_str}**")
        else:
            st.error(f"**STATUS: ❌ {ai_verdict_str}**")
        
    with col2:
        st.warning("### 🔍 The 'Why' (Individual Explainability)")
        
        try:
            raw_features = applicant.drop(['Probability', 'AI_Decision_Value', 'AI_Verdict'])
            numeric_features = pd.to_numeric(raw_features, errors='coerce').fillna(0)
            
            feature_diff = (numeric_features - X_test.mean()) / (X_test.std() + 1e-9)
            local_contribution = feature_diff * model.feature_importances_
            scaling_factor = abs(diff) / (local_contribution.abs().sum() + 1e-9)
            calibrated_contribution = local_contribution * scaling_factor
            
            top_indices = calibrated_contribution.abs().nlargest(8).index
            plot_series = calibrated_contribution[top_indices].sort_values()

            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in plot_series]
            plot_series.plot(kind='barh', color=colors, ax=ax)
            ax.set_title(f"Factors pulling score from {baseline_prob*100:.1f}% to {prob_val*100:.1f}%")
            ax.axvline(0, color='black', linewidth=0.8)
            plt.tight_layout()
            st.pyplot(fig)
            
            st.write(f"**Interpreting logic for ID {selected_index}:**")
            st.write(f"The AI starts at the baseline of **{baseline_prob*100:.1f}%**. These factors combined to move the final score by **{diff*100:.1f}%**.")
        except Exception as e:
            st.error("Visualization Error: Chart failed to load.")

    st.divider()
    st.subheader("📝 Final Human Decision (Ethical Oversight)")
    c1, c2 = st.columns([1, 2])
    with c1:
        human_decision = st.radio("Audit Action:", ["Confirm AI Decision", "Manual Override (Approve)", "Manual Override (Reject)"])
    with c2:
        reason = st.text_area("Audit Justification:", placeholder="Provide reason for decision...", key=f"reason_{selected_index}")
        
        if st.button("Submit Audit Record"):
            if reason:
                log_entry = {
                    "ID": selected_index,
                    "AI Confidence": f"{prob_val*100:.1f}%",
                    "AI Verdict": ai_verdict_str,
                    "Human Action": human_decision,
                    "Reasoning": reason
                }
                st.session_state.audit_log.append(log_entry)
                st.success(f"Successfully logged decision for ID {selected_index}")
            else:
                st.warning("Justification is required for the audit trail.")

# === PAGE 2: FAIRNESS AUDIT ===
elif page == "Fairness Audit & Metrics":
    st.subheader("📊 Bias Detection & Performance Metrics")
    acc = (results_df['AI_Decision_Value'] == y_test).mean()
    st.metric("System Accuracy", f"{acc*100:.1f}%")
    
    st.divider()
    st.write("### ⚖️ Demographic Parity Audit")
    
    def plot_parity(column_prefix):
        cols = [c for c in X_test.columns if column_prefix in c]
        for col in cols:
            group_mask = X_test[col] == 1
            if group_mask.any():
                rate = results_df[group_mask]['AI_Decision_Value'].mean()
                label = col.replace(f"{column_prefix}_", "")
                st.progress(rate, text=f"**{label}** Approval Rate: {rate*100:.1f}%")

    st.write("**Gender Approval Rates**")
    plot_parity('Gender')
    st.write("**Race Approval Rates**")
    plot_parity('Race')

# === PAGE 3: AUDIT TRAIL ===
elif page == "Session Audit Trail":
    st.subheader("📜 Human-in-the-Loop Audit Trail")
    if st.session_state.audit_log:
        audit_df = pd.DataFrame(st.session_state.audit_log)
        st.table(audit_df)
        csv = audit_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Audit Report (CSV)",
            data=csv,
            file_name="loan_audit_report.csv",
            mime="text/csv",
        )
    else:
        st.info("No audit records found.")