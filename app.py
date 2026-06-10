import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(
    page_title="AI Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

st.markdown("""
<style>
.metric-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px; padding: 20px; text-align: center; color: white;
}
.stButton>button {
    background: linear-gradient(90deg, #667eea, #764ba2);
    color: white; border: none; border-radius: 8px;
    padding: 10px 30px; font-size: 16px; font-weight: bold; width: 100%;
}
</style>
""", unsafe_allow_html=True)

st.title("🎓 AI-Powered Student Performance Prediction System")
st.markdown("**Predict Placement, Salary & Burnout Risk using Machine Learning**")
st.markdown("---")

@st.cache_resource
def load_models():
    models = {}
    for name, path in [('placement','models/placement_model.pkl'),
                        ('salary','models/salary_model.pkl'),
                        ('burnout','models/burnout_model.pkl')]:
        if os.path.exists(path):
            with open(path,'rb') as f:
                models[name] = pickle.load(f)
    return models

models = load_models()

if not models:
    st.warning("⚠️ Models not found! Run notebooks 01 → 03 first to train and save models.")
    st.info("Steps: 01_data_preparation → 03a_placement_model → 03b_salary_model → 03c_burnout_model")
    st.stop()

# ── Sidebar Info ────────────────────────────────────────────────────────────
st.sidebar.header("📝 Student Profile")
st.sidebar.info("👉 Fill in your details in the form below to get predictions")
st.sidebar.markdown("---")
st.sidebar.markdown("**📊 Features:**")
st.sidebar.markdown("- 🎓 Academic Performance")
st.sidebar.markdown("- 💻 Coding Skills")
st.sidebar.markdown("- 🧠 Soft Skills")
st.sidebar.markdown("- 😴 Lifestyle Habits")

# ── Main Page Input Form ────────────────────────────────────────────────────────────
st.subheader("📝 Student Profile")

with st.expander("🎓 Academic Info", expanded=True):
    ac1, ac2, ac3 = st.columns(3)
    with ac1:
        cgpa = st.slider("CGPA", 4.0, 10.0, 7.5, 0.1)
        college_tier = st.selectbox("College Tier", [1, 2, 3])
    with ac2:
        branch = st.selectbox("Branch", [0, 1, 2, 3, 4, 5],
                               format_func=lambda x: ['CSE','IT','ECE','ME','CE','EE'][x])
        backlog = st.number_input("Backlogs", 0, 10, 0)
    with ac3:
        attendance = st.slider("Attendance %", 40, 100, 80)
        internships = st.slider("Internships Completed", 0, 5, 1)

with st.expander("💻 Coding Profile", expanded=True):
    cd1, cd2, cd3 = st.columns(3)
    with cd1:
        dsa = st.slider("DSA Problems Solved", 0, 500, 100)
        github = st.slider("GitHub Repos", 0, 50, 5)
    with cd2:
        hackathons = st.slider("Hackathons", 0, 20, 2)
        dev_projects = st.slider("Dev Projects", 0, 20, 3)
    with cd3:
        ai_ml_proj = st.slider("AI/ML Projects", 0, 10, 1)

with st.expander("🧠 Skills & Soft Skills", expanded=True):
    sk1, sk2 = st.columns(2)
    with sk1:
        communication = st.slider("Communication (1-10)", 1, 10, 7)
        aptitude = st.slider("Aptitude Score (1-100)", 1, 100, 65)
    with sk2:
        mock_score = st.slider("Mock Interview (1-100)", 1, 100, 60)
        resume_score = st.slider("Resume Score (1-100)", 1, 100, 65)

with st.expander("😴 Lifestyle", expanded=True):
    lf1, lf2, lf3 = st.columns(3)
    with lf1:
        sleep = st.slider("Sleep Hours/day", 3.0, 10.0, 7.0, 0.5)
        study = st.slider("Study Hours/day", 0.0, 12.0, 4.0, 0.5)
    with lf2:
        stress = st.slider("Stress Level (1-10)", 1, 10, 5)
        gaming = st.slider("Gaming Hours/day", 0.0, 8.0, 1.0, 0.5)
    with lf3:
        gym = st.slider("Gym Frequency (days/week)", 0, 7, 3)
        screen = st.slider("Screen Time (hrs/day)", 2.0, 16.0, 7.0, 0.5)

predict_btn = st.button("🚀 Predict Now!", type="primary", use_container_width=True)

# ── Feature Engineering (same as notebook) ───────────────────────────────────
def compute_features(cgpa, dsa, github, hackathons, dev_projects, ai_ml_proj,
                      internships, communication, aptitude, mock_score, resume_score,
                      college_tier, branch, backlog, attendance,
                      sleep, study, stress, gaming, gym, screen):

    coding_score = (dsa*0.5 + github*10 + hackathons*20 + dev_projects*15 + ai_ml_proj*25) / 100
    ai_readiness  = (2*20 + 5*10 + (10-5)*5 + 7*8) / 100  # default mid values
    wellness = (sleep*10 + (24-screen)*5 + (8-min(gaming,8))*5 + (10-stress)*8 + gym*6) / 100
    interview_r = mock_score*0.3 + communication*0.25 + aptitude*0.25 + resume_score*0.2

    placement_feat = [cgpa, backlog, attendance, dsa, github, hackathons, dev_projects,
                       ai_ml_proj, internships, resume_score, communication, aptitude,
                       mock_score, college_tier, branch, coding_score, ai_readiness, interview_r]

    salary_feat = [cgpa, college_tier, branch, coding_score, ai_readiness,
                    interview_r, internships, 1, 2, 0]  # offer_count=1, rounds=2, company_type=0

    burnout_feat = [sleep, screen, gaming, study, stress, gym, 2.0, 7, wellness, cgpa, attendance]

    return placement_feat, salary_feat, burnout_feat

# ── Prediction & Display ──────────────────────────────────────────────────────
if predict_btn:
    p_feat, s_feat, b_feat = compute_features(
        cgpa, dsa, github, hackathons, dev_projects, ai_ml_proj,
        internships, communication, aptitude, mock_score, resume_score,
        college_tier, branch, backlog, attendance,
        sleep, study, stress, gaming, gym, screen
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    # PLACEMENT
    with col1:
        st.subheader("🎯 Placement Prediction")
        if 'placement' in models:
            try:
                pm = models['placement']
                n_feat = pm.n_features_in_
                X_p = np.array(p_feat[:n_feat]).reshape(1, -1)
                pred = pm.predict(X_p)[0]
                prob = pm.predict_proba(X_p)[0][1]
                if pred == 1:
                    st.success(f"✅ **PLACED**")
                    st.metric("Confidence", f"{prob:.1%}")
                else:
                    st.error(f"❌ **NOT PLACED**")
                    st.metric("Confidence", f"{1-prob:.1%}")
                st.progress(float(prob))
                st.caption(f"Placement probability: {prob:.1%}")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.info("Model not loaded yet")

    # SALARY
    with col2:
        st.subheader("💰 Salary Forecast")
        if 'salary' in models:
            try:
                sm = models['salary']
                n_feat = sm.n_features_in_
                X_s = np.array(s_feat[:n_feat]).reshape(1, -1)
                sal = sm.predict(X_s)[0]
                sal = max(0, sal)
                st.info(f"💵 **{sal:.1f} LPA**")
                st.metric("Expected Package", f"₹{sal:.1f} LPA")
                tier = "🌟 Premium" if sal>15 else ("✅ Good" if sal>8 else "📈 Entry")
                st.caption(f"Package tier: {tier}")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.info("Model not loaded yet")

    # BURNOUT
    with col3:
        st.subheader("🔥 Burnout Risk")
        if 'burnout' in models:
            try:
                bm = models['burnout']
                n_feat = bm.n_features_in_
                X_b = np.array(b_feat[:n_feat]).reshape(1, -1)
                brisk = bm.predict(X_b)[0]
                bprob = bm.predict_proba(X_b)[0][1]
                if brisk == 1:
                    st.warning(f"⚠️ **HIGH RISK**")
                    st.metric("Burnout Probability", f"{bprob:.1%}")
                    st.caption("💡 Tip: Improve sleep & reduce screen time")
                else:
                    st.success(f"✅ **LOW RISK**")
                    st.metric("Burnout Probability", f"{bprob:.1%}")
                    st.caption("Keep maintaining your wellness habits!")
                st.progress(float(bprob))
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.info("Model not loaded yet")

    st.markdown("---")

    # ── Key Stats ──────────────────────────────────────────────────────────────
    st.subheader("📊 Your Profile Analysis")
    c1, c2, c3, c4 = st.columns(4)
    coding_score = (dsa*0.5 + github*10 + hackathons*20 + dev_projects*15 + ai_ml_proj*25) / 100
    wellness = (sleep*10 + (24-screen)*5 + (8-min(gaming,8))*5 + (10-stress)*8 + gym*6) / 100
    interview_r = mock_score*0.3 + communication*0.25 + aptitude*0.25 + resume_score*0.2

    c1.metric("Coding Score", f"{coding_score:.1f}")
    c2.metric("Wellness Score", f"{wellness:.1f}")
    c3.metric("Interview Ready", f"{interview_r:.1f}/100")
    c4.metric("CGPA", f"{cgpa:.1f}/10")

    st.markdown("---")
    st.markdown("*Built with ❤️ using Python, Scikit-learn, XGBoost & Streamlit | Dataset: 25K synthetic records*")
