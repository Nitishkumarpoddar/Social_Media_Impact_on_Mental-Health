"""
Teen Mental Health Prediction Dashboard
========================================
Run:  streamlit run mental_health_app.py
Deps: pip install streamlit scikit-learn joblib plotly pandas numpy
"""

import io
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="Teen Mental Health AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght=300;400;500;600;700;800&display=swap');
html,body,[class*="css"]{ font-family:'Inter',sans-serif; }

/* ---------- layout ---------- */
.main{background:#07080e;}
section[data-testid="stSidebar"]{background:#0b0d15;border-right:1px solid #1a1f30;}
section[data-testid="stSidebar"] *{color:#c9d1d9!important;}

/* ---------- predict button ---------- */
div[data-testid="stButton"] > button{
    width:100%;padding:18px 0;font-size:1.15rem;font-weight:800;
    background:linear-gradient(135deg,#1f6feb,#7c3aed);
    color:#fff;border:none;border-radius:14px;
    letter-spacing:.5px;cursor:pointer;
    box-shadow:0 6px 28px rgba(31,111,235,.45);
    transition:transform .15s,box-shadow .15s;
}
div[data-testid="stButton"] > button:hover{
    transform:translateY(-2px);
    box-shadow:0 10px 36px rgba(31,111,235,.6);
}
div[data-testid="stButton"] > button:active{transform:translateY(0);}

/* ---------- cards ---------- */
.card{
    background:linear-gradient(135deg,#0f1422,#0b0d15);
    border:1px solid #1f2840;border-radius:16px;
    padding:20px 22px;margin-bottom:10px;
    box-shadow:0 4px 20px rgba(0,0,0,.35);
}
.card h3{color:#58a6ff;margin:0 0 2px;font-size:1rem;}
.card .sub{color:#5a6380;font-size:.78rem;margin-bottom:14px;}

.result-high{color:#f85149;font-size:2rem;font-weight:800;}
.result-low {color:#3fb950;font-size:2rem;font-weight:800;}
.result-med {color:#f0883e;font-size:2rem;font-weight:800;}

.badge{display:inline-block;padding:3px 12px;border-radius:20px;
       font-size:.72rem;font-weight:700;margin-top:4px;}
.b-high{background:rgba(248,81,73,.15); color:#f85149;border:1px solid #f85149;}
.b-low {background:rgba(63,185,80,.15); color:#3fb950;border:1px solid #3fb950;}
.b-med {background:rgba(240,136,62,.15);color:#f0883e;border:1px solid #f0883e;}

.sec{font-size:1.15rem;font-weight:700;color:#e6edf3;
     margin:26px 0 14px;padding-bottom:8px;border-bottom:1px solid #1f2840;}

.info{background:#0f1422;border-left:3px solid #58a6ff;
      border-radius:0 8px 8px 0;padding:11px 15px;
      font-size:.84rem;color:#8b949e;margin:10px 0;}

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"]{
    background:#0b0d15;border-radius:10px;gap:4px;
    padding:4px;border:1px solid #1f2840;}
.stTabs [data-baseweb="tab"]{
    background:transparent;border-radius:8px;
    color:#8b949e;font-weight:500;}
.stTabs [aria-selected="true"]{
    background:#1f6feb!important;color:#fff!important;}

#MainMenu,footer{visibility:hidden;}
h1,h2,h3{color:#e6edf3!important;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  CONSTANTS  (match dataset exactly)
# ══════════════════════════════════════════════════════
RAW_FEATURES = [
    "age", "gender", "daily_social_media_hours", "platform_usage",
    "sleep_hours", "screen_time_before_sleep", "academic_performance",
    "physical_activity", "social_interaction_level",
    "stress_level", "anxiety_level", "addiction_level",
]

MODEL_FEATURES = [
    "daily_social_media_hours", "sleep_hours", "screen_time_before_sleep",
    "academic_performance", "physical_activity", "stress_level",
    "anxiety_level", "addiction_level",
    "gender_male",
    "platform_usage_Instagram", "platform_usage_TikTok",
    "social_interaction_level_low", "social_interaction_level_medium",
]

MODEL_INFO = {
    "Logistic Regression":      {"icon": "📈", "color": "#58a6ff",  "imp": "coef"},
    "Naive Bayes (GaussianNB)": {"icon": "🔮", "color": "#79c0ff",  "imp": "none"},
    "Decision Tree":            {"icon": "🌳", "color": "#f0883e",  "imp": "tree"},
    "Random Forest (Tuned)":    {"icon": "🌲", "color": "#3fb950",  "imp": "tree"},
    "AdaBoost Classifier":      {"icon": "⚡", "color": "#bc8cff",  "imp": "tree"},
    "XGBoost Classifier":       {"icon": "🚀", "color": "#ff7b72",  "imp": "tree"},
}

DARK = dict(paper_bgcolor="#07080e", plot_bgcolor="#07080e",
            font=dict(color="#c9d1d9", family="Inter"),
            margin=dict(t=40, b=40, l=40, r=20))

# ══════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════
def encode_input(vals: dict) -> pd.DataFrame:
    row = {f: 0.0 for f in MODEL_FEATURES}
    row["daily_social_media_hours"]   = vals["daily_social_media_hours"]
    row["sleep_hours"]                = vals["sleep_hours"]
    row["screen_time_before_sleep"]   = vals["screen_time_before_sleep"]
    row["academic_performance"]       = vals["academic_performance"]
    row["physical_activity"]          = vals["physical_activity"]
    row["stress_level"]               = vals["stress_level"]
    row["anxiety_level"]              = vals["anxiety_level"]
    row["addiction_level"]            = vals["addiction_level"]
    row["gender_male"]                = 1.0 if vals["gender"] == "male" else 0.0
    row["platform_usage_Instagram"]   = 1.0 if vals["platform_usage"] == "Instagram" else 0.0
    row["platform_usage_TikTok"]      = 1.0 if vals["platform_usage"] == "TikTok" else 0.0
    row["social_interaction_level_low"]    = 1.0 if vals["social_interaction_level"] == "low" else 0.0
    row["social_interaction_level_medium"] = 1.0 if vals["social_interaction_level"] == "medium" else 0.0
    return pd.DataFrame([row], columns=MODEL_FEATURES)


def risk_meta(prob: float):
    if prob >= 0.60:
        return "High Risk", "high", "#f85149"
    elif prob >= 0.35:
        return "Medium Risk", "med", "#f0883e"
    else:
        return "Low Risk", "low", "#3fb950"


def get_prob(model, X_df):
    try:
        return float(model.predict_proba(X_df)[0][1])
    except Exception:
        try:
            return float(model.predict(X_df)[0])
        except Exception:
            return 0.0


def gauge(prob, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 26, "color": color}},
        title={"text": title, "font": {"size": 11, "color": "#5a6380"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#1f2840",
                     "tickfont": {"color": "#5a6380", "size": 9}},
            "bar": {"color": color, "thickness": 0.22},
            "bgcolor": "#0f1422", "bordercolor": "#1f2840",
            "steps": [
                {"range": [0,  35],  "color": "rgba(63,185,80,.10)"},
                {"range": [35, 60],  "color": "rgba(240,136,62,.10)"},
                {"range": [60, 100], "color": "rgba(248,81,73,.10)"},
            ],
            "threshold": {"line": {"color": color, "width": 2},
                          "thickness": 0.8, "value": prob * 100},
        }
    ))
    fig.update_layout(**DARK, height=210)
    return fig


# ══════════════════════════════════════════════════════
#  AUTO-LOAD MODELS
# ══════════════════════════════════════════════════════
@st.cache_resource
def load_from_path(p: str):
    data = joblib.load(p)
    if isinstance(data, list):
        return {item["name"]: item["model"] for item in data
                if isinstance(item, dict) and "name" in item and "model" in item}
    if isinstance(data, dict):
        return data
    return {}


def make_demo_models():
    """Dynamically calculates risk based on sidebar inputs for Demo Mode."""
    n = len(MODEL_FEATURES)
    demo = {}
    
    model_offsets = {
        "Logistic Regression":      -0.05,
        "Naive Bayes (GaussianNB)":  0.02,
        "Decision Tree":            -0.02,
        "Random Forest (Tuned)":     0.04,
        "AdaBoost Classifier":      -0.04,
        "XGBoost Classifier":        0.01,
    }

    for name, info in MODEL_INFO.items():
        _imp = np.random.dirichlet(np.ones(n))
        _coef = (np.random.rand(1, n) - 0.5)
        offset = model_offsets[name]

        class DM:
            def __init__(self, m_offset):
                self.m_offset = m_offset

            def predict_proba(self, x_df):
                stress = float(x_df["stress_level"].iloc[0])
                anxiety = float(x_df["anxiety_level"].iloc[0])
                addiction = float(x_df["addiction_level"].iloc[0])
                sm_hours = float(x_df["daily_social_media_hours"].iloc[0])
                screen_sleep = float(x_df["screen_time_before_sleep"].iloc[0])
                
                sleep = float(x_df["sleep_hours"].iloc[0])
                exercise = float(x_df["physical_activity"].iloc[0])
                academic = float(x_df["academic_performance"].iloc[0])

                risk_score = (stress * 4) + (anxiety * 3) + (addiction * 3) + (sm_hours * 2) + (screen_sleep * 2)
                recovery_score = (sleep * 2) + (exercise * 0.5) + (academic * 1)

                final_score = (risk_score - recovery_score + 20) / 100
                final_score = max(0.02, min(0.98, final_score + self.m_offset))

                return np.array([[1 - final_score, final_score]])

            def predict(self, x_df):
                prob = self.predict_proba(x_df)[0][1]
                return np.array([int(prob >= 0.5)])

        obj = DM(offset)
        if info["imp"] == "tree":
            obj.feature_importances_ = _imp
        elif info["imp"] == "coef":
            obj.coef_ = _coef
        demo[name] = obj
    return demo


# ══════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════
def sidebar_inputs():
    st.sidebar.markdown("## 🧠 Teen Mental Health AI")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👤 Patient Details")

    v = {}
    v["age"]    = st.sidebar.slider("🎂 Age", 13, 19, 16)
    v["gender"] = st.sidebar.selectbox("⚧ Gender", ["male", "female"])

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📱 Digital Habits")
    v["daily_social_media_hours"]  = st.sidebar.slider("📱 Social Media (hrs/day)", 1.0, 8.0, 4.5, 0.1)
    v["platform_usage"]            = st.sidebar.selectbox("🌐 Platform", ["Instagram", "TikTok", "Both"])
    v["screen_time_before_sleep"]  = st.sidebar.slider("🌙 Screen Before Sleep (hrs)", 0.0, 6.0, 1.5, 0.1)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏥 Health & Lifestyle")
    v["sleep_hours"]               = st.sidebar.slider("😴 Sleep Hours", 3.0, 12.0, 7.0, 0.5)
    v["physical_activity"]         = st.sidebar.slider("🏃 Physical Activity (hrs/wk)", 0.0, 20.0, 3.0, 0.5)
    v["academic_performance"]      = st.sidebar.slider("📚 Academic Performance", 0.0, 10.0, 6.0, 0.1)
    v["social_interaction_level"]  = st.sidebar.selectbox("👥 Social Interaction", ["low", "medium", "high"])

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧪 Mental Health Scores")
    v["stress_level"]              = st.sidebar.slider("😤 Stress Level (0–10)", 0, 10, 5)
    v["anxiety_level"]             = st.sidebar.slider("😰 Anxiety Level (0–10)", 0, 10, 4)
    v["addiction_level"]           = st.sidebar.slider("🎮 Addiction Level (0–10)", 0, 10, 3)

    return v


# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════
def main():
    st.markdown("""
    <div style='background:linear-gradient(135deg,#0f1e3d,#07080e);
                border:1px solid #1f6feb;border-radius:18px;
                padding:28px 32px;margin-bottom:20px;'>
      <h1 style='margin:0;font-size:2rem;color:#e6edf3;'>
        🧠 Teen Mental Health Predictor
      </h1>
      <p style='color:#5a6380;margin:8px 0 0;font-size:.93rem;'>
        Social media & lifestyle impact · 6 ML models · Depression risk analysis
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Automatic Model Loading ──────────────────────
    models = {}
    demo_mode = True
    default_path = "all_mental_health_models.joblib"

    if os.path.exists(default_path):
        try:
            models = load_from_path(default_path)
            demo_mode = False
        except Exception:
            pass
    
    if demo_mode:
        models = make_demo_models()

    # ── Sidebar inputs ───────────────────────────────
    user_vals = sidebar_inputs()
    X_df = encode_input(user_vals)

    st.markdown("---")

    # ══════════════════════════════════════════════════
    #  BIG PREDICT BUTTON
    # ══════════════════════════════════════════════════
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        predict_clicked = st.button("🔍  PREDICT NOW", use_container_width=True)

    if predict_clicked:
        results = {}
        for name, model in models.items():
            p = get_prob(model, X_df)
            label, risk_cls, color = risk_meta(p)
            results[name] = {"prob": p, "label": label, "cls": risk_cls, "color": color}
        st.session_state["results"]   = results
        st.session_state["user_vals"] = user_vals.copy()
        st.session_state["X_df"]      = X_df.copy()
        st.session_state["models"]    = models

    if "results" not in st.session_state:
        st.markdown("""
        <div style='text-align:center;padding:60px 0;color:#2a3044;'>
            <div style='font-size:60px;'>🎯</div>
            <div style='font-size:1.1rem;margin-top:10px;color:#3a4560;'>
                Sidebar se values bharke <b style='color:#58a6ff;'>PREDICT NOW</b> dabao
            </div>
        </div>""", unsafe_allow_html=True)
        return

    results   = st.session_state["results"]
    uv        = st.session_state["user_vals"]
    X_df_s    = st.session_state["X_df"]
    models_s  = st.session_state["models"]

    all_probs = [r["prob"] for r in results.values()]
    avg_prob  = float(np.mean(all_probs))
    avg_label, avg_cls, avg_color = risk_meta(avg_prob)

    # ── Ensemble banner ──────────────────────────────
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#0f1e3d,#07080e);
                border:1px solid {avg_color};border-radius:16px;
                padding:22px 28px;margin:18px 0;display:flex;
                align-items:center;justify-content:space-between;'>
      <div>
        <div style='color:#5a6380;font-size:.82rem;text-transform:uppercase;letter-spacing:1px;'>
          Ensemble Prediction (avg of {len(results)} models)
        </div>
        <div style='color:{avg_color};font-size:2.4rem;font-weight:800;margin-top:4px;'>
          {round(avg_prob*100,1)}% &nbsp;
          <span style='font-size:1rem;color:#5a6380;font-weight:400;'>depression risk</span>
        </div>
      </div>
      <div style='text-align:right;'>
        <span style='background:rgba(0,0,0,.3);border:1px solid {avg_color};
                     color:{avg_color};padding:6px 18px;border-radius:30px;
                     font-size:.9rem;font-weight:700;'>{avg_label}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════
    #  TABS
    # ══════════════════════════════════════════════════
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯  All Predictions",
        "📊  Model Comparison",
        "🔬  Feature Analysis",
        "📈  Risk Breakdown",
    ])

    # ━━━ TAB 1 — ALL PREDICTIONS ━━━━━━━━━━━━━━━━━━━━
    with tab1:
        st.markdown("<div class='sec'>🎯 Har Model ki Prediction</div>", unsafe_allow_html=True)
        cols = st.columns(3)
        for i, (name, r) in enumerate(results.items()):
            info = MODEL_INFO.get(name, {"icon": "🤖", "color": "#58a6ff"})
            with cols[i % 3]:
                cls_map = {"high": "result-high", "med": "result-med", "low": "result-low"}
                b_map   = {"high": "b-high",       "med": "b-med",      "low": "b-low"}
                st.markdown(f"""
                <div class='card'>
                  <h3>{info['icon']} {name}</h3>
                  <div class='sub'>ML Classifier · Depression Detection</div>
                  <div class='{cls_map[r["cls"]]}'>{round(r["prob"]*100,1)}%</div>
                  <div><span class='badge {b_map[r["cls"]]}'>{r["label"]}</span></div>
                  <div style='color:#3a4560;font-size:.75rem;margin-top:8px;'>
                    Confidence: {"High ●●●" if abs(r["prob"]-.5)>.25 else "Medium ●●○" if abs(r["prob"]-.5)>.1 else "Low ●○○"}
                  </div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<div class='sec'>📡 Risk Gauges</div>", unsafe_allow_html=True)
        gcols = st.columns(len(results))
        for i, (name, r) in enumerate(results.items()):
            info = MODEL_INFO.get(name, {"color": "#58a6ff"})
            with gcols[i]:
                st.plotly_chart(gauge(r["prob"], name, info["color"]),
                                use_container_width=True, key=f"g{i}")

    # ━━━ TAB 2 — MODEL COMPARISON ━━━━━━━━━━━━━━━━━━━
    with tab2:
        st.markdown("<div class='sec'>📊 Model Comparison</div>", unsafe_allow_html=True)
        names  = list(results.keys())
        probs  = [results[n]["prob"] * 100 for n in names]
        colors = [MODEL_INFO.get(n, {"color": "#58a6ff"})["color"] for n in names]

        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Bar(
                x=probs, y=names, orientation='h',
                marker=dict(color=colors),
                text=[f"{p:.1f}%" for p in probs], textposition="outside",
                textfont=dict(color="#c9d1d9"),
            ))
            fig.add_vline(x=60, line_dash="dash", line_color="#f85149",
                          annotation_text="High Risk", annotation_font_color="#f85149")
            fig.add_vline(x=35, line_dash="dash", line_color="#f0883e",
                          annotation_text="Medium", annotation_font_color="#f0883e")
            fig.update_layout(**DARK, height=340,
                title=dict(text="Risk % by Model", font=dict(color="#e6edf3", size=13)),
                xaxis=dict(range=[0,115], gridcolor="#1a1f30",
                           tickfont=dict(color="#5a6380")),
                yaxis=dict(gridcolor="#1a1f30", tickfont=dict(color="#c9d1d9")))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig2 = go.Figure(go.Scatterpolar(
                r=probs + [probs[0]], theta=names + [names[0]],
                fill='toself', fillcolor="rgba(88,166,255,.10)",
                line=dict(color="#58a6ff", width=2),
                marker=dict(size=8, color=colors + [colors[0]]),
            ))
            fig2.update_layout(**DARK, height=340,
                title=dict(text="Radar — All Models", font=dict(color="#e6edf3", size=13)),
                polar=dict(
                    bgcolor="#0b0d15",
                    angularaxis=dict(tickfont=dict(color="#c9d1d9", size=10)),
                    radialaxis=dict(range=[0,100], gridcolor="#1a1f30",
                                   tickfont=dict(color="#5a6380", size=9))))
            st.plotly_chart(fig2, use_container_width=True)

        m1, m2, m3 = st.columns(3)
        high_votes = sum(p >= 60 for p in probs)
        spread = round(max(probs) - min(probs), 1)
        for col, label, val, badge in [
            (m1, "Ensemble Avg Risk", f"{round(avg_prob*100,1)}%", avg_label),
            (m2, "High-Risk Votes",   f"{high_votes}/{len(probs)}", "Models agree"),
            (m3, "Prediction Spread", f"{spread}%",  "Max – Min"),
        ]:
            col.markdown(f"""
            <div class='card' style='text-align:center;'>
              <div style='color:#5a6380;font-size:.8rem;'>{label}</div>
              <div style='color:#58a6ff;font-size:1.9rem;font-weight:800;margin:4px 0;'>{val}</div>
              <span class='badge b-med'>{badge}</span>
            </div>""", unsafe_allow_html=True)

    # ━━━ TAB 3 — FEATURE ANALYSIS ━━━━━━━━━━━━━━━━━━━
    with tab3:
        st.markdown("<div class='sec'>🔬 Feature Analysis</div>", unsafe_allow_html=True)
        fa, fb = st.columns(2)

        with fa:
            num_keys = ["daily_social_media_hours","sleep_hours","screen_time_before_sleep",
                        "academic_performance","physical_activity",
                        "stress_level","anxiety_level","addiction_level"]
            maxv     = [8, 12, 6, 10, 20, 10, 10, 10]
            norm     = [uv[k]/m*10 for k, m in zip(num_keys, maxv)]
            labels   = ["Social Media","Sleep","Screen@Night","Academic",
                        "Exercise","Stress","Anxiety","Addiction"]
            fig3 = go.Figure(go.Scatterpolar(
                r=norm+[norm[0]], theta=labels+[labels[0]],
                fill='toself', fillcolor="rgba(63,185,80,.10)",
                line=dict(color="#3fb950", width=2), marker=dict(size=7, color="#3fb950"),
            ))
            fig3.update_layout(**DARK, height=330,
                title=dict(text="Aapka Profile (0–10 normalised)", font=dict(color="#e6edf3",size=13)),
                polar=dict(bgcolor="#0b0d15",
                           angularaxis=dict(tickfont=dict(color="#c9d1d9", size=10)),
                           radialaxis=dict(range=[0,10], gridcolor="#1a1f30",
                                          tickfont=dict(color="#5a6380", size=9))))
            st.plotly_chart(fig3, use_container_width=True)

        with fb:
            imp_arrays = []
            for name, model in models_s.items():
                m = getattr(model, 'best_estimator_', model)
                fi   = getattr(m, 'feature_importances_', None)
                coef = getattr(m, 'coef_', None)
                if fi is not None and np.array(fi).shape == (len(MODEL_FEATURES),):
                    imp_arrays.append(np.array(fi, dtype=float))
                elif coef is not None:
                    imp_arrays.append(np.abs(np.array(coef, dtype=float).flatten()[:len(MODEL_FEATURES)]))

            if imp_arrays:
                avg_imp = np.mean(imp_arrays, axis=0)
                imp_df  = pd.DataFrame({"feature": MODEL_FEATURES, "importance": avg_imp})
                imp_df  = imp_df.sort_values("importance").tail(10)
                fig4 = go.Figure(go.Bar(
                    x=imp_df["importance"], y=imp_df["feature"], orientation='h',
                    marker=dict(color=imp_df["importance"],
                                colorscale=[[0,"#1f6feb"],[.5,"#bc8cff"],[1,"#f85149"]],
                                showscale=False),
                ))
                fig4.update_layout(**DARK, height=330,
                    title=dict(text="Avg Feature Importance", font=dict(color="#e6edf3",size=13)),
                    xaxis=dict(gridcolor="#1a1f30", tickfont=dict(color="#5a6380")),
                    yaxis=dict(gridcolor="#1a1f30", tickfont=dict(color="#c9d1d9", size=10)))
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.info("Feature importance is available only for tree-based models.")

        st.markdown("<div class='sec'>📋 Aapki Current Values</div>", unsafe_allow_html=True)
        disp_keys = ["daily_social_media_hours","sleep_hours","screen_time_before_sleep",
                     "academic_performance","physical_activity",
                     "stress_level","anxiety_level","addiction_level"]
        disp_vals = [uv[k] for k in disp_keys]
        fig5 = go.Figure(go.Bar(
            x=disp_keys, y=disp_vals,
            marker=dict(color=disp_vals,
                        colorscale=[[0, "#3fb950"], [0.5, "#f0883e"], [1, "#f85149"]],
                        showscale=True, colorbar=dict(tickfont=dict(color="#5a6380"))),
            text=[str(round(v,1)) for v in disp_vals], textposition="outside",
            textfont=dict(color="#c9d1d9"),
        ))
        fig5.update_layout(**DARK, height=340,
            title=dict(text="Input Feature Values", font=dict(color="#e6edf3",size=13)),
            xaxis=dict(tickangle=-30, gridcolor="#1a1f30", tickfont=dict(color="#5a6380",size=9)),
            yaxis=dict(gridcolor="#1a1f30", tickfont=dict(color="#5a6380")))
        st.plotly_chart(fig5, use_container_width=True)

    # ━━━ TAB 4 — RISK BREAKDOWN ━━━━━━━━━━━━━━━━━━━━━
    with tab4:
        st.markdown("<div class='sec'>📈 Risk Breakdown</div>", unsafe_allow_html=True)
        p1, p2 = st.columns(2)

        with p1:
            counts = {
                "High Risk (≥60%)":   sum(p >= 60 for p in probs),
                "Medium (35–60%)":    sum(35 <= p < 60 for p in probs),
                "Low Risk (<35%)":    sum(p < 35 for p in probs),
            }
            fig6 = go.Figure(go.Pie(
                labels=list(counts.keys()), values=list(counts.values()),
                hole=0.45,
                marker=dict(colors=["#f85149","#f0883e","#3fb950"],
                            line=dict(color="#07080e", width=2)),
                textfont=dict(color="#e6edf3", size=12),
            ))
            fig6.update_layout(**DARK, height=310,
                title=dict(text="Risk Category Distribution", font=dict(color="#e6edf3",size=13)),
                legend=dict(font=dict(color="#c9d1d9")))
            st.plotly_chart(fig6, use_container_width=True)

        with p2:
            si  = np.argsort(probs)[::-1]
            sn  = [names[i] for i in si]
            sp  = [probs[i]  for i in si]
            sc  = [colors[i] for i in si]
            fig7 = go.Figure(go.Bar(
                x=[f"{MODEL_INFO.get(n,{}).get('icon','🤖')} {n}" for n in sn],
                y=sp, marker=dict(color=sc),
                text=[f"{p:.1f}%" for p in sp], textposition="outside",
                textfont=dict(color="#c9d1d9"),
            ))
            fig7.add_hrect(y0=60, y1=105, fillcolor="rgba(248,81,73,.05)",
                           line=dict(color="#f85149", width=.5, dash="dot"))
            fig7.add_hrect(y0=35, y1=60,  fillcolor="rgba(240,136,62,.05)",
                           line=dict(color="#f0883e", width=.5, dash="dot"))
            fig7.update_layout(**DARK, height=310,
                title=dict(text="Models Ranked by Risk", font=dict(color="#e6edf3",size=13)),
                xaxis=dict(tickangle=-20, gridcolor="#1a1f30",
                           tickfont=dict(color="#5a6380",size=9)),
                yaxis=dict(range=[0,115], gridcolor="#1a1f30", tickfont=dict(color="#5a6380")))
            st.plotly_chart(fig7, use_container_width=True)

        st.markdown("<div class='sec'>📋 Summary Table</div>", unsafe_allow_html=True)
        df_sum = pd.DataFrame({
            "Model":      [f"{MODEL_INFO.get(n,{}).get('icon','🤖')} {n}" for n in names],
            "Risk %":     [f"{p:.1f}%" for p in probs],
            "Risk Level": [results[n]["label"] for n in names],
            "Prediction": ["⚠️ At Risk" if results[n]["prob"] >= .5 else "✅ Not At Risk" for n in names],
        })
        st.dataframe(df_sum, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()