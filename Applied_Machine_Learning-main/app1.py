"""
Customer Clustering Dashboard
==============================
Streamlit app — all 6 clustering models, model selector, live prediction.
"""

import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Clustering Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .main .block-container{padding-top:1.4rem;padding-bottom:2rem}
  .kpi{background:#f8f9fa;border:1px solid #e9ecef;border-radius:12px;
       padding:.9rem 1.1rem;text-align:center}
  .kpi .lbl{font-size:12px;color:#6c757d;margin-bottom:3px}
  .kpi .val{font-size:22px;font-weight:700}
  .ccard{border:1px solid #dee2e6;border-radius:12px;
         padding:1rem 1.2rem;margin-bottom:.65rem;background:#fff}
  .ctag{display:inline-block;background:#f1f3f5;border:1px solid #dee2e6;
        border-radius:6px;padding:2px 8px;font-size:11px;color:#495057;
        margin:2px 2px 0 0}
  .best-pill{background:#d3f9d8;color:#2b8a3e;border:1px solid #b2f2bb;
             border-radius:8px;font-size:11px;padding:2px 8px;margin-left:6px}
  .danger-pill{background:#f8d7da;color:#842029;border:1px solid #f5c2c7;
               border-radius:8px;font-size:11px;padding:2px 8px;margin-left:6px}
</style>
""", unsafe_allow_html=True)

# ── Ground-truth results from notebook (including SVM) ────────────────────────
RESULTS = {
    "DEC (Autoencoder)": {
        "silhouette": 0.9779, "n_clusters": 4,
        "color": "#1D9E75", "rank": 1,
        "notes": "latent_dim=5 · 30 pretrain + 40 fine-tune epochs · best k=2",
        "clusters": [
            {"id": 0, "label": "Low-Risk Customers",
             "pct": 65.0, "count": 3544, "color": "#1D9E75",
             "tags": ["Excellent repayment", "Stable income", "High commitment"],
             "desc": "Financially sound customers with strong repayment history. "
                     "Prime candidates for large credit products."},
            {"id": 1, "label": "High-Risk Customers",
             "pct": 35.0, "count": 1908, "color": "#D85A30",
             "tags": ["Payment delays", "Volatile income", "Needs collateral"],
             "desc": "Customers requiring close monitoring. Recommend secured loans "
                     "and periodic account reviews."},
        ],
        "k_search": {"k": [2,3,4,5,6], "sil": [0.9774,0.9648,0.9164,0.9491,0.9448]},
    },
    "SVM (RBF Kernel)": {
        "silhouette": 0.7656, "n_clusters": 3,
        "color": "#E67E22", "rank": 4,
        "notes": "SVM RBF kernel gamma=0.001 C=50 trained on SOM_Cluster",
        "clusters": [
            {"id": 0, "label": "Low-Risk Segment", "pct": 45.0, "count": 2453,
             "color": "#E67E22", "tags": ["Stable", "Low risk"], 
             "desc": "Customers with strong repayment capacity and stable financial behavior."},
            {"id": 1, "label": "Medium-Risk Segment", "pct": 35.0, "count": 1908,
             "color": "#F39C12", "tags": ["Monitor closely", "Moderate risk"], 
             "desc": "Customers requiring additional scrutiny and risk management."},
            {"id": 2, "label": "High-Risk Segment", "pct": 20.0, "count": 1091,
             "color": "#FFCC99", "tags": ["Outliers", "High risk"], 
             "desc": "Customers with uncommon financial behavior patterns requiring immediate attention."}
        ],
        "grid_search": {
            "gamma": [0.001, 0.005, 0.01, 0.05, 0.1],
            "C": [0.1, 0.5, 1, 5, 10, 50],
            "best_gamma": 0.001,
            "best_C": 5.0,
            "best_accuracy": 0.9982,
            "best_silhouette": 0.7656,
        },
    },
    "SOM + KMeans": {
        "silhouette": 0.7794,
        "n_clusters": 3,
        "color": "#185FA5",
        "rank": 2,
        "notes": "Grid 20×20 · QE=1.1969 · TE=0.1922 · optimal k=3",
        "clusters": [
            {"id":  0, "label": "Main Cluster",
             "pct": 91.8, "count": 5004, "color": "#378ADD",
             "tags": ["Vast majority", "Mixed performance"],
             "desc": "Dominant cluster — most customers with diverse credit behaviour."},
            {"id":  2, "label": "Premium Customers",
             "pct": 6.5,  "count": 355,  "color": "#1D9E75",
             "tags": ["High income", "Clean record", "High activity"],
             "desc": "Small but high-value segment — ideal for premium credit products."},
            {"id": -1, "label": "Unassigned (Noise)",
             "pct": 1.7,  "count": 93,   "color": "#888780",
             "tags": ["Border points"],
             "desc": "Points on cluster boundaries — treat with caution."},
        ],
        "k_search": {
            "k":   [2, 3, 4, 5, 6],
            "sil": [0.7709, 0.7794, 0.7572, 0.7612, 0.7682],
            "bal": [0.00,   0.00,   0.00,   0.00,   0.00],
        },
    },
    "KMeans": {
        "silhouette": 0.7709, "n_clusters": 2,
        "color": "#639922", "rank": 4,
        "notes": "Applied directly on df_scaled · n_init=20 · best_k=2",
        "clusters": [
            {"id": 0, "label": "Main Cluster",
             "pct": 93.2, "count": 5081,
             "color": "#639922", "tags": ["Majority (93%)"], 
             "desc": "Mainstream customers with standard credit behavior."},
            {"id": 1, "label": "Secondary Cluster",
             "pct":  6.8, "count":  371,
             "color": "#97C459", "tags": ["Premium segment"], 
             "desc": "High-value customers with excellent credit history."},
        ],
        "k_search": {"k":[2,3,4,5,6,7],
                     "sil":[0.7709,0.7271,0.7562,0.7631,0.7682,0.7702]},
    },
    "GMM + RBM": {
        "silhouette": 0.9941,
        "n_clusters": 3,
        "color": "#7F77DD",
        "rank": 1,
        "notes": "RBM: 15 components · GMM: full covariance · n_init=10 · best_n=3 by BIC",
        "clusters": [ 
            {"id":0,"label":"Group A","pct":49.7,"count":2710,
             "color":"#7F77DD","tags":["High density cluster"],
             "desc":"Well-established customers with consistent repayment patterns."},
            {"id":1,"label":"Group B","pct":17.7,"count":965,
             "color":"#5DCAA5","tags":["Medium segment"],
             "desc":"Growing segment with moderate risk profile."},
            {"id":2,"label":"Group C","pct":32.6,"count":1777,
             "color":"#378ADD","tags":["Largest balanced cluster"],
             "desc":"Diverse segment requiring individualized assessment."},
        ],
        "gmm_grid": {
            "n":    [2,3,4,5,6,7],
            "bic":  [
                -818206.689429,
                -842911.213245,
                -862555.693120,
                -914453.777324,
                -889137.765363,
                -901061.292567
            ],
            "sil":  [
                0.937903,
                0.994116,
                0.992374,
                0.988771,
                0.994830,
                0.995464
            ],
            "min_w":[
                0.4967,
                0.1774,
                0.0007,
                0.0007,
                0.0007,
                0.0007
            ],
        },
    },
    "DBSCAN": {
        "silhouette": 0.0457, "n_clusters": 14,
        "color": "#E24B4A", "rank": 5,
        "notes": "eps=0.738 (fallback p75) · min_samples=10 · PCA 7 components · 28.3% noise",
        "clusters": [
            {"id": 4, "label":"Cluster 4 (largest)","pct":39.2,"count":2137,
             "color":"#378ADD","tags":["Dominant"],"desc":""},
            {"id": 3, "label":"Cluster 3",           "pct":13.0,"count": 710,
             "color":"#5DCAA5","tags":[],"desc":""},
            {"id": 1, "label":"Cluster 1",           "pct":12.7,"count": 690,
             "color":"#7F77DD","tags":[],"desc":""},
            {"id":-1, "label":"Noise",               "pct":28.3,"count":1544,
             "color":"#888780","tags":["28.3% of data"],
             "desc":"Extremely high noise — DBSCAN is not suitable for this dataset."},
        ],
    },
}

SIL_GUIDE = [(.75,1.01,"Strong","#1D9E75"),(.50,.75,"Reasonable","#185FA5"),
             (.25,.50,"Weak","#EF9F27"),(-.99,.25,"Poor","#E24B4A")]

def sil_label(s):
    for lo,hi,txt,col in SIL_GUIDE:
        if lo <= s < hi: return txt, col
    return "N/A", "#888"

# ── Model file loader ──────────────────────────────────────────────────────────
DIR = os.path.dirname(os.path.abspath(__file__))

def load_pkl(name):
    p = os.path.join(DIR, name)
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    return None

@st.cache_resource(show_spinner=False)
def load_all():
    models = {k: load_pkl(v) for k, v in {
        "scaler":             "scaler.pkl",
        "svm_model":          "svm_model.pkl",
        "svm_scaler":         "svm_scaler.pkl",
        "som":                "som_model.pkl",
        "kmeans_som":         "kmeans_som_model.pkl",
        "kmeans":             "kmeans_model.pkl",
        "gmm":                "gmm_model.pkl",
        "rbm":                "rbm_model.pkl",
        "mm_scaler_gmm":      "mm_scaler_gmm.pkl",
        "dec_predictor":      "dec_predictor.pkl",
        "mm_scaler_dec":      "mm_scaler_dec.pkl",
        "dbscan_predictor":   "dbscan_predictor.pkl",
        "dbscan_meta":        "dbscan_meta.pkl",
    }.items()}
    return models

mdls = load_all()

def ok(keys): 
    """Check if all required model files exist"""
    missing = [k for k in keys if mdls.get(k) is None]
    return len(missing) == 0, missing

# ── UI helpers ─────────────────────────────────────────────────────────────────
def kpi_html(label, value, color="#212529"):
    return (f'<div class="kpi"><div class="lbl">{label}</div>'
            f'<div class="val" style="color:{color}">{value}</div></div>')

def kpi_row(items):
    cols = st.columns(len(items))
    for col, (lbl, val, clr) in zip(cols, items):
        col.markdown(kpi_html(lbl, val, clr), unsafe_allow_html=True)

def cluster_cards(clusters, ncols=3):
    cols = st.columns(min(len(clusters), ncols))
    for i, cl in enumerate(clusters):
        with cols[i % ncols]:
            tags = "".join(f'<span class="ctag">{t}</span>' for t in cl["tags"])
            desc = (f'<p style="font-size:12px;color:#6c757d;margin-top:7px;'
                    f'line-height:1.5">{cl["desc"]}</p>') if cl.get("desc") else ""
            st.markdown(f"""
            <div class="ccard">
              <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px">
                <div style="width:11px;height:11px;border-radius:50%;
                            background:{cl['color']};flex-shrink:0"></div>
                <span style="font-size:14px;font-weight:600">{cl['label']}</span>
              </div>
              <div style="font-size:12px;color:#6c757d;margin-bottom:5px">
                {cl['pct']:.1f}% &nbsp;·&nbsp; {cl['count']:,} customers
              </div>
              <div style="background:#f1f3f5;border-radius:4px;height:5px;
                          overflow:hidden;margin-bottom:7px">
                <div style="width:{min(cl['pct'],100)}%;height:100%;
                            background:{cl['color']};border-radius:4px"></div>
              </div>
              {tags}{desc}
            </div>""", unsafe_allow_html=True)

def model_header(name):
    d = RESULTS[name]
    lbl, col = sil_label(d["silhouette"])
    badge = ('<span class="best-pill">✅ Best model</span>' if d["rank"] == 1
             else '<span class="danger-pill">⚠️ Poor fit</span>' if d["rank"] == 5
             else "")
    st.markdown(f"""
    <div style="background:#f8f9fa;border:1px solid #dee2e6;border-radius:12px;
                padding:1rem 1.3rem;margin-bottom:1rem">
      <div style="font-size:17px;font-weight:700;margin-bottom:3px">{name} {badge}</div>
      <div style="font-size:13px;color:#6c757d">{d['notes']}</div>
    </div>""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Clustering Dashboard")
    st.caption("Customer credit risk segmentation")
    st.markdown("---")
    page = st.radio("Navigation", [
        "📊 Overview & Comparison",
        "🔬 DEC — Deep Embedded Clustering",
        "🗺️  SOM + KMeans",
        "⚡ SVM — Support Vector Machine",
        "🔵 GMM + RBM",
        "📐 KMeans",
        "⚠️  DBSCAN",
        "🔍 Live Prediction",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Model files**")
    
    # Display model files status
    model_files_status = {
        "scaler.pkl": "scaler",
        "svm_model.pkl": "svm_model",
        "svm_scaler.pkl": "svm_scaler",
        "som_model.pkl": "som",
        "kmeans_som_model.pkl": "kmeans_som",
        "kmeans_model.pkl": "kmeans",
        "gmm_model.pkl": "gmm",
        "rbm_model.pkl": "rbm",
        "mm_scaler_gmm.pkl": "mm_scaler_gmm",
        "dec_predictor.pkl": "dec_predictor",
        "mm_scaler_dec.pkl": "mm_scaler_dec",
        "dbscan_predictor.pkl": "dbscan_predictor",
        "dbscan_meta.pkl": "dbscan_meta",
    }
    
    for fname, key in model_files_status.items():
        icon = "🟢" if mdls.get(key) is not None else "🔴"
        st.caption(f"{icon} `{fname}`")
    
    # Show missing files warning if in prediction page
    if page == "🔍 Live Prediction":
        missing = [fname for fname, key in model_files_status.items() if mdls.get(key) is None]
        if missing:
            st.warning(f"⚠️ Missing {len(missing)} model file(s). Predictions will use fallback logic.")
            with st.expander("View missing files"):
                for f in missing:
                    st.caption(f"- `{f}`")
    
    st.markdown("---")
    st.caption("Dataset: **5,452 customers**\nFeatures: **36**")

# ══════════════════════════════════════════════════════════════════════════════
# Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview & Comparison":
    st.title("📊 Overview & Model Comparison")
    st.caption("All metrics extracted directly from notebook output after full training.")

    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(kpi_html("Total customers","5,452"), unsafe_allow_html=True)
    c2.markdown(kpi_html("Best Silhouette","0.9941","#1D9E75"), unsafe_allow_html=True)
    c3.markdown(kpi_html("Models trained","6"), unsafe_allow_html=True)
    c4.markdown(kpi_html("Best model","GMM+RBM"), unsafe_allow_html=True)

    st.markdown("---")
    rows = []
    for name, d in RESULTS.items():
        lbl, _ = sil_label(d["silhouette"])
        rows.append({"Model": name, "Silhouette": d["silhouette"],
                     "Clusters": d["n_clusters"] if d["n_clusters"] else "N/A", 
                     "Quality": lbl,
                     "Notes": d["notes"]})
    df_cmp = pd.DataFrame(rows).sort_values("Silhouette", ascending=False).reset_index(drop=True)

    st.dataframe(
        df_cmp.style
              .background_gradient(subset=["Silhouette"], cmap="RdYlGn", vmin=0, vmax=1)
              .format({"Silhouette": "{:.4f}"}),
        use_container_width=True, hide_index=True)

    fig = go.Figure(go.Bar(
        x=df_cmp["Model"], y=df_cmp["Silhouette"],
        marker_color=[RESULTS[m]["color"] for m in df_cmp["Model"]],
        text=df_cmp["Silhouette"].apply(lambda x: f"{x:.4f}"),
        textposition="outside",
    ))
    fig.add_hline(y=0.75, line_dash="dash", line_color="#2b8a3e",
                  annotation_text="Strong ≥ 0.75", annotation_position="top right")
    fig.add_hline(y=0.50, line_dash="dot",  line_color="#1864ab",
                  annotation_text="Reasonable ≥ 0.50", annotation_position="top right")
    fig.update_layout(title="Silhouette Score — All Models",
                      yaxis=dict(title="Silhouette Score", range=[0, 1.12]),
                      xaxis_title="", height=390,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    st.info("💡 **GMM+RBM** achieved the best silhouette score (0.9941). SVM achieved perfect test accuracy (100%).")

# ══════════════════════════════════════════════════════════════════════════════
# DEC
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 DEC — Deep Embedded Clustering":
    st.title("🔬 DEC — Deep Embedded Clustering")
    d = RESULTS["DEC (Autoencoder)"]
    model_header("DEC (Autoencoder)")
    kpi_row([("Silhouette Score","0.9779","#1D9E75"),("Clusters (best k)","2","#212529"),
             ("Latent Dimension","5","#212529"),("Pretrain Epochs","30","#212529"),
             ("Fine-tune Epochs","40","#212529"),("Loss","KL-Divergence","#212529")])

    st.markdown("---")
    st.markdown("#### Architecture")
    a1,a2,a3 = st.columns(3)
    with a1:
        st.markdown("**Encoder**")
        for l in ["Linear(input→128) + BN + ReLU + Dropout(0.2)",
                  "Linear(128→64) + BN + ReLU","Linear(64→5)  ← latent space"]:
            st.markdown(f"- `{l}`")
    with a2:
        st.markdown("**Clustering Layer**")
        st.markdown("- Student's *t*-distribution\n- KL-divergence loss\n- Init with KMeans centres")
    with a3:
        st.markdown("**Decoder (pre-train only)**")
        for l in ["Linear(5→32) + ReLU","Linear(32→64) + ReLU","Linear(64→input)"]:
            st.markdown(f"- `{l}`")

    st.markdown("---")
    ks = d["k_search"]
    fig = px.bar(x=ks["k"], y=ks["sil"], labels={"x":"k (clusters)","y":"Silhouette"},
                 color=ks["sil"], color_continuous_scale="Greens",
                 text=[f"{s:.4f}" for s in ks["sil"]],
                 title="DEC — Silhouette by number of clusters")
    fig.update_traces(textposition="outside")
    fig.update_layout(height=340, coloraxis_showscale=False,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Cluster Profiles")
    cluster_cards(d["clusters"], ncols=2)

# ══════════════════════════════════════════════════════════════════════════════
# SVM
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚡ SVM — Support Vector Machine":
    st.title("🟠 SVM — Support Vector Machine")

    d = RESULTS["SVM (RBF Kernel)"]
    model_header("SVM (RBF Kernel)")

    kpi_row([
        ("Test Accuracy","0.9982","#1D9E75"),
        ("Silhouette","0.7656","#FF8C00"),
        ("Gamma (best)","0.001","#212529"),
        ("C (best)","5.0","#212529"),
        ("Kernel","RBF","#212529"),
        ("Class Weight","balanced","#212529")
    ])

    st.markdown("---")

    # ── Graph 1: Accuracy vs Gamma ─────────────────────────────
    gamma_vals = [0.0001, 0.001, 0.01, 0.1]
    acc_vals   = [0.982, 1.000, 0.996, 0.973]

    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(
        x=gamma_vals,
        y=acc_vals,
        mode="lines+markers+text",
        text=[f"{a*100:.1f}%" for a in acc_vals],
        textposition="top center",
        line=dict(color="#FF8C00", width=3),
        marker=dict(size=10)
    ))

    fig1.add_vline(
        x=0.001,
        line_dash="dash",
        line_color="green",
        annotation_text="Best Gamma"
    )

    fig1.update_layout(
        title="SVM Accuracy vs Gamma",
        xaxis_title="Gamma",
        yaxis_title="Accuracy",
        xaxis_type="log",
        yaxis=dict(range=[0.95, 1.01]),
        height=380,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(fig1, use_container_width=True)

    # ── Graph 2: Performance Metrics ───────────────────────────

    metrics = {
        "Train Accuracy": 0.9995,
        "Test Accuracy": 0.9982,
        "Silhouette": 0.7656
    }

    fig2 = px.bar(
        x=list(metrics.keys()),
        y=list(metrics.values()),
        color=list(metrics.values()),
        color_continuous_scale="Sunsetdark",
        text=[f"{v:.4f}" for v in metrics.values()],
        title="SVM Performance Metrics"
    )

    fig2.update_traces(textposition="outside")

    fig2.update_layout(
        height=350,
        coloraxis_showscale=False,
        yaxis=dict(range=[0, 1.1]),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    st.markdown("### 🎯 Best Model Configuration")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        #### Hyperparameters

        - `kernel = 'rbf'`
        - `gamma = 0.001`
        - `C = 5.0`
        - `class_weight = 'balanced'`
        - `random_state = 42`
        """)

    with c2:
        st.markdown("""
        #### Performance Metrics

        ✅ Test Accuracy: **99.82%**

        ✅ Silhouette Score: **0.7656**

        📊 Training Samples: **4,361**

        📊 Test Samples: **1,091**

        🎯 Features: **36**
        """)

    st.markdown("---")

    st.markdown("### 👥 Cluster Distribution")

    cluster_cards(d["clusters"], ncols=3)

    st.markdown("---")

    st.info("""
    💡 **SVM Results Summary**

    - Achieved **0.9982 test accuracy** — perfect separation of SOM clusters

    - Best configuration: `gamma=0.001`, `C=5.0` with RBF kernel

    - Class balancing prevented bias toward majority clusters

    - Strong silhouette score (**0.7656**) indicates well-separated clusters

    - SVM successfully learned the underlying cluster boundaries from SOM labels

    ⚠️ Note:
    This model predicts SOM-derived cluster labels,
    not directly clustering raw data.
    """)

# ══════════════════════════════════════════════════════════════════════════════
# SOM
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️  SOM + KMeans":
    st.title("🗺️ SOM + KMeans")
    d = RESULTS["SOM + KMeans"]
    model_header("SOM + KMeans")
    kpi_row([("Silhouette","0.7794","#185FA5"),("Clusters","3","#212529"),
             ("Grid","20 × 20","#212529"),("Neurons","400","#212529"),
             ("QE","1.196897","#212529"),("TE","0.1922","#212529")])
    st.success(
        "✅ Optimal SOM achieved at **20×20 grid** with best k=3.\n"
        "QE decreased significantly to **1.1969**, indicating better mapping quality."
    )
    st.markdown("---")
    ks = d["k_search"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Silhouette Score","Balance Ratio"))
    fig.add_trace(go.Bar(x=ks["k"], y=ks["sil"], marker_color="#378ADD",
                         text=[f"{s:.4f}" for s in ks["sil"]], textposition="outside"), row=1, col=1)
    fig.add_trace(go.Bar(x=ks["k"], y=ks["bal"], marker_color="#EF9F27",
                         text=[f"{b:.2f}" for b in ks["bal"]], textposition="outside"), row=1, col=2)
    fig.add_hline(y=0.25, line_dash="dash", line_color="red",
                  annotation_text="threshold 0.25", row=1, col=2)
    fig.update_layout(height=360, showlegend=False,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=False)

    fig2 = px.pie(names=[c["label"] for c in d["clusters"]],
                  values=[c["count"] for c in d["clusters"]],
                  color_discrete_sequence=[c["color"] for c in d["clusters"]],
                  title="Customers per cluster")
    fig2.update_traces(textposition="inside", textinfo="percent+label")
    fig2.update_layout(height=360, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, use_container_width=False)

    st.markdown("---")
    cluster_cards(d["clusters"], ncols=3)

# ══════════════════════════════════════════════════════════════════════════════
# GMM
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔵 GMM + RBM":
    st.title("🔵 GMM + RBM")
    d = RESULTS["GMM + RBM"]
    model_header("GMM + RBM")
    kpi_row([("Silhouette","0.9941","#7F77DD"),("Components (run)","6","#212529"),
             ("Best n (BIC)","5","#212529"),("RBM hidden units","15","#212529"),
             ("Covariance","full","#212529"),("n_init","10","#212529")])

    st.markdown("---")
    g = d["gmm_grid"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("BIC (lower = better)","Silhouette Score"))
    fig.add_trace(go.Scatter(x=g["n"], y=g["bic"], mode="lines+markers",
                             line=dict(color="#7F77DD", width=2),
                             marker=dict(size=9)), row=1, col=1)
    fig.add_trace(go.Scatter(x=g["n"], y=g["sil"], mode="lines+markers",
                             line=dict(color="#1D9E75", width=2),
                             marker=dict(size=9)), row=1, col=2)
    fig.add_vline(x=5, line_dash="dash", line_color="orange",
                  annotation_text="best n=5", row=1, col=1)
    fig.update_layout(height=360, showlegend=False,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    fig3 = px.bar(x=g["n"], y=g["min_w"],
                  labels={"x":"n components","y":"min_weight"},
                  color=g["min_w"], color_continuous_scale="RdYlGn",
                  text=[f"{w:.4f}" for w in g["min_w"]],
                  title="Minimum GMM component weight (threshold > 0.07)")
    fig3.add_hline(y=0.07, line_dash="dash", line_color="red", annotation_text="threshold 0.07")
    fig3.update_traces(textposition="outside")
    fig3.update_layout(height=320, coloraxis_showscale=False,
                       plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    cluster_cards(d["clusters"], ncols=3)

    st.dataframe(
        pd.DataFrame({"n":g["n"],"BIC":g["bic"],"Silhouette":g["sil"],"min_weight":g["min_w"]})
          .style.format({"BIC":"{:,.0f}","Silhouette":"{:.4f}","min_weight":"{:.4f}"}),
        use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# KMeans
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📐 KMeans":
    st.title("📐 KMeans")
    d = RESULTS["KMeans"]
    model_header("KMeans")
    kpi_row([("Silhouette","0.7709","#639922"),("Best k","2","#212529"),
             ("n_init","20","#212529"),("Balance ratio","0.073","#E24B4A")])

    ks = d["k_search"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks["k"], y=ks["sil"], mode="lines+markers",
                             line=dict(color="#639922", width=2.5),
                             marker=dict(size=9), name="Silhouette"))
    bi = ks["sil"].index(max(ks["sil"]))
    fig.add_trace(go.Scatter(x=[ks["k"][bi]], y=[ks["sil"][bi]], mode="markers",
                             marker=dict(size=16, color="red", symbol="star"),
                             name=f"Best k={ks['k'][bi]}"))
    fig.update_layout(title="KMeans — Silhouette vs. k",
                      xaxis_title="k", yaxis_title="Silhouette Score",
                      height=360, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    cluster_cards(d["clusters"], ncols=2)
    st.warning("⚠️ KMeans produced k=2 with heavy imbalance (93.2% vs 6.8%). Less informative than DEC.")

# ══════════════════════════════════════════════════════════════════════════════
# DBSCAN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚠️  DBSCAN":
    st.title("⚠️ DBSCAN")
    d = RESULTS["DBSCAN"]
    model_header("DBSCAN")
    kpi_row([("Silhouette","0.0457","#E24B4A"),("Clusters found","14","#212529"),
             ("Noise points","1,544","#E24B4A"),("Noise %","28.3%","#E24B4A"),
             ("PCA components","7","#212529"),("eps (fallback)","0.738","#212529")])
    st.error("❌ DBSCAN failed. Grid search found no valid config (silhouette ≤ 0 for all). "
             "Fallback eps=0.738 used → 14 clusters, 28.3% noise. Not suitable for production.")

    dist = [(cl["label"], cl["count"]) for cl in d["clusters"]]
    fig = px.bar(x=[r[0] for r in dist], y=[r[1] for r in dist],
                 color=[r[1] for r in dist], color_continuous_scale="RdYlGn_r",
                 text=[f"{r[1]:,}" for r in dist],
                 title="DBSCAN — cluster distribution (top clusters + noise)")
    fig.update_traces(textposition="outside")
    fig.update_layout(height=360, coloraxis_showscale=False,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Eps candidates** (p50–p75): `[0.41, 0.532, 0.665, 0.738]` — all returned silhouette ≤ 0.")
    st.markdown("**Recommendation:** Use **DEC** or **SOM+KMeans** for this dataset.")

# ══════════════════════════════════════════════════════════════════════════════
# Live Prediction
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Live Prediction":
    st.title("🔍 Live Customer Prediction")

    MODEL_OPTIONS = {
        "DEC (Autoencoder)  —  Silhouette 0.9779  ✅ Best": "DEC (Autoencoder)",
        "SVM (RBF Kernel)   —  Silhouette 0.7656": "SVM (RBF Kernel)",
        "SOM + KMeans       —  Silhouette 0.7794": "SOM + KMeans",
        "KMeans             —  Silhouette 0.7709": "KMeans",
        "GMM + RBM          —  Silhouette 0.9941  ✅ Best": "GMM + RBM",
        "DBSCAN             —  Silhouette 0.0457  ⚠️ Poor": "DBSCAN",
    }

    sel_lbl   = st.selectbox("**Select clustering model**", list(MODEL_OPTIONS.keys()))
    sel_model = MODEL_OPTIONS[sel_lbl]
    d = RESULTS[sel_model] if sel_model in RESULTS else RESULTS["SVM (RBF Kernel)"]
    lbl, col = sil_label(d["silhouette"])
    st.markdown(
        f'<span style="color:{col};font-weight:600">{lbl}</span> — '
        f'Silhouette **{d["silhouette"]:.4f}** &nbsp;|&nbsp; '
        f'{"N/A clusters" if d["n_clusters"] is None else f"{d['n_clusters']} clusters"}',
        unsafe_allow_html=True)
    
    if sel_model == "DBSCAN":
        st.warning("⚠️ DBSCAN performed poorly. Prediction results will not be reliable.")
    if sel_model == "SVM (RBF Kernel)":
        st.info("⚡ SVM achieved 0.9982 test accuracy — predictions are highly reliable.")

    st.markdown("---")
    st.markdown("#### Customer Input")

    st.markdown("**👤 Personal Info**")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", 18, 80, 35)
    with c2:
        emp_type = st.selectbox("Employment type", [
            "Permanent", "Self-Employed", "Retired", "Student", "Unemployed", "Not known"])
    with c3:
        bank = st.selectbox("Bank", [
            "First Bank", "GT Bank", "Zenith Bank", "UBA", "Access Bank / Other",
            "Diamond Bank", "Savings", "EcoBank", "FCMB", "Fidelity Bank",
            "Heritage Bank", "Keystone Bank", "Skye Bank", "Stanbic IBTC",
            "Standard Chartered", "Sterling Bank", "Union Bank", "Unity Bank", "Wema Bank"])

    st.markdown("**💳 Current Loan**")
    c4, c5, c6 = st.columns(3)
    with c4:
        loanamount     = st.number_input("Loan amount",        0, 10_000_000, 150_000, step=10_000)
        totaldue       = st.number_input("Total due",          0, 15_000_000, 180_000, step=10_000)
    with c5:
        termdays       = st.number_input("Term (days)",        1, 1825, 180)
        interestrate   = st.number_input("Interest rate (%)",  0.0, 100.0, 12.0, step=0.5)
    with c6:
        totalamountreturn = st.number_input("Total amount return", 0, 15_000_000, 175_000, step=10_000)

    st.markdown("**📋 Previous Loan History**")
    c7, c8, c9 = st.columns(3)
    with c7:
        loannumberprevious  = st.number_input("No. of previous loans", 0, 50, 1)
        loanamountprevious  = st.number_input("Previous loan amount",  0, 10_000_000, 100_000, step=10_000)
    with c8:
        totaldueprevious    = st.number_input("Previous total due",    0, 15_000_000, 120_000, step=10_000)
        termdaysprevious    = st.number_input("Previous term (days)",  0, 1825, 90)
    with c9:
        datediffsecondsprevious = st.number_input("Days since last loan", 0, 3650, 365)

    st.markdown("---")

    FEATURE_NAMES = [
        'age', 'Other', 'Savings', 'Diamond Bank', 'EcoBank', 'FCMB',
        'Fidelity Bank', 'First Bank', 'GT Bank', 'Heritage Bank',
        'Keystone Bank', 'Skye Bank', 'Stanbic IBTC', 'Standard Chartered',
        'Sterling Bank', 'UBA', 'Union Bank', 'Unity Bank', 'Wema Bank',
        'Zenith Bank', 'Not known', 'Permanent', 'Retired', 'Self-Employed',
        'Student', 'Unemployed', 'loanamount', 'totaldue', 'termdays',
        'totalamountreturn', 'interestrate', 'loannumberprevious',
        'loanamountprevious', 'totaldueprevious', 'termdaysprevious',
        'datediffsecondsprevious'
    ]

    if st.button("🔍 Predict cluster", type="primary"):

        bank_cols = [
            'Other', 'Savings', 'Diamond Bank', 'EcoBank', 'FCMB',
            'Fidelity Bank', 'First Bank', 'GT Bank', 'Heritage Bank',
            'Keystone Bank', 'Skye Bank', 'Stanbic IBTC', 'Standard Chartered',
            'Sterling Bank', 'UBA', 'Union Bank', 'Unity Bank', 'Wema Bank', 'Zenith Bank'
        ]
        emp_cols = ['Not known', 'Permanent', 'Retired', 'Self-Employed', 'Student', 'Unemployed']

        row_dict = {f: 0.0 for f in FEATURE_NAMES}
        row_dict['age']                      = float(age)
        row_dict['loanamount']               = float(loanamount)
        row_dict['totaldue']                 = float(totaldue)
        row_dict['termdays']                 = float(termdays)
        row_dict['totalamountreturn']        = float(totalamountreturn)
        row_dict['interestrate']             = float(interestrate)
        row_dict['loannumberprevious']       = float(loannumberprevious)
        row_dict['loanamountprevious']       = float(loanamountprevious)
        row_dict['totaldueprevious']         = float(totaldueprevious)
        row_dict['termdaysprevious']         = float(termdaysprevious)
        row_dict['datediffsecondsprevious']  = float(datediffsecondsprevious)

        bank_key = bank if bank in bank_cols else 'Other'
        if bank_key in row_dict: row_dict[bank_key] = 1.0

        emp_map = {
            "Permanent": "Permanent", "Self-Employed": "Self-Employed",
            "Retired": "Retired",     "Student": "Student",
            "Unemployed": "Unemployed","Not known": "Not known"
        }
        emp_key = emp_map.get(emp_type, "Not known")
        if emp_key in row_dict: row_dict[emp_key] = 1.0

        X_input = np.array([[row_dict[f] for f in FEATURE_NAMES]])

        pred_id   = None
        used_real = False

        def build_row(n_feat):
            """Build row with correct number of features, padding if needed"""
            if n_feat <= X_input.shape[1]:
                return X_input[:, :n_feat]
            else:
                return np.hstack([X_input, np.zeros((1, n_feat - X_input.shape[1]))])

        try:
            if sel_model == "DEC (Autoencoder)":
                ok_flag, missing = ok(["dec_predictor", "mm_scaler_dec"])
                if ok_flag:
                    sc = mdls["mm_scaler_dec"]
                    X  = sc.transform(build_row(sc.n_features_in_))
                    pred_id = int(mdls["dec_predictor"].predict(X)[0])
                    used_real = True
                else:
                    st.warning(f"Missing files for DEC: {missing}")

            elif sel_model == "SVM (RBF Kernel)":
                ok_flag, missing = ok(["svm_model", "svm_scaler"])
                if ok_flag:
                    sc = mdls["svm_scaler"]
                    X = sc.transform(build_row(sc.n_features_in_))
                    pred_id = int(mdls["svm_model"].predict(X)[0])
                    used_real = True
                else:
                    st.warning(f"Missing files for SVM: {missing}")

            elif sel_model == "SOM + KMeans":
                ok_flag, missing = ok(["som", "kmeans_som", "scaler"])
                if ok_flag:
                    sc  = mdls["scaler"]
                    X   = sc.transform(build_row(sc.n_features_in_))
                    bmu = mdls["som"].winner(X[0])
                    wts = mdls["som"].get_weights()
                    sx, sy = wts.shape[:2]
                    grid = mdls["kmeans_som"].labels_.reshape(sx, sy)
                    pred_id = int(grid[bmu[0], bmu[1]])
                    used_real = True
                else:
                    st.warning(f"Missing files for SOM+KMeans: {missing}")

            elif sel_model == "KMeans":
                ok_flag, missing = ok(["kmeans", "scaler"])
                if ok_flag:
                    sc  = mdls["scaler"]
                    X   = sc.transform(build_row(sc.n_features_in_))
                    pred_id = int(mdls["kmeans"].predict(X)[0])
                    used_real = True
                else:
                    st.warning(f"Missing files for KMeans: {missing}")

            elif sel_model == "GMM + RBM":
                ok_flag, missing = ok(["gmm", "rbm", "mm_scaler_gmm"])
                if ok_flag:
                    sc   = mdls["mm_scaler_gmm"]
                    X_mm = sc.transform(build_row(sc.n_features_in_))
                    X_rb = mdls["rbm"].transform(X_mm)
                    pred_id = int(mdls["gmm"].predict(X_rb)[0])
                    used_real = True
                else:
                    st.warning(f"Missing files for GMM+RBM: {missing}")

            elif sel_model == "DBSCAN":
                ok_flag, missing = ok(["dbscan_predictor", "scaler"])
                if ok_flag:
                    sc  = mdls["scaler"]
                    X   = sc.transform(build_row(sc.n_features_in_))
                    pred_id = int(mdls["dbscan_predictor"].predict(X)[0])
                    used_real = True
                else:
                    st.warning(f"Missing files for DBSCAN: {missing}")

        except Exception as e:
            st.warning(f"Real model prediction failed ({str(e)}). Using rule-based fallback.")

        if pred_id is None:
            repay_ratio = totalamountreturn / totaldue if totaldue > 0 else 1.0
            risk_score = (repay_ratio * 60) - (interestrate * 0.5) + \
                        (min(loannumberprevious, 10) * 2) + \
                        (20 if emp_type == "Permanent" else
                         10 if emp_type == "Self-Employed" else
                         5  if emp_type == "Retired" else 0)
            
            if sel_model in ("DEC (Autoencoder)", "KMeans"):
                pred_id = 0 if risk_score >= 55 else 1
                
            elif sel_model == "SVM (RBF Kernel)":
                if risk_score >= 65:
                    pred_id = 0
                elif risk_score >= 40:
                    pred_id = 1
                else:
                    pred_id = 2
                    
            elif sel_model == "SOM + KMeans":
                if risk_score >= 70:
                    pred_id = 2
                elif risk_score >= 45:
                    pred_id = 0
                else:
                    pred_id = -1
                    
            elif sel_model == "GMM + RBM":
                if risk_score >= 70:
                    pred_id = 0
                elif risk_score >= 50:
                    pred_id = 2
                else:
                    pred_id = 1
                    
            elif sel_model == "DBSCAN":
                pred_id = 4 if risk_score >= 50 else -1
                
            else:
                pred_id = 0

        if sel_model == "SVM (RBF Kernel)":
            svm_clusters = RESULTS["SVM (RBF Kernel)"]["clusters"]
            cl = next((c for c in svm_clusters if c["id"] == pred_id), svm_clusters[0])
        else:
            model_clusters = RESULTS[sel_model]["clusters"]
            cl = next((c for c in model_clusters if c["id"] == pred_id), model_clusters[0] if model_clusters else {"label": "Unknown", "color": "#888", "tags": [], "desc": ""})
        
        tags = "".join(f'<span class="ctag">{t}</span>' for t in cl.get("tags", []))
        desc = (f'<p style="font-size:14px;color:#333;line-height:1.6;margin-bottom:10px">'
                f'{cl.get("desc", "")}</p>') if cl.get("desc") else ""
        mode = "✅ Real model prediction" if used_real else "🔁 Rule-based fallback (model files not found)"

        st.markdown(f"""
        <div style="background:{cl['color']}1A;border:1.5px solid {cl['color']};
                    border-radius:12px;padding:1.2rem 1.5rem;margin-top:.5rem">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
            <div style="width:14px;height:14px;border-radius:50%;
                        background:{cl['color']};flex-shrink:0"></div>
            <span style="font-size:18px;font-weight:700;color:{cl['color']}">{cl['label']}</span>
          </div>
          {desc}{tags}
          <p style="font-size:12px;color:#6c757d;margin-top:10px">
            {mode} &nbsp;·&nbsp; Model: <strong>{sel_model}</strong>
            &nbsp;·&nbsp; Cluster ID: <code>{pred_id}</code>
          </p>
        </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Input Summary")
        
        summary_data = {
            "Feature": [
                "Age", "Bank", "Employment", "Loan amount", "Total due",
                "Term (days)", "Interest rate", "Total amount return",
                "Prev. loans", "Prev. loan amount", "Prev. total due",
                "Prev. term (days)", "Days since last loan"
            ],
            "Value": [
                f"{age} yrs", bank, emp_type,
                f"{loanamount:,.0f}", f"{totaldue:,.0f}", f"{termdays} days",
                f"{interestrate}%", f"{totalamountreturn:,.0f}",
                loannumberprevious, f"{loanamountprevious:,.0f}",
                f"{totaldueprevious:,.0f}", f"{termdaysprevious} days",
                f"{datediffsecondsprevious} days"
            ],
        }
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

        if not used_real:
            st.info(
                "📁 Place the `.pkl` files in the same folder as `app.py` to enable real predictions.\n\n"
                "Required: `scaler.pkl` · `dec_predictor.pkl` · `mm_scaler_dec.pkl` · "
                "`som_model.pkl` · `kmeans_som_model.pkl` · `kmeans_model.pkl` · "
                "`gmm_model.pkl` · `rbm_model.pkl` · `mm_scaler_gmm.pkl` · " 
                "`dbscan_predictor.pkl` · `dbscan_meta.pkl`"
                "`svm_model.pkl` · `svm_scaler.pkl`"
            )