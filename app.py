import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
import altair as alt
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Customer Segmentation App",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- ADVANCED CUSTOM CSS FOR PREMIUM UI/UX ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Space Grotesk', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0A0F24 0%, #171E36 100%);
    }

    .premium-header {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        position: relative;
        overflow: hidden;
    }
    .premium-header h1 {
        background: linear-gradient(135deg, #4DA8DA 0%, #A78BFA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 3rem;
        margin-bottom: 5px;
    }
    .premium-header p {
        color: #E2E8F0;
        font-size: 1.2rem;
        font-weight: 400;
        letter-spacing: 1px;
    }
    .premium-header hr {
        border-top: 2px solid rgba(255, 255, 255, 0.15);
        width: 40%;
        margin: 20px auto;
    }
    .premium-header h4, .premium-header h5 {
        color: #94A3B8;
        margin: 0;
        font-weight: 400;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 25px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: 100%;
    }
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(167, 139, 250, 0.3);
    }
    h3, h4 { color: white !important; }
    p { color: #CBD5E1; }
    .stButton>button {
        background: linear-gradient(135deg, #7C3AED 0%, #3B82F6 100%) !important;
        color: white !important;
        border-radius: 50px !important;
        padding: 12px 30px !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        border: none !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4) !important;
    }
    .stButton>button:hover {
        box-shadow: 0 0 25px rgba(124, 58, 237, 0.8) !important;
        transform: scale(1.02) !important;
    }
    .stSlider > div > div > div > div { background: #7C3AED !important; }
</style>
""", unsafe_allow_html=True)

# --- HEADER SECTION ---
st.markdown("""
<div class="premium-header">
    <h1>Customer Segmentation using K-Means Clustering</h1>
    <p>A Data-Driven Analysis of Mall Customer Behaviour</p>
    <hr>
    <h4>DEPARTMENT OF COMPUTER SCIENCE</h4>
    <h5>KIET DEEMED TO BE UNIVERSITY</h5>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("Mall_Customers.csv")

df = load_data()


# --- 1. EXPLORATORY DATA ANALYSIS ---
st.markdown("### 🔍 Exploratory Data Analysis")
st.markdown("<p style='color:#94A3B8; margin-bottom:20px;'>Mirroring the Colab visualizations for initial dataset understanding.</p>", unsafe_allow_html=True)

eda_c1, eda_c2, eda_c3 = st.columns(3)

def apply_theme(chart):
    return chart.configure_axis(labelColor='#94A3B8', titleColor='#A78BFA', gridColor='rgba(255, 255, 255, 0.05)').configure_title(color='white')

with eda_c1:
    age_chart = alt.Chart(df).mark_bar(color='#4DA8DA', opacity=0.85, cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
        x=alt.X('Age:Q', bin=alt.Bin(maxbins=10), title="Age"),
        y=alt.Y('count():Q', title="Count"),
        tooltip=['count()']
    ).properties(height=300, title="Age Distribution", background='transparent')
    st.altair_chart(apply_theme(age_chart), use_container_width=True)

with eda_c2:
    scatter_colab = alt.Chart(df).mark_circle(size=80, color='#A78BFA', opacity=0.8).encode(
        x=alt.X('Annual Income (k$):Q', title="Annual Income (k$)"),
        y=alt.Y('Spending Score (1-100):Q', title="Spending Score (1-100)"),
        tooltip=['Annual Income (k$)', 'Spending Score (1-100)']
    ).properties(height=300, title="Income vs Spending (Pre-Clustering)", background='transparent')
    st.altair_chart(apply_theme(scatter_colab), use_container_width=True)

with eda_c3:
    gender_chart = alt.Chart(df).mark_bar(opacity=0.85, cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
        x=alt.X('Gender:N', title="Gender", axis=alt.Axis(labelAngle=0)),
        y=alt.Y('count():Q', title="Count"),
        color=alt.Color('Gender:N', legend=None, scale=alt.Scale(range=['#FBBF24', '#34D399'])),
        tooltip=['Gender', 'count()']
    ).properties(height=300, title="Gender Distribution", background='transparent')
    st.altair_chart(apply_theme(gender_chart), use_container_width=True)

st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin:40px 0;'>", unsafe_allow_html=True)


# --- 2. TRAIN K-MEANS MODEL ---
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

@st.cache_resource
def train_model(X_data):
    model = KMeans(n_clusters=5, init='k-means++', random_state=42)
    labels = model.fit_predict(X_data)
    sil_score = silhouette_score(X_data, labels)
    
    # Calculate WCSS (Inertia) for Elbow method
    wcss = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, init='k-means++', random_state=42)
        km.fit(X_data)
        wcss.append(km.inertia_)
        
    return model, labels, sil_score, wcss

model, cluster_labels, sil_score, inertia_values = train_model(X)

cluster_names = {}
centers = model.cluster_centers_

for i, center in enumerate(centers):
    inc, spend = center[0], center[1]
    if inc < 45 and spend < 45: cluster_names[i] = "Budget Customer"
    elif inc < 45 and spend >= 45: cluster_names[i] = "Impulsive Customer"
    elif inc >= 65 and spend < 45: cluster_names[i] = "Target Customer"
    elif inc >= 65 and spend >= 45: cluster_names[i] = "Premium Customer"
    else: cluster_names[i] = "Medium Customer"

df['Cluster'] = cluster_labels
df['Customer Segment'] = df['Cluster'].map(cluster_names)

color_scale = alt.Scale(domain=[
    "Budget Customer", "Impulsive Customer", "Medium Customer", 
    "Target Customer", "Premium Customer"
], range=['#FF4B4B', '#4DA8DA', '#34D399', '#FBBF24', '#A78BFA'])


# --- 3. MAIN PREDICTION & CLUSTERING SECTION ---
col1, space, col2 = st.columns([2, 0.1, 1.2])

with col1:
    st.markdown("### 📊 Customer Segments Visualized")
    chart = alt.Chart(df).mark_circle(size=120, opacity=0.9).encode(
        x=alt.X('Annual Income (k$):Q', title='Annual Income (k$)', scale=alt.Scale(zero=False)),
        y=alt.Y('Spending Score (1-100):Q', title='Spending Score (1-100)', scale=alt.Scale(zero=False)),
        color=alt.Color('Customer Segment:N', scale=color_scale, legend=alt.Legend(title="Segments", orient="bottom", titleColor="white", labelColor="white")),
        tooltip=['Age', 'Gender', 'Annual Income (k$)', 'Spending Score (1-100)', 'Customer Segment']
    ).interactive().properties(height=480, background='transparent')
    
    st.altair_chart(apply_theme(chart), use_container_width=True)

with col2:
    st.markdown("### 🎯 Predict New Segment")
    
    with st.form("custom_predict"):
        st.markdown("<p style='color:#CBD5E1;'>Enter customer details to analyse their cluster.</p>", unsafe_allow_html=True)
        income = st.slider("💰 Annual Income (k$)", min_value=0, max_value=150, value=60, step=1)
        spending = st.slider("🛒 Spending Score", min_value=0, max_value=100, value=50, step=1)
        
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("✨ Analyze Customer ✨")
        
    if submitted:
        new_customer = [[income, spending]]
        cluster_idx = model.predict(new_customer)[0]
        segment = cluster_names[cluster_idx]
        
        if segment == "Budget Customer":
            icon, desc = "🪙", "This customer has LOW income and LOW spending.<br>👉 They are budget customers who spend less."
        elif segment == "Impulsive Customer":
            icon, desc = "🛍️", "This customer has LOW income but HIGH spending.<br>👉 They spend more than expected → impulsive buyers."
        elif segment == "Target Customer":
            icon, desc = "🎯", "This customer has HIGH income but LOW spending.<br>👉 They can spend more but currently don’t.<br>👉 Businesses target them with offers → 'Target Customers'."
        elif segment == "Premium Customer":
            icon, desc = "💎", "This customer has HIGH income and HIGH spending.<br>👉 These are premium customers (very valuable)."
        else:
            icon, desc = "🛒", "This customer has MEDIUM income and MEDIUM spending.<br>👉 They are normal/average customers."
            
        st.markdown(f"""
        <div class="glass-card" style="border-left: 5px solid #A78BFA; animation: fadeIn 0.5s; padding: 20px;">
            <p style="margin:0; font-size:0.9rem; color:#94A3B8;">========== CUSTOMER ANALYSIS ==========</p>
            <h3 style="border-bottom:none; margin:10px 0; color:#A78BFA;">{icon} Cluster Assigned</h3>
            <p style="margin:0; font-weight:600; color:white;">Customer Type: {segment}</p>
            <br>
            <p style="margin:0; font-size:1rem; color:#4DA8DA; font-weight:bold;">🔍 Explanation:</p>
            <p style="margin-top:5px; line-height:1.5;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

# --- 4. ADVANCED UNSUPERVISED LEARNING VISUALS (NEW) ---
st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin:40px 0;'>", unsafe_allow_html=True)
st.markdown("### 🌌 3D Unsupervised Spatial Mapping")
st.markdown("<p style='color:#CBD5E1; margin-bottom:20px;'>Plotting all 3 primary dimensions (Age, Income, Spending) interactively to prove how our Unsupervised K-Means clustering algorithm dominates the hyper-dimensional spatial data.</p>", unsafe_allow_html=True)

plotly_colors = {'Budget Customer': '#FF4B4B', 'Impulsive Customer': '#4DA8DA', 'Medium Customer': '#34D399', 'Target Customer': '#FBBF24', 'Premium Customer': '#A78BFA'}

fig_3d = px.scatter_3d(df, x='Age', y='Annual Income (k$)', z='Spending Score (1-100)',
                     color='Customer Segment', opacity=0.8,
                     color_discrete_map=plotly_colors)

fig_3d.update_layout(
    scene=dict(
        xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.1)", showbackground=False, title_font=dict(color='white'), tickfont=dict(color='white')),
        yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.1)", showbackground=False, title_font=dict(color='white'), tickfont=dict(color='white')),
        zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.1)", showbackground=False, title_font=dict(color='white'), tickfont=dict(color='white'))
    ),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=0, r=0, b=0, t=0),
    legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0)")
)
st.plotly_chart(fig_3d, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 🧬 Cluster 'DNA' Profiling")
st.markdown("<p style='color:#CBD5E1; margin-bottom:20px;'>Analyzing the exact behavioral trait distributions (Age, Income, Spending) of each specific unsupervised cluster through aggregated normalized Radar Charts.</p>", unsafe_allow_html=True)

radar_cols = st.columns(5)
metrics = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

scaler = MinMaxScaler()
radar_data = df.copy()
radar_data[metrics] = scaler.fit_transform(radar_data[metrics])
cluster_means = radar_data.groupby('Customer Segment')[metrics].mean().reset_index()
order = ["Budget Customer", "Impulsive Customer", "Medium Customer", "Target Customer", "Premium Customer"]
cluster_means['Customer Segment'] = pd.Categorical(cluster_means['Customer Segment'], categories=order, ordered=True)
cluster_means = cluster_means.sort_values('Customer Segment')

for i, row in cluster_means.iterrows():
    seg = row['Customer Segment']
    col_idx = order.index(seg)
    c_hex = plotly_colors[seg]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[row['Age'], row['Annual Income (k$)'], row['Spending Score (1-100)'], row['Age']],
        theta=['Age', 'Income', 'Spending', 'Age'],
        fill='toself',
        name=seg,
        line_color=c_hex
    ))
    fig.update_traces(opacity=0.7)
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 1]),
            bgcolor='rgba(0,0,0,0)',
            angularaxis=dict(color="white", tickfont=dict(size=11))
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=25, r=25, t=30, b=25),
        title=dict(text=seg, font=dict(color=c_hex, size=15), x=0.5, y=0.98)
    )
    radar_cols[col_idx].plotly_chart(fig, use_container_width=True)


# --- 5. ELBOW METHOD ---
st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin:40px 0;'>", unsafe_allow_html=True)
st.markdown("### 📉 Elbow Method for Optimal Clusters")

eb_col1, eb_col2 = st.columns([1.5, 2])

with eb_col1:
    st.markdown("""
    <div style="margin-bottom: 15px;">
        <h4 style="color:#A78BFA; font-weight:600; text-transform:uppercase; letter-spacing:1px; margin-bottom:5px;">Determining the Optimal Number of Clusters</h4>
    </div>
    <p style="color:#CBD5E1; font-size:1.05rem; line-height:1.6;">
    The Elbow Method plots the number of clusters (K) against Inertia (within-cluster sum of squares). As K increases, inertia decreases, but at some point the rate of decrease sharply slows—forming an "elbow" shape. 
    This elbow point indicates the optimal number of clusters where adding more clusters yields diminishing returns.
    </p>
    <p style="color:#CBD5E1; font-size:1.05rem; line-height:1.6; margin-top:15px;">
    For our customer segmentation analysis, the elbow point was clearly identified at <strong>K = 5</strong>, suggesting five distinct customer groups provide the best balance between cluster compactness and model simplicity.
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card" style="display:inline-block; border-left: 5px solid #3B82F6; margin-top:10px; padding: 15px 30px;">
        <h3 style="margin:0; border:none; padding:0; color:#4DA8DA; font-size:2.4rem; justify-content:center;">K=5</h3>
        <p style="margin:0; font-weight:bold; letter-spacing:1px; color:#94A3B8; text-transform:uppercase; font-size:0.85rem; text-align:center;">Optimal Clusters<br>Identified</p>
    </div>
    """, unsafe_allow_html=True)

with eb_col2:
    elbow_df = pd.DataFrame({'Number of Clusters (K)': range(1, 11), 'Inertia (WCSS)': inertia_values})
    elbow_base = alt.Chart(elbow_df).encode(x=alt.X('Number of Clusters (K):O', title='Number of Clusters (K)', axis=alt.Axis(labelAngle=0)))
    elbow_line = elbow_base.mark_line(color='#4DA8DA', strokeWidth=3).encode(y=alt.Y('Inertia (WCSS):Q', title='Inertia / WCSS'))
    elbow_points = elbow_base.mark_circle(color='#FBBF24', size=80, opacity=1).encode(y=alt.Y('Inertia (WCSS):Q'), tooltip=['Number of Clusters (K)', 'Inertia (WCSS)'])
    
    elbow_chart = (elbow_line + elbow_points).interactive().properties(height=320, background='transparent')
    st.altair_chart(apply_theme(elbow_chart).configure_axis(domainColor='rgba(255, 255, 255, 0.2)'), use_container_width=True)

# --- 6. REASONING CARDS ---
st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin:40px 0;'>", unsafe_allow_html=True)
st.markdown("### 💡 Model Evaluation & Project Reasoning")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class="glass-card">
        <h3 style="color: #4DA8DA;">📐 ≈ {sil_score:.2f} Silhouette Score</h3>
        <p>This score indicates strong clustering quality. It measures how perfectly separated our {len(cluster_names)} customer groups are mathematically from each other, validating our model's accuracy.</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="glass-card">
        <h3 style="color: #FBBF24;">🤖 Why K-Means Algorithm?</h3>
        <p>We selected K-Means because it is exceptionally efficient at finding hidden patterns and grouping numerical data without needing pre-labeled training datasets. It is the industry standard for customer segmentation.</p>
    </div>
    """, unsafe_allow_html=True)
    
with c3:
    st.markdown("""
    <div class="glass-card">
        <h3 style="color: #34D399;">📊 Why Income & Spending?</h3>
        <p>These two features were specifically chosen because they directly reflect a customer's purchasing behavior and economic capacity, which are the main actionable driving forces for marketing strategies.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)
