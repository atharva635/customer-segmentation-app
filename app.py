import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import time
import pickle
import io

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Customer Segmentation App",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
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
        margin-bottom: 20px;
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
    
    /* Pipeline UI */
    .pipeline-bar {
        text-align: center;
        background: rgba(0,0,0,0.3);
        padding: 15px;
        border-radius: 50px;
        color: #A78BFA;
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 40px;
        border: 1px solid rgba(167, 139, 250, 0.2);
    }
    .pipeline-arrow {
        color: #4DA8DA;
        margin: 0 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR (EXPORT ZONE) ---
st.sidebar.markdown("<h2 style='color:white; text-align:center;'>⬇️ Export Center</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color:#94A3B8; text-align:center;'>Download generated assets</p><hr>", unsafe_allow_html=True)

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

# --- PIPELINE UI ---
st.markdown("""
<div class="pipeline-bar">
    📂 Data <span class="pipeline-arrow">➔</span> 📊 EDA <span class="pipeline-arrow">➔</span> 🤖 Clustering <span class="pipeline-arrow">➔</span> 📈 Evaluation <span class="pipeline-arrow">➔</span> 🎯 Prediction & Business Insights
</div>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    return pd.read_csv("Mall_Customers.csv")

df = load_data()


# --- 1. EXPLORATORY DATA ANALYSIS ---
st.markdown("### 🔍 Exploratory Data Analysis (EDA)")
st.markdown("<p style='color:#94A3B8; margin-bottom:20px;'>Pre-Clustering Data Distributions.</p>", unsafe_allow_html=True)

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


# --- 2. MODEL TRAINING (WITH ANIMATION & AUTO-K) ---
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

@st.cache_resource
def train_model(X_data):
    with st.spinner("🤖 Discovering Data Patterns dynamically..."):
        time.sleep(1.5) # Simulate processing for presentation flair
        
        # Calculate WCSS (Inertia) & Silhouette for Auto-K Selection
        wcss = []
        sil_scores = []
        K_range_sil = range(2, 11)
        
        # We need inertia for 1 to 10
        wcss.append(KMeans(n_clusters=1, init='k-means++', random_state=42).fit(X_data).inertia_)
        
        for k in K_range_sil:
            km = KMeans(n_clusters=k, init='k-means++', random_state=42)
            labels = km.fit_predict(X_data)
            wcss.append(km.inertia_)
            sil_scores.append(silhouette_score(X_data, labels))
            
        best_k = K_range_sil[np.argmax(sil_scores)]
        
        # Train final optimal model
        model = KMeans(n_clusters=best_k, init='k-means++', random_state=42)
        labels = model.fit_predict(X_data)
        final_sil = silhouette_score(X_data, labels)
        
    return model, labels, final_sil, wcss, sil_scores, best_k

model, cluster_labels, sil_score, inertia_values, silhouette_values, best_k_auto = train_model(X)

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
plotly_colors = {'Budget Customer': '#FF4B4B', 'Impulsive Customer': '#4DA8DA', 'Medium Customer': '#34D399', 'Target Customer': '#FBBF24', 'Premium Customer': '#A78BFA'}

# Model & Data Exports
st.sidebar.download_button(
    label="📄 Download Clustered Data (.CSV)",
    data=df.to_csv(index=False),
    file_name="clustered_mall_customers.csv",
    mime="text/csv"
)

# Export Pickle properly
model_bytes = pickle.dumps(model)
st.sidebar.download_button(
    label="⚙️ Download K-Means Model (.PKL)",
    data=model_bytes,
    file_name="kmeans_model.pkl",
    mime="application/octet-stream"
)

# --- 3. MAIN PREDICTION & CLUSTERING SECTION ---
col1, space, col2 = st.columns([2, 0.1, 1.2])

with col1:
    st.markdown("### 🤖 K-Means Clustering Result")
    st.markdown("<p style='color:#94A3B8; margin-bottom:10px;'>Post-clustering boundary assignments.</p>", unsafe_allow_html=True)
    chart = alt.Chart(df).mark_circle(size=120, opacity=0.9).encode(
        x=alt.X('Annual Income (k$):Q', title='Annual Income (k$)', scale=alt.Scale(zero=False)),
        y=alt.Y('Spending Score (1-100):Q', title='Spending Score (1-100)', scale=alt.Scale(zero=False)),
        color=alt.Color('Customer Segment:N', scale=color_scale, legend=alt.Legend(title="Segments", orient="bottom", titleColor="white", labelColor="white")),
        tooltip=['Age', 'Gender', 'Annual Income (k$)', 'Spending Score (1-100)', 'Customer Segment']
    ).interactive().properties(height=450, background='transparent')
    
    st.altair_chart(apply_theme(chart), use_container_width=True)

with col2:
    st.markdown("### 🎯 Real-Time Custom Prediction")
    
    with st.form("custom_predict"):
        st.markdown("<p style='color:#CBD5E1;'>Enter details to predict cluster.</p>", unsafe_allow_html=True)
        income = st.slider("💰 Annual Income (k$)", min_value=0, max_value=150, value=60, step=1)
        spending = st.slider("🛒 Spending Score", min_value=0, max_value=100, value=50, step=1)
        
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("✨ Analyze Customer ✨")
        
    if submitted:
        with st.spinner("Classifying..."):
            time.sleep(0.5)
            new_customer = [[income, spending]]
            cluster_idx = model.predict(new_customer)[0]
            segment = cluster_names[cluster_idx]
            
            if segment == "Budget Customer":
                icon, logic = "🪙", "Low Income & Low Spending"
            elif segment == "Impulsive Customer":
                icon, logic = "🛍️", "Low Income & High Spending"
            elif segment == "Target Customer":
                icon, logic = "🎯", "High Income & Low Spending"
            elif segment == "Premium Customer":
                icon, logic = "💎", "High Income & High Spending"
            else:
                icon, logic = "🛒", "Medium Income & Medium Spending"
                
            st.markdown(f"""
            <div class="glass-card" style="border-left: 5px solid #A78BFA; animation: fadeIn 0.5s; padding: 20px; margin-top:20px;">
                <h4 style="margin:0; color:#A78BFA;">{icon} Predicted Segment</h4>
                <h3 style="margin:5px 0 10px 0; color:white;">{segment}</h3>
                <p style="margin:0; font-size:0.95rem; color:#4DA8DA; font-weight:bold;">📊 Data Reason:</p>
                <p style="margin:5px 0 0 0; color:#CBD5E1; font-size:0.9rem;">{logic} → Falls into Cluster {cluster_idx}</p>
            </div>
            """, unsafe_allow_html=True)

# --- 4. ADVANCED UNSUPERVISED LEARNING VISUALS ---
st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin:40px 0;'>", unsafe_allow_html=True)
st.markdown("### 🌌 3D Spatial Mapping & Cluster Density")
st.markdown("<p style='color:#CBD5E1; margin-bottom:20px;'>Plotting all 3 primary dimensions interactively, along with segment sizes.</p>", unsafe_allow_html=True)

adv_c1, adv_c2 = st.columns([2, 1])

with adv_c1:
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
        legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_3d, use_container_width=True)

with adv_c2:
    cluster_counts = df['Customer Segment'].value_counts()
    fig_pie = px.pie(values=cluster_counts.values, names=cluster_counts.index, 
                     color=cluster_counts.index, color_discrete_map=plotly_colors, hole=0.4)
    fig_pie.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        annotations=[dict(text='Market<br>Share', x=0.5, y=0.5, font_size=16, font_color='white', showarrow=False)]
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#0A0F24', width=2)))
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 🧬 Exact Cluster 'DNA' Profiling")

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
        fill='toself', name=seg, line_color=c_hex
    ))
    fig.update_traces(opacity=0.7)
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 1]),
            bgcolor='rgba(0,0,0,0)', angularaxis=dict(color="white", tickfont=dict(size=11))
        ),
        showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=25, r=25, t=30, b=25), title=dict(text=seg, font=dict(color=c_hex, size=15), x=0.5, y=0.98)
    )
    radar_cols[col_idx].plotly_chart(fig, use_container_width=True)


# --- 5. EVALUATION: AUTO K & ELBOW ---
st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin:40px 0;'>", unsafe_allow_html=True)
st.markdown("### 📉 Model Evaluation & Auto-Cluster Detection")

eb_col1, eb_col2, eb_col3 = st.columns([1.2, 1.5, 1.5])

with eb_col1:
    st.markdown(f"""
    <div class="glass-card" style="border-left: 5px solid #FBBF24;">
        <h4 style="color:#A78BFA; margin-top:0;">Auto-Detected Best 'K'</h4>
        <h1 style="color:#FBBF24; font-size:4rem; margin:0; line-height:1;">{best_k_auto}</h1>
        <p style="color:#CBD5E1; font-size:1rem; margin-top:10px;">Our algorithm automatically computed the Silhouette Score for clusters 2 through 10 and dynamically identified <strong>K={best_k_auto}</strong> as mathematically optimal without human bias.</p>
    </div>
    """, unsafe_allow_html=True)

with eb_col2:
    elbow_df = pd.DataFrame({'Number of Clusters (K)': range(1, 11), 'Inertia (WCSS)': inertia_values})
    elbow_base = alt.Chart(elbow_df).encode(x=alt.X('Number of Clusters (K):O', title='Clusters (K)', axis=alt.Axis(labelAngle=0)))
    elbow_line = elbow_base.mark_line(color='#4DA8DA', strokeWidth=3).encode(y=alt.Y('Inertia (WCSS):Q', title='Inertia / WCSS'))
    elbow_points = elbow_base.mark_circle(color='#3B82F6', size=80).encode(y=alt.Y('Inertia (WCSS):Q'), tooltip=['Number of Clusters (K)', 'Inertia (WCSS)'])
    
    elbow_chart = (elbow_line + elbow_points).interactive().properties(height=280, title="Elbow Method", background='transparent')
    st.altair_chart(apply_theme(elbow_chart), use_container_width=True)

with eb_col3:
    sil_df = pd.DataFrame({'Number of Clusters (K)': range(2, 11), 'Silhouette Score': silhouette_values})
    sil_base = alt.Chart(sil_df).encode(x=alt.X('Number of Clusters (K):O', title='Clusters (K)', axis=alt.Axis(labelAngle=0)))
    sil_line = sil_base.mark_line(color='#FBBF24', strokeWidth=3).encode(y=alt.Y('Silhouette Score:Q', title='Silhouette Score', scale=alt.Scale(zero=False)))
    sil_points = sil_base.mark_circle(color='#F59E0B', size=80).encode(y=alt.Y('Silhouette Score:Q'), tooltip=['Number of Clusters (K)', 'Silhouette Score'])
    
    sil_chart = (sil_line + sil_points).interactive().properties(height=280, title="Silhouette Score Validation", background='transparent')
    st.altair_chart(apply_theme(sil_chart), use_container_width=True)

# --- 6. BUSINESS STRATEGY INSIGHTS ---
st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin:40px 0;'>", unsafe_allow_html=True)
st.markdown("### 💼 Real-World Business Strategies")
st.markdown("<p style='color:#CBD5E1; margin-bottom:20px;'>Transforming machine learning boundaries into actionable product marketing.</p>", unsafe_allow_html=True)

b1, b2, b3 = st.columns(3)

with b1:
    st.markdown("""
    <div class="glass-card" style="border-top: 4px solid #A78BFA;">
        <h3 style="color:#A78BFA; margin-top:0;">💎 Premium Customers</h3>
        <p><strong>Action:</strong> Push luxury items & VIP memberships.</p>
        <p style="font-size:0.9rem; color:#94A3B8;">Since they have high income and spend heavily, target them with exclusive early-access sales and expensive high-margin products.</p>
    </div>
    """, unsafe_allow_html=True)

with b2:
    st.markdown("""
    <div class="glass-card" style="border-top: 4px solid #FBBF24;">
        <h3 style="color:#FBBF24; margin-top:0;">🎯 Target Customers</h3>
        <p><strong>Action:</strong> Send targeted conversion offers.</p>
        <p style="font-size:0.9rem; color:#94A3B8;">They have high income but low spending. They are the biggest untapped potential. Send them tailored discounts to convert them into Premium spenders.</p>
    </div>
    """, unsafe_allow_html=True)
    
with b3:
    st.markdown("""
    <div class="glass-card" style="border-top: 4px solid #4DA8DA;">
        <h3 style="color:#4DA8DA; margin-top:0;">🛍️ Impulsive Buyers</h3>
        <p><strong>Action:</strong> Highlight flash sales & FOMO items.</p>
        <p style="font-size:0.9rem; color:#94A3B8;">They have low income but love to spend. Target them with fast-moving consumer goods, limited-time offers, and bulk discounts.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)
