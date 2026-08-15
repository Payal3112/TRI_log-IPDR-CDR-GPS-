# app.py
import streamlit as st
import os
import json
import pandas as pd
import base64
from utils import normalize_columns, check_required, extract_gps_from_android_image,compute_file_hash
from train_model import train_anomaly_model, format_output_table,detect_spoofing_and_sim_swap
from streamlit_folium import st_folium
from map_utils import create_hybrid_movement_map_with_labels
import altair as alt

st.set_page_config(page_title="Android Forensics", layout="wide")

def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception:
        return ""

img_base64 = get_base64_of_bin_file('hero_bg_blue.png')

st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');
        
        /* Hide default Streamlit elements */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}
        
        @keyframes forensicDive {{
            0% {{ transform: perspective(1000px) translateZ(0) rotateX(0deg); opacity: 1; filter: hue-rotate(0deg); }}
            40% {{ transform: perspective(1000px) translateZ(-150px) rotateX(5deg); opacity: 1; filter: hue-rotate(180deg) brightness(150%) contrast(120%); box-shadow: inset 0 0 150px rgba(56,189,248,0.5); }}
            100% {{ transform: perspective(1000px) translateZ(800px) rotateX(-20deg); opacity: 0; pointer-events: none; visibility: hidden; filter: blur(20px); }}
        }}
        
        /* Restored Dark Theme Background */
        .stApp {{
            background: transparent !important;
            color: #f8fafc;
        }}
        
        .stApp::before {{
            content: '';
            position: fixed;
            top: 50%; left: 50%;
            width: 150vw;
            height: 150vw;
            background-image: radial-gradient(circle at 50% 50%, rgba(0, 40, 100, 0.4) 0%, rgba(2, 6, 23, 1) 70%), url("data:image/png;base64,{img_base64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            z-index: -2;
            animation: rotate-bg 150s linear infinite;
        }}
        
        /* App Container (Glowing Glassmorphism) */
        .block-container {{
            padding-top: 1rem; padding-bottom: 3rem; max-width: 1200px;
            background: rgba(15, 23, 42, 0.65);
            backdrop-filter: blur(16px);
            border-radius: 24px;
            box-shadow: 0 0 40px rgba(56, 189, 248, 0.15);
            margin-top: 2rem;
            border: 1px solid rgba(56, 189, 248, 0.2);
            color: #f8fafc;
        }}
        
        /* Interactive Splash Screen Checkbox Hack */
        #splash-toggle {{
            display: none;
        }}
        #splash-toggle:checked ~ .hero-section {{
            animation: forensicDive 1.8s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }}
        
        @keyframes rotate-bg {{
            0% {{ transform: translate(-50%, -50%) rotate(0deg); }}
            100% {{ transform: translate(-50%, -50%) rotate(360deg); }}
        }}
        
        /* Splash Screen (Hero Section - Exact Replica) */
        .hero-section {{
            position: fixed;
            top: 0; left: 0;
            width: 100vw;
            height: 100vh;
            z-index: 9999999999 !important;
            background-color: transparent !important;
            display: flex;
            flex-direction: column;
            align-items: center;
            color: white;
            font-family: 'Inter', sans-serif;
            overflow: hidden;
        }}

        .hero-content {{
            margin-top: 18vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
        }}

        .new-pill {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 6px 16px 6px 6px;
            border-radius: 50px;
            font-size: 0.85rem;
            color: #cbd5e1;
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 35px;
            cursor: pointer;
            font-weight: 500;
        }}
        .new-pill .badge {{
            background: rgba(255, 255, 255, 0.1);
            color: white;
            padding: 4px 10px;
            border-radius: 50px;
            font-weight: 600;
        }}

        .hero-title {{
            font-size: 4rem;
            font-family: 'Playfair Display', serif;
            font-weight: 500;
            line-height: 1.1;
            margin-bottom: 25px;
            letter-spacing: -1px;
            color: #ffffff;
        }}

        .hero-subtitle {{
            font-size: 1.1rem;
            font-weight: 400;
            color: #94a3b8;
            max-width: 650px;
            line-height: 1.6;
            margin-bottom: 40px;
        }}
        
        .enter-btn {{
            background: #d9f95d; 
            color: #020617 !important; 
            padding: 14px 32px;
            border-radius: 50px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        .enter-btn:hover {{
            transform: scale(1.03);
            background: #c3e24a;
        }}
        
        /* Hide the Streamlit Sidebar when splash screen is active */
        .stApp:has(#splash-toggle:not(:checked)) section[data-testid="stSidebar"] {{
            display: none !important;
        }}
        
        /* Hide main dashboard elements when splash screen is active */
        .stApp:has(#splash-toggle:not(:checked)) div[data-testid="stVerticalBlock"] > *:not(:first-child) {{
            display: none !important;
        }}

        /* Hide the glassmorphism block-container background when splash is active */
        .stApp:has(#splash-toggle:not(:checked)) .block-container {{
            background: transparent !important;
            backdrop-filter: none !important;
            border: none !important;
            box-shadow: none !important;
        }}
        
        /* 🌟 Shiny Animations for Buttons 🌟 */
        @keyframes shine {{
            0% {{ background-position: -200% center; }}
            100% {{ background-position: 200% center; }}
        }}
        
        .stButton button, [data-testid="stFileUploader"] button, button[data-testid*="baseButton"] {{
            background: linear-gradient(90deg, rgba(15, 23, 42, 0.8), rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.8)) !important;
            background-size: 200% auto !important;
            color: #38bdf8 !important;
            border: 1px solid rgba(56, 189, 248, 0.3) !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
        }}
        
        .stButton button:hover, [data-testid="stFileUploader"] button:hover, button[data-testid*="baseButton"]:hover {{
            animation: shine 3s infinite linear !important;
            transform: translateY(-2px) scale(1.05) !important;
            box-shadow: 0 0 25px rgba(56, 189, 248, 0.8), 0 0 50px rgba(56, 189, 248, 0.4), inset 0 0 15px rgba(56, 189, 248, 0.5) !important;
            border-color: #7dd3fc !important;
            color: #ffffff !important;
            text-shadow: 0 0 8px rgba(255,255,255,0.8) !important;
        }}
        
        /* Input & Dropdown Neon Glow */
        .stTextInput input, .stSelectbox div[data-baseweb="select"] {{
            transition: all 0.3s ease !important;
        }}
        .stTextInput input:hover, .stSelectbox div[data-baseweb="select"]:hover {{
            box-shadow: 0 0 15px rgba(56, 189, 248, 0.4) !important;
            border-color: #38bdf8 !important;
        }}
        .stTextInput input:focus, .stSelectbox div[data-baseweb="select"]:focus-within, .stNumberInput input:focus {{
            box-shadow: 0 0 25px rgba(56, 189, 248, 0.7) !important;
            border-color: #7dd3fc !important;
        }}

        /* Checkbox Glow Effect */
        .stCheckbox {{
            transition: all 0.3s ease !important;
        }}
        .stCheckbox:hover {{
            transform: scale(1.02);
            filter: drop-shadow(0 0 10px rgba(56, 189, 248, 0.6));
        }}

        /* File Uploader Dropzone */
        [data-testid="stFileUploaderDropzone"] {{
            border: 2px dashed rgba(56, 189, 248, 0.4) !important;
            background: rgba(15, 23, 42, 0.5) !important;
            border-radius: 16px !important;
            transition: all 0.3s ease !important;
        }}
        [data-testid="stFileUploaderDropzone"]:hover {{
            border-color: #38bdf8 !important;
            background: rgba(56, 189, 248, 0.1) !important;
            box-shadow: 0 0 25px rgba(56, 189, 248, 0.4) !important;
            transform: scale(1.02);
        }}

        /* Success/Warning Alerts */
        .stAlert {{
            border: 1px solid rgba(56, 189, 248, 0.3) !important;
            background: rgba(15, 23, 42, 0.7) !important;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
            transition: all 0.3s ease !important;
        }}
        .stAlert:hover {{
            box-shadow: 0 0 20px rgba(56, 189, 248, 0.5) !important;
            transform: scale(1.02);
        }}

        /* Interactive Expanders */
        [data-testid="stExpander"] {{
            background: rgba(15, 23, 42, 0.5) !important;
            border: 1px solid rgba(56, 189, 248, 0.2) !important;
            border-radius: 12px !important;
            transition: transform 0.3s ease, box-shadow 0.3s ease !important;
        }}
        [data-testid="stExpander"]:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 5px 20px rgba(56, 189, 248, 0.4) !important;
            border-color: rgba(56, 189, 248, 0.6) !important;
        }}

        /* 🌟 Glowy Tabs 🌟 */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 15px;
            border-bottom: 2px solid rgba(56, 189, 248, 0.1);
        }}
        .stTabs [data-baseweb="tab"] {{
            height: 50px;
            background-color: transparent;
            border-radius: 8px 8px 0px 0px;
            padding: 10px 25px;
            color: #94a3b8;
            font-weight: 600;
            transition: all 0.3s ease;
        }}
        .stTabs [data-baseweb="tab"]:hover {{
            color: #e2e8f0;
            background: rgba(56, 189, 248, 0.05);
            text-shadow: 0 0 10px rgba(56, 189, 248, 0.8);
        }}
        .stTabs [aria-selected="true"] {{
            color: #38bdf8 !important;
            border-bottom: 3px solid #38bdf8 !important;
            background: rgba(56, 189, 248, 0.15);
            box-shadow: 0 -10px 20px -10px rgba(56, 189, 248, 0.4);
            text-shadow: 0 0 15px rgba(56, 189, 248, 0.8) !important;
        }}
        
        /* App Title Styles */
        .main-app-title {{
            font-size: 3.5rem;
            font-family: 'Playfair Display', serif;
            font-weight: 600;
            background: linear-gradient(90deg, #f8fafc, #38bdf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0px;
            text-align: center;
            letter-spacing: -1px;
            filter: drop-shadow(0 0 10px rgba(56,189,248,0.2));
        }}
        .main-app-subtitle {{
            text-align: center;
            color: #94a3b8;
            font-size: 1.1rem;
            margin-bottom: 40px;
            letter-spacing: 2px;
            text-transform: uppercase;
        }}
        /* Overriding Streamlit's default light text elements for Dark Mode */
        label, .stCheckbox > div {{font-size: 15px; font-weight: 500; color: #cbd5e1 !important; transition: all 0.3s ease;}}
        h1, h2, h3, h4, h5, h6, p, div {{color: #f8fafc; font-family: 'Inter', sans-serif; transition: all 0.3s ease;}}
        .stRadio > div, .stSelectbox > div, .stSlider > div {{font-size: 15px; color: #cbd5e1; transition: all 0.3s ease;}}
        
        /* 🌟 Interactive Text Hover Glow 🌟 */
        p:hover, h1:hover, h2:hover, h3:hover, h4:hover, h5:hover, h6:hover, label:hover, .stCheckbox > div:hover {{
            color: #ffffff !important;
            text-shadow: 0 0 10px rgba(56, 189, 248, 0.8), 0 0 20px rgba(56, 189, 248, 0.4) !important;
        }}
        
        /* Dataframes & Tables */
        .stDataFrame, .stTable {{
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(56, 189, 248, 0.1);
        }}
        /* Force table text to be visible */
        table {{ color: #f8fafc !important; background-color: transparent !important; }}
        th, td {{ border-color: rgba(56, 189, 248, 0.1) !important; }}
        
        /* Ensure sidebar is dark too */
        section[data-testid="stSidebar"] {{
            background-color: #020617 !important;
            border-right: 1px solid rgba(56, 189, 248, 0.2);
        }}
        /* Discover Modal Hack */
        #discover-toggle {{
            display: none;
        }}
        
        .modal-overlay {{
            position: fixed;
            top: 0; left: 0; width: 100vw; height: 100vh;
            background: rgba(0,0,0,0.6);
            backdrop-filter: blur(5px);
            z-index: 10000000000;
            opacity: 0;
            pointer-events: none;
            transition: all 0.3s ease;
            display: flex;
            justify-content: center;
            align-items: center;
        }}
        
        .discover-modal {{
            background: white;
            width: 800px;
            max-width: 90vw;
            border-radius: 16px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.3);
            transform: scale(0.95) translateY(20px);
            opacity: 0;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            display: flex;
            flex-direction: column;
            overflow: hidden;
            color: #111827;
        }}
        
        #discover-toggle:checked ~ .hero-section .modal-overlay {{
            opacity: 1;
            pointer-events: all;
        }}
        
        #discover-toggle:checked ~ .hero-section .modal-overlay .discover-modal {{
            transform: scale(1) translateY(0);
            opacity: 1;
        }}
        
        .modal-header {{
            padding: 20px 30px;
            border-bottom: 1px solid #f3f4f6;
            font-size: 1.1rem;
            font-weight: 600;
            color: #111827;
            text-align: left;
        }}
        
        .modal-body {{
            display: flex;
            height: 400px;
            text-align: left;
        }}
        
        .modal-left {{
            flex: 1;
            padding: 20px;
            border-right: 1px solid #f3f4f6;
            overflow-y: auto;
        }}
        
        .modal-right {{
            width: 300px;
            padding: 20px;
            background: #f9fafb;
        }}
        
        .discover-item {{
            display: flex;
            gap: 15px;
            padding: 15px;
            border-radius: 12px;
            cursor: pointer;
            transition: background 0.2s;
        }}
        .discover-item:hover {{ background: #f3f4f6; }}
        
        .discover-icon {{
            width: 40px; height: 40px;
            border-radius: 8px;
            background: #e5e7eb;
            display: flex; justify-content: center; align-items: center;
            font-size: 1.2rem;
            flex-shrink: 0;
        }}
        .discover-text h4 {{ margin: 0 0 4px 0 !important; font-size: 1rem !important; color: #111827 !important; font-weight: 600; }}
        .discover-text p {{ margin: 0 !important; font-size: 0.85rem !important; color: #6b7280 !important; line-height: 1.4 !important; }}
        
        .search-box {{
            display: flex;
            align-items: center;
            background: white;
            border: 1px solid #e5e7eb;
            padding: 10px 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            color: #94a3b8;
            font-size: 0.9rem;
            justify-content: space-between;
        }}
        
        .search-box span {{ font-size: 0.8rem; background: #f3f4f6; padding: 2px 6px; border-radius: 4px; border: 1px solid #e5e7eb; color: #6b7280; }}
        
        .modal-right h5 {{ font-size: 0.8rem !important; text-transform: uppercase; color: #94a3b8 !important; letter-spacing: 0.5px; margin-bottom: 10px !important; }}
        .modal-right ul {{ list-style: none; padding: 0; margin: 0; }}
        .modal-right li {{ font-size: 0.9rem; color: #4b5563; padding: 8px 0; cursor: pointer; }}
        .modal-right li:hover {{ color: #111827; }}
    </style>

<input type="checkbox" id="splash-toggle">
<input type="checkbox" id="discover-toggle">
<div class="hero-section">
<div class="hero-content">
<label for="discover-toggle" class="new-pill">
<span class="badge">New</span> Forensic Analysis Capabilities &gt;
</label>
<div class="hero-title">Android Forensics<br>GPS + IPDR + CDR Analyzer</div>
<label for="splash-toggle" class="enter-btn">Enter Dashboard</label>
</div>
<!-- Modal Overlay -->
<label for="discover-toggle" class="modal-overlay">
<div class="discover-modal" onclick="event.preventDefault(); event.stopPropagation();">
<div class="modal-header">Discover</div>
<div class="modal-body">
<div class="modal-left">
<div class="discover-item">
<div class="discover-icon">🦴</div>
<div class="discover-text">
<h4>Dogfood</h4>
<p>Dogfood is basically eating your own...</p>
</div>
</div>
<div class="discover-item">
<div class="discover-icon">✈️</div>
<div class="discover-text">
<h4>Fly</h4>
<p>Learn how to deploy globally with Fly.</p>
</div>
</div>
<div class="discover-item">
<div class="discover-icon">🔍</div>
<div class="discover-text">
<h4>Found</h4>
<p>Identify and resolve system anomalies.</p>
</div>
</div>
</div>
<div class="modal-right">
<div class="search-box">
Find research <span>⌘K</span>
</div>
<h5>Top searches</h5>
<ul>
<li>Deployment patterns</li>
<li>Observability metrics</li>
<li>Cloud integration</li>
</ul>
</div>
</div>
</div>
</label>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-app-title">Android Forensics</div>
<div class="main-app-subtitle">GPS + IPDR + CDR Analyzer</div>
""", unsafe_allow_html=True)


# Sidebar: Uploads and settings
st.sidebar.header("\U0001F4C2 Upload Forensic Logs")
use_logical_image = st.sidebar.checkbox("\U0001F4C1 Extract GPS from logical image folder", value=True)
gps_only = st.sidebar.checkbox("\U0001F4CD Analyze GPS only (skip IPDR/CDR)", value=True)
profile = st.sidebar.selectbox("\U0001F3AF Detection Profile", ["Conservative", "Balanced", "Aggressive"], index=1)

if profile == "Conservative":
    gps_threshold_default, max_gap_default, speed_threshold_default = 200, 1800, 800
elif profile == "Balanced":
    gps_threshold_default, max_gap_default, speed_threshold_default = 100, 900, 500
else:
    gps_threshold_default, max_gap_default, speed_threshold_default = 50, 600, 300

with st.sidebar.expander("\u2699\ufe0f Detection Parameters", expanded=False):
    gps_threshold_km = st.slider("\U0001F4CD GPS Distance Threshold (km)", 10, 500, value=gps_threshold_default, step=10)
    max_gap_secs = st.slider("\u23F1\ufe0f Max Time Gap Between Logs (seconds)", 60, 3600, value=max_gap_default, step=60)
    speed_threshold = st.slider("\U0001F697 High-Speed Movement Threshold (km/h)", 100, 1000, value=speed_threshold_default, step=50)

# Load Data
if use_logical_image:
    folder_path = st.sidebar.text_input("Enter folder path (e.g. extracted_logical_image/)", value="my_folder")
    gps_df = extract_gps_from_android_image(folder_path) if os.path.exists(folder_path) else pd.DataFrame()
    if gps_df.empty:
        st.sidebar.error("\u274C Folder not found or empty!")
    else:
        st.sidebar.success(f"\u2705 Loaded {len(gps_df)} GPS points from folder")
else:
    gps_file = st.sidebar.file_uploader("Upload GPS CSV", type="csv")
    gps_df = pd.read_csv(gps_file) if gps_file else pd.DataFrame()

ipdr_file = st.sidebar.file_uploader("Upload IPDR CSV (optional)", type="csv")
cdr_file = st.sidebar.file_uploader("Upload CDR CSV (optional)", type="csv")

file_hashes = []
if not use_logical_image :
    if gps_file:
        file_hashes.append(("GPS", gps_file.name, compute_file_hash(gps_file), gps_file.getbuffer().nbytes))
if ipdr_file:
    file_hashes.append(("IPDR", ipdr_file.name, compute_file_hash(ipdr_file), ipdr_file.getbuffer().nbytes))
if cdr_file:
    file_hashes.append(("CDR", cdr_file.name, compute_file_hash(cdr_file), cdr_file.getbuffer().nbytes))


# Proceed if data is ready
if not gps_df.empty and (gps_only or (ipdr_file and cdr_file)):
    gps_df = normalize_columns(gps_df, type="gps")
    ipdr_df = normalize_columns(pd.read_csv(ipdr_file), type="ipdr") if ipdr_file else pd.DataFrame()
    cdr_df = normalize_columns(pd.read_csv(cdr_file), type="cdr") if cdr_file else pd.DataFrame()

    check_required(gps_df, ["timestamp", "lat", "lon"], "GPS")
    check_required(ipdr_df, ["timestamp", "ip", "domain", "lat", "lon"], "IPDR")
    check_required(cdr_df, ["timestamp", "contact", "call_type", "lat", "lon"], "CDR")

    model, scaler, timeline_df, features_df, alerts = train_anomaly_model(gps_df, ipdr_df, cdr_df)
    st.toast("\u2705 Model trained and timeline generated!", icon="\U0001F680")

    # Filters
    anomaly_only = st.sidebar.checkbox("🚨 Show anomalies only")
    long_jump_only = st.sidebar.checkbox("📍 Long GPS jumps only")
    event_types = timeline_df['type'].dropna().unique().tolist()
    selected_types = st.sidebar.multiselect("📊 Event Types", event_types, default=event_types)

    suspicious_domains = ["telegram", "onion", "vpn", "tor"]
    with st.sidebar.expander("⚙️ Advanced Filters", expanded=False):
        suspicious_only = st.checkbox("🕵️ Suspicious domains")
        start_time = st.time_input("🕐 Start Time", value=pd.to_datetime("00:00").time())
        end_time = st.time_input("🕐 End Time", value=pd.to_datetime("23:59").time())

    filtered_df = timeline_df.copy()
    if anomaly_only: filtered_df = filtered_df[filtered_df['anomaly'] == 1]
    if selected_types: filtered_df = filtered_df[filtered_df['type'].isin(selected_types)]
    if suspicious_only and 'domain' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['domain'].str.contains('|'.join(suspicious_domains), na=False, case=False)]
    if 'speed_kmph' in filtered_df.columns:
        filtered_df = filtered_df[pd.to_numeric(filtered_df['speed_kmph'], errors='coerce') > speed_threshold]
    if long_jump_only and 'notes' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['notes'].str.contains('jump', na=False)]
    filtered_df['hour'] = pd.to_datetime(filtered_df['timestamp']).dt.time
    filtered_df = filtered_df[(filtered_df['hour'] >= start_time) & (filtered_df['hour'] <= end_time)]

    # Investigation Summary
    st.markdown("---")
    st.markdown("## 🧠 Investigation Summary")
    summary = {
        "Total Events": len(filtered_df),
        "Anomalies Detected": filtered_df['anomaly'].sum(),
        "GPS Jumps": filtered_df['notes'].str.contains("jump", na=False).sum(),
        "SIM Swap Events": filtered_df['notes'].str.contains("swap", na=False).sum(),
        "Spoofing Detected": filtered_df['notes'].str.contains("spoof", na=False).sum(),
        "Correlation Score > 3": (filtered_df['correlation_score'] > 3).sum() if 'correlation_score' in filtered_df.columns else 0
    }
    st.table(pd.DataFrame(summary.items(), columns=["Metric", "Value"]))

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["📋 Timeline", "📊 Chart", "📍 Map"])
    with tab1:
        st.dataframe(format_output_table(filtered_df).style.set_properties(**{"white-space": "pre-line"}))
    with tab2:
        counts = filtered_df[filtered_df['anomaly'] == 1]['type'].value_counts().reset_index()
        if not counts.empty:
            counts.columns = ['Event Type', 'Count']
            st.altair_chart(alt.Chart(counts).mark_bar().encode(x='Event Type', y='Count', color='Event Type'), use_container_width=True)
    with tab3:
        st.markdown("### 🗺️ Movement Map")
        st_folium(create_hybrid_movement_map_with_labels(filtered_df), width=800, height=550)

    with st.expander("🚨 View Alert Messages", expanded=False):
        if alerts:
            for alert in alerts: st.warning(f"{alert[0]} ➜ {alert[1]}")
        else:
            st.info("✅ No alerts raised based on current filters.")

    st.markdown("---")
    st.header("\U0001F4C4 Forensic Report")
    st.subheader("\U0001F512 Uploaded File Hashes")
    if file_hashes:
        st.table(pd.DataFrame(file_hashes, columns=["Type", "Filename", "SHA256", "Size (bytes)"]))

    st.subheader("\u2699\ufe0f Parameters Used")
    st.json({
        "Profile": profile,
        "GPS Threshold (km)": gps_threshold_km,
        "Max Time Gap (s)": max_gap_secs,
        "Speed Threshold (km/h)": speed_threshold,
        "Anomaly Only": anomaly_only,
        "Jump Only": long_jump_only,
        "Suspicious Domains": suspicious_only,
        "Event Types": selected_types,
        "Time Window": f"{start_time} - {end_time}"
    })

    st.subheader("\U0001F4AC Notable Events")
    top_alerts = timeline_df[timeline_df['notes'].notna()].sort_values("timestamp").head(10)
    for _, row in top_alerts.iterrows():
        st.markdown(f"- **{row['timestamp']}** — {row['notes']}")

    report = {
        "file_hashes": [
            {"type": t, "filename": fn, "sha256": sha, "size": sz} for t, fn, sha, sz in file_hashes
        ],
        "parameters": {
            "profile": profile,
            "gps_threshold_km": gps_threshold_km,
            "max_gap_secs": max_gap_secs,
            "speed_threshold_kmph": speed_threshold,
            "anomaly_only": anomaly_only,
            "long_jump_only": long_jump_only,
            "suspicious_only": suspicious_only,
            "event_types": selected_types,
            "time_window": f"{start_time} - {end_time}"
        },
        "findings": summary,
        "alerts": top_alerts[["timestamp", "notes"]].to_dict(orient="records")
    }
    st.download_button(
    "⬇️ Download Report as JSON",
    json.dumps(report, indent=2, default=str),
    file_name="forensic_report.json"
)


else:
    st.warning("\u26A0\uFE0F Please upload at least the GPS data. IPDR and CDR required unless GPS-only mode is selected.")




