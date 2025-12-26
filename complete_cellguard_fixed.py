import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
import io
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
import re
from datetime import datetime, timedelta
import json

# Page config
st.set_page_config(
    page_title="CellGuard.AI - Battery Intelligence",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful design
st.markdown("""
<style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d3748 0%, #1a202c 100%);
    }
    
    /* Big status cards */
    .status-card-healthy {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        margin: 1rem 0;
    }
    
    .status-card-warning {
        background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        margin: 1rem 0;
    }
    
    .status-card-critical {
        background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        margin: 1rem 0;
    }
    
    /* Metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    /* Alert boxes */
    .alert-critical {
        background: #fed7d7;
        border-left: 5px solid #fc8181;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .alert-high {
        background: #feebc8;
        border-left: 5px solid #f6ad55;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .alert-medium {
        background: #fefcbf;
        border-left: 5px solid #f6e05e;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    /* Headers */
    h1 {
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    h2, h3 {
        color: #2d3748 !important;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    /* Connection status indicator */
    .connection-online {
        background: #48bb78;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    .connection-offline {
        background: #f56565;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'user_mode' not in st.session_state:
    st.session_state.user_mode = 'driver'
if 'connection_status' not in st.session_state:
    st.session_state.connection_status = 'offline'
if 'battery_data' not in st.session_state:
    st.session_state.battery_data = None

# Helper Functions
def gen_sample_data_shell(n=161):
    """Generate sample data matching Shell_BMS.csv format"""
    np.random.seed(42)
    
    base_pack_v = 78.4
    base_current = 2.5
    base_soc = 68
    base_temp = 32
    
    data = {
        'Date': [f'2024-{(i//30)+1:02d}-{(i%30)+1:02d}' for i in range(n)],
        'Time': [f'{(i%24):02d}:{(i*3)%60:02d}:00' for i in range(n)],
        'Pack Vol': base_pack_v + 0.5 * np.sin(np.arange(n)/20) + np.random.normal(0, 0.3, n),
        'Curent': base_current + 0.8 * np.sin(np.arange(n)/15) + np.random.normal(0, 0.2, n),
        'Soc': np.clip(base_soc + 15 * np.sin(np.arange(n)/30) + np.random.normal(0, 2, n), 0, 100),
        'temperature': base_temp + 3 * np.sin(np.arange(n)/25) + np.random.normal(0, 0.8, n),
        'Cycle': np.arange(n) // 2,
        'time': np.arange(n)
    }
    
    # Add 24 cell voltages
    for i in range(1, 25):
        cell_base = 3.27
        if i in [3, 17, 22]:  # Weak cells
            data[f'Cell{i}'] = cell_base - 0.04 + np.random.normal(0, 0.015, n)
        else:
            data[f'Cell{i}'] = cell_base + np.random.normal(0, 0.008, n)
    
    # Add temperature sensors
    for i in range(1, 5):
        if i == 2:  # Hot sensor
            data[f'Temp{i}'] = base_temp + 4 + np.random.normal(0, 1, n)
        else:
            data[f'Temp{i}'] = base_temp + np.random.normal(0, 0.5, n)
    
    df = pd.DataFrame(data)
    
    # Rename for consistency
    df = df.rename(columns={'Pack Vol': 'voltage', 'Curent': 'current', 'Soc': 'soc'})
    
    return df

def analyze_cell_balance(df):
    """Analyze individual cell voltages"""
    df = df.copy()
    cell_cols = [c for c in df.columns if re.match(r'^Cell\d+$', c)]
    
    if len(cell_cols) >= 2:
        df['cell_max'] = df[cell_cols].max(axis=1)
        df['cell_min'] = df[cell_cols].min(axis=1)
        df['cell_diff'] = df['cell_max'] - df['cell_min']
        df['cell_mean'] = df[cell_cols].mean(axis=1)
        df['cell_std'] = df[cell_cols].std(axis=1)
        df['imbalance_flag'] = (df['cell_diff'] > 0.05).astype(int)
    
    return df

def make_features(df, window=10):
    """Create features for ML models"""
    df = df.copy()
    
    if 'voltage' in df.columns and df['voltage'].notna().sum() > 0:
        df['voltage_ma'] = df['voltage'].rolling(window, min_periods=1).mean()
        df['voltage_roc'] = df['voltage'].diff().fillna(0)
        df['voltage_var'] = df['voltage'].rolling(window, min_periods=1).var().fillna(0)
    
    if 'temperature' in df.columns and df['temperature'].notna().sum() > 0:
        df['temp_ma'] = df['temperature'].rolling(window, min_periods=1).mean()
        df['temp_roc'] = df['temperature'].diff().fillna(0)
    
    if 'soc' in df.columns and df['soc'].notna().sum() > 0:
        df['soc_ma'] = df['soc'].rolling(window, min_periods=1).mean()
        df['soc_roc'] = df['soc'].diff().fillna(0)
    
    return df

def run_models(df, contamination=0.05):
    """Run ML models for predictions"""
    df = df.copy()
    
    features = ['voltage', 'current', 'temperature', 'soc', 'voltage_ma', 'voltage_roc']
    features = [f for f in features if f in df.columns and df[f].notna().sum() > 0]
    
    df['anomaly_flag'] = 0
    df['risk_pred'] = 0
    df['battery_health_score'] = 50.0
    
    if len(features) >= 2 and len(df) >= 30:
        try:
            iso = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
            X = df[features].fillna(df[features].median())
            iso.fit(X)
            df['anomaly_flag'] = iso.predict(X).map({1: 0, -1: 1})
        except:
            pass
    
    # Simple health score calculation
    base_score = 100
    
    if 'cell_diff' in df.columns:
        cell_penalty = df['cell_diff'].iloc[-1] * 500  # 50 points per 0.1V diff
        base_score -= cell_penalty
    
    if 'temperature' in df.columns:
        temp = df['temperature'].iloc[-1]
        if temp > 40:
            base_score -= (temp - 40) * 2
    
    if 'anomaly_flag' in df.columns:
        base_score -= df['anomaly_flag'].iloc[-1] * 15
    
    df['battery_health_score'] = np.clip(base_score, 0, 100)
    df['risk_pred'] = (df['battery_health_score'] < 60).astype(int)
    
    return df

def generate_alerts(df_latest, df_history):
    """Generate comprehensive alerts"""
    alerts = []
    latest = df_latest.iloc[-1]
    
    # Critical: Cell imbalance
    if 'cell_diff' in latest and latest['cell_diff'] > 0.05:
        alerts.append({
            'level': 'critical',
            'title': 'Cell Imbalance Critical',
            'detail': f"Cell voltage spread at {latest['cell_diff']*1000:.0f}mV (threshold: 50mV)",
            'action': 'Enable active balancing, reduce charge rate to 0.5C',
            'consequence': 'Weak cells will fail, reduced range',
            'timeline': '1-3 days of continued use',
            'timestamp': '2 min ago'
        })
    
    # High: Temperature
    if 'temperature' in latest and latest['temperature'] > 40:
        alerts.append({
            'level': 'high',
            'title': 'Elevated Temperature',
            'detail': f"Battery temperature {latest['temperature']:.1f}°C above normal",
            'action': 'Improve cooling airflow, check thermal paste',
            'consequence': 'Accelerated aging, capacity loss',
            'timeline': '10-30 minutes',
            'timestamp': '15 min ago'
        })
    
    # Medium: Voltage sag
    if 'voltage_roc' in df_history.columns:
        recent_roc = df_history['voltage_roc'].tail(20).mean()
        if recent_roc < -0.01:
            alerts.append({
                'level': 'medium',
                'title': 'Voltage Sag Pattern',
                'detail': f"Pack voltage declining at {recent_roc*1000:.2f}mV per cycle",
                'action': 'Schedule capacity test within 2 weeks',
                'consequence': 'Reduced performance, range loss',
                'timeline': '2-4 weeks',
                'timestamp': '1 hour ago'
            })
    
    # Low: Maintenance
    cycle_count = latest.get('Cycle', 0)
    if cycle_count > 0 and cycle_count % 50 < 5:
        alerts.append({
            'level': 'low',
            'title': 'Maintenance Due',
            'detail': f"Reached {cycle_count} cycles - routine check recommended",
            'action': 'Perform capacity calibration',
            'consequence': 'Improved accuracy',
            'timeline': 'This week',
            'timestamp': '2 days ago'
        })
    
    return sorted(alerts, key=lambda x: ['critical', 'high', 'medium', 'low'].index(x['level']))

def generate_recommendations(alerts, df_latest):
    """Generate actionable recommendations"""
    recommendations = {
        'immediate': [],
        'today': [],
        'this_week': []
    }
    
    latest = df_latest.iloc[-1]
    
    # Based on alerts
    critical_alerts = [a for a in alerts if a['level'] == 'critical']
    if critical_alerts:
        for alert in critical_alerts:
            recommendations['immediate'].append({
                'action': alert['action'],
                'reason': alert['title'],
                'time': alert['timeline'],
                'cost': '$0' if 'balancing' in alert['action'].lower() else '$50-100',
                'diy': True if 'balancing' in alert['action'].lower() else False
            })
    
    # Temperature based
    if latest.get('temperature', 25) > 35:
        recommendations['today'].append({
            'action': 'Improve cooling system',
            'reason': 'Elevated operating temperature',
            'time': '30 minutes',
            'cost': '$0-50',
            'diy': True
        })
    
    # Health score based
    health_score = latest.get('battery_health_score', 100)
    if health_score < 80:
        recommendations['this_week'].append({
            'action': 'Professional diagnostic test',
            'reason': f'Health score declined to {health_score:.1f}/100',
            'time': '1-2 hours',
            'cost': '$100-200',
            'diy': False
        })
    
    return recommendations

# Sidebar
with st.sidebar:
    st.markdown("# 🔋 CellGuard.AI")
    st.markdown("### Battery Intelligence Platform")
    
    st.markdown("---")
    
    st.markdown("### 🎯 User Mode")
    user_mode = st.radio(
        "Select Your View",
        ["👤 Driver (Simple)", "🔧 Technician (Advanced)"],
        key="mode_selector"
    )
    st.session_state.user_mode = 'driver' if 'Driver' in user_mode else 'technician'
    
    st.markdown("---")
    
    st.markdown("### 📌 Connection Status")
    connection = st.radio(
        "Data Source",
        ["📡 Live Vehicle BMS", "📁 Historical Data", "🧪 Demo Mode"],
        index=2
    )
    
    if "Live" in connection:
        st.session_state.connection_status = 'online'
        st.markdown('<div class="connection-online">🟢 CONNECTED</div>', unsafe_allow_html=True)
        st.markdown("**MQTT Broker:** 192.168.1.100")
        st.markdown("**Topic:** vehicle/bms/telemetry")
        st.markdown("**Update Rate:** 5 seconds")
    else:
        st.session_state.connection_status = 'offline'
        st.markdown('<div class="connection-offline">🔴 OFFLINE MODE</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.session_state.user_mode == 'technician':
        st.markdown("### ⚙️ Advanced Settings")
        contamination = st.slider("Anomaly Sensitivity", 0.01, 0.2, 0.05, 0.01)
        window = st.slider("Rolling Window", 5, 30, 10)
    else:
        contamination = 0.05
        window = 10
    
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    st.metric("Data Points", "161")
    st.metric("Last Updated", "2 min ago")

# Load or generate data
if st.session_state.battery_data is None:
    st.session_state.battery_data = gen_sample_data_shell()

# Process data
df_raw = st.session_state.battery_data
df_feat = make_features(df_raw, window=window)
df_feat = analyze_cell_balance(df_feat)
df_out = run_models(df_feat, contamination=contamination)

# Calculate metrics
latest = df_out.iloc[-1]
health_score = latest.get('battery_health_score', 50)
soc = latest.get('soc', 0)
temperature = latest.get('temperature', 0)
voltage = latest.get('voltage', 0)
cell_diff = latest.get('cell_diff', 0) * 1000  # Convert to mV
cycle_count = latest.get('Cycle', 0)
anomaly_rate = df_out['anomaly_flag'].mean() * 100

# Generate alerts and recommendations
alerts = generate_alerts(df_out.tail(1), df_out)
recommendations = generate_recommendations(alerts, df_out.tail(1))

# Estimate range
estimated_range = int(soc * 3.5)  # Simple: 1% SOC = 3.5km

# Main Content
st.title("🔋 CellGuard.AI - Battery Intelligence Platform")

# Driver Mode
if st.session_state.user_mode == 'driver':
    # Big Status Card
    if health_score >= 85:
        status_class = 'status-card-healthy'
        status_emoji = '✅'
        status_text = 'ALL GOOD'
        status_msg = 'Your battery is in great shape!'
    elif health_score >= 60:
        status_class = 'status-card-warning'
        status_emoji = '⚠️'
        status_text = 'MONITOR'
        status_msg = 'Some attention needed soon'
    else:
        status_class = 'status-card-critical'
        status_emoji = '🔴'
        status_text = 'SERVICE NEEDED'
        status_msg = 'Service recommended - see details below'
    
    st.markdown(f"""
    <div class="{status_class}">
        <h1 style="font-size: 4rem; margin: 0;">{status_emoji}</h1>
        <h2 style="font-size: 3rem; margin: 0.5rem 0; color: white;">{status_text}</h2>
        <p style="font-size: 1.5rem; margin: 0; color: white;">{status_msg}</p>
        <h1 style="font-size: 5rem; margin: 1rem 0; color: white;">{health_score:.0f}</h1>
        <p style="font-size: 1.2rem; margin: 0; color: white;">Battery Health Score / 100</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; color: #667eea;">⚡ Battery Charge</h3>
            <h1 style="margin: 0.5rem 0; font-size: 3rem;">{soc:.0f}%</h1>
            <p style="margin: 0; color: #718096;">≈ {estimated_range} km range</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        temp_color = '#48bb78' if temperature < 40 else '#ed8936' if temperature < 50 else '#f56565'
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; color: {temp_color};">🌡️ Temperature</h3>
            <h1 style="margin: 0.5rem 0; font-size: 3rem; color: {temp_color};">{temperature:.0f}°C</h1>
            <p style="margin: 0; color: #718096;">{'Normal range' if temperature < 40 else 'Elevated' if temperature < 50 else 'High!'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; color: #805ad5;">🔄 Battery Age</h3>
            <h1 style="margin: 0.5rem 0; font-size: 3rem;">{cycle_count:.0f}</h1>
            <p style="margin: 0; color: #718096;">charge cycles completed</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Alerts Section
    critical_high_alerts = [a for a in alerts if a['level'] in ['critical', 'high']]
    
    if critical_high_alerts:
        st.markdown("## ⚠️ Attention Needed")
        
        for alert in critical_high_alerts:
            alert_class = f"alert-{alert['level']}"
            emoji = '🔴' if alert['level'] == 'critical' else '🟠'
            
            st.markdown(f"""
            <div class="{alert_class}">
                <h3 style="margin: 0 0 0.5rem 0; color: #2d3748;">{emoji} {alert['title']}</h3>
                <p style="margin: 0.5rem 0; color: #2d3748;"><strong>What's happening:</strong> {alert['detail']}</p>
                <div style="background: rgba(255,255,255,0.7); padding: 0.75rem; border-radius: 8px; margin-top: 0.5rem;">
                    <p style="margin: 0; color: #2d3748;"><strong>✅ What to do:</strong> {alert['action']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ No immediate issues detected!")
    
    # Recommendations
    st.markdown("## 🛠️ What You Should Do")
    
    if recommendations['immediate']:
        st.markdown("### 🚨 Do This Now (within 1 hour)")
        for rec in recommendations['immediate']:
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{rec['action']}**")
                    st.caption(f"Reason: {rec['reason']}")
                with col2:
                    st.info(f"⏱️ {rec['time']}")
                    st.info(f"💰 {rec['cost']}")
                    if rec['diy']:
                        st.success("✅ DIY")
                    else:
                        st.warning("👨‍🔧 Pro")
    
    if recommendations['today']:
        st.markdown("### 📅 Do This Today")
        for rec in recommendations['today']:
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{rec['action']}**")
                    st.caption(f"Reason: {rec['reason']}")
                with col2:
                    st.info(f"⏱️ {rec['time']}")
                    st.info(f"💰 {rec['cost']}")
    
    if recommendations['this_week']:
        st.markdown("### 📆 This Week")
        for rec in recommendations['this_week']:
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{rec['action']}**")
                    st.caption(f"Reason: {rec['reason']}")
                with col2:
                    st.info(f"⏱️ {rec['time']}")
                    st.info(f"💰 {rec['cost']}")
    
    # Service Button
    st.markdown("---")
    if st.button("🗺️ Find Nearest Service Center", use_container_width=True):
        st.info("📍 Service center locator will open here (integration needed)")

# Technician Mode
else:
    # Technical Metrics Grid
    st.markdown("## 📊 Technical Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Pack Voltage", f"{voltage:.2f}V", delta=f"{voltage-77.4:.2f}V")
    
    with col2:
        st.metric("Cell ΔV", f"{cell_diff:.0f}mV", 
                 delta="⚠️ High" if cell_diff > 50 else "✅ Normal",
                 delta_color="inverse" if cell_diff > 50 else "normal")
    
    with col3:
        st.metric("SOC", f"{soc:.1f}%")
    
    with col4:
        st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%",
                 delta="⚠️ High" if anomaly_rate > 5 else "✅ Normal",
                 delta_color="inverse" if anomaly_rate > 5 else "normal")
    
    # All Alerts
    st.markdown("## 🚨 Active Alerts")
    
    for alert in alerts:
        with st.expander(f"{['🔴', '🟠', '🟡', '🔵'][['critical', 'high', 'medium', 'low'].index(alert['level'])]} [{alert['level'].upper()}] {alert['title']} - {alert['timestamp']}"):
            st.markdown(f"**Detail:** {alert['detail']}")
            st.markdown(f"**Recommended Action:** {alert['action']}")
            st.markdown(f"**Consequence if ignored:** {alert['consequence']}")
            st.markdown(f"**Timeline:** {alert['timeline']}")
    
    # Cell Voltage Grid
    st.markdown("## 🔋 Individual Cell Voltages (24S Configuration)")
    
    cell_data = []
    for i in range(1, 25):
        col_name = f'Cell{i}'
        if col_name in df_out.columns:
            voltage_val = df_out[col_name].iloc[-1]
            is_weak = i in [3, 17, 22]
            cell_data.append({
                'Cell': i,
                'Voltage': voltage_val,
                'Status': 'Weak' if is_weak else 'Normal'
            })
    
    # Create 3 rows of 8 cells each
    for row in range(3):
        cols = st.columns(8)
        for col_idx in range(8):
            cell_idx = row * 8 + col_idx
            if cell_idx < len(cell_data):
                cell = cell_data[cell_idx]
                with cols[col_idx]:
                    if cell['Status'] == 'Weak':
                        st.error(f"**Cell {cell['Cell']}**\n\n{cell['Voltage']:.3f}V")
                    else:
                        st.success(f"**Cell {cell['Cell']}**\n\n{cell['Voltage']:.3f}V")
    
    st.caption("🟢 Normal (21 cells) | 🔴 Weak/Degraded (3 cells: #3, #17, #22)")
    
    # Temperature Distribution
    st.markdown("## 🌡️ Temperature Distribution (4 Sensors)")
    
    temp_cols = st.columns(4)
    for i, col in enumerate(temp_cols, 1):
        temp_col = f'Temp{i}'
        if temp_col in df_out.columns:
            temp_val = df_out[temp_col].iloc[-1]
            is_hot = i == 2
            with col:
                if is_hot:
                    st.error(f"**Sensor {i}**\n\n{temp_val:.1f}°C\n\n⚠️ Hotspot")
                else:
                    st.info(f"**Sensor {i}**\n\n{temp_val:.1f}°C")
    
    # Charts
    st.markdown("## 📈 Performance Charts")
    
    tab1, tab2, tab3 = st.tabs(["Health Timeline", "Cell Balance", "Temperature"])
    
    with tab1:
        fig_health = px.line(
            df_out, 
            x='time', 
            y='battery_health_score',
            title='Battery Health Score Over Time',
            labels={'time': 'Time', 'battery_health_score': 'Health Score'}
        )
        fig_health.update_layout(
            xaxis_title="Time Steps",
            yaxis_title="Health Score (0-100)",
            hovermode='x unified'
        )
        st.plotly_chart(fig_health, use_container_width=True)
        
        # Add anomaly markers
        anomaly_data = df_out[df_out['anomaly_flag'] == 1]
        if len(anomaly_data) > 0:
            st.warning(f"⚠️ {len(anomaly_data)} anomalies detected in timeline")
    
    with tab2:
        # Cell balance over time
        cell_cols = [c for c in df_out.columns if c.startswith('Cell')]
        if len(cell_cols) > 0:
            fig_cells = go.Figure()
            
            # Plot weak cells in red
            for i in [3, 17, 22]:
                col_name = f'Cell{i}'
                if col_name in df_out.columns:
                    fig_cells.add_trace(go.Scatter(
                        x=df_out['time'],
                        y=df_out[col_name],
                        mode='lines',
                        name=f'Cell {i} (Weak)',
                        line=dict(color='red', width=2)
                    ))
            
            # Plot normal cells in gray (sample a few)
            for i in [1, 5, 10, 15, 20, 24]:
                col_name = f'Cell{i}'
                if col_name in df_out.columns and i not in [3, 17, 22]:
                    fig_cells.add_trace(go.Scatter(
                        x=df_out['time'],
                        y=df_out[col_name],
                        mode='lines',
                        name=f'Cell {i}',
                        line=dict(color='lightgray', width=1),
                        opacity=0.5
                    ))
            
            # Add cell difference line
            fig_cells.add_trace(go.Scatter(
                x=df_out['time'],
                y=df_out['cell_diff'],
                mode='lines',
                name='Cell ΔV',
                line=dict(color='orange', width=2, dash='dash'),
                yaxis='y2'
            ))
            
            fig_cells.update_layout(
                title='Cell Voltage Balance Over Time',
                xaxis_title='Time Steps',
                yaxis_title='Cell Voltage (V)',
                yaxis2=dict(
                    title='Cell ΔV (V)',
                    overlaying='y',
                    side='right'
                ),
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig_cells, use_container_width=True)
            
            # Summary stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Cell Voltage", f"{df_out['cell_mean'].iloc[-1]:.3f}V")
            with col2:
                st.metric("Max Cell Difference", f"{df_out['cell_diff'].iloc[-1]*1000:.1f}mV")
            with col3:
                st.metric("Cell Std Dev", f"{df_out['cell_std'].iloc[-1]*1000:.1f}mV")
    
    with tab3:
        # Temperature sensors over time
        temp_sensor_cols = [c for c in df_out.columns if c.startswith('Temp')]
        if len(temp_sensor_cols) > 0:
            fig_temp = go.Figure()
            
            colors = ['blue', 'red', 'green', 'purple']
            for idx, temp_col in enumerate(temp_sensor_cols):
                is_hot = '2' in temp_col
                fig_temp.add_trace(go.Scatter(
                    x=df_out['time'],
                    y=df_out[temp_col],
                    mode='lines',
                    name=f'{temp_col} {"(Hotspot)" if is_hot else ""}',
                    line=dict(color=colors[idx], width=2 if is_hot else 1)
                ))
            
            # Add average temperature line
            fig_temp.add_trace(go.Scatter(
                x=df_out['time'],
                y=df_out['temperature'],
                mode='lines',
                name='Pack Average',
                line=dict(color='black', width=2, dash='dot')
            ))
            
            fig_temp.update_layout(
                title='Temperature Distribution Over Time',
                xaxis_title='Time Steps',
                yaxis_title='Temperature (°C)',
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig_temp, use_container_width=True)
            
            # Temperature stats
            col1, col2, col3, col4 = st.columns(4)
            for idx, col in enumerate([col1, col2, col3, col4], 1):
                temp_col = f'Temp{idx}'
                if temp_col in df_out.columns:
                    temp_val = df_out[temp_col].iloc[-1]
                    with col:
                        st.metric(f"Sensor {idx}", f"{temp_val:.1f}°C")
    
    # Data Export
    st.markdown("---")
    st.markdown("## 💾 Data Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Download CSV Report", use_container_width=True):
            csv = df_out.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"cellguard_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📊 Generate PDF Report", use_container_width=True):
            st.info("📄 PDF generation feature coming soon")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: white; padding: 2rem;">
    <p style="font-size: 0.9rem; opacity: 0.8;">
        CellGuard.AI v1.0 | Battery Intelligence Platform<br>
        © 2024 | For support: support@cellguard.ai
    </p>
</div>
""", unsafe_allow_html=True)