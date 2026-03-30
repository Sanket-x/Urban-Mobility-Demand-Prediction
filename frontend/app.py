import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import requests
import os

# ------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & THEME
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Ola Mobility Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------------
# 2. CUSTOM CSS FOR INDUSTRY-GRADE UI
# ------------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0E1117;
    color: #FAFAFA;
}

h1, h2, h3 {
    font-weight: 700 !important;
}

h1 {
    background: -webkit-linear-gradient(45deg, #00FFAE, #00B8FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0px !important;
    padding-bottom: 0px !important;
}

.subtitle {
    color: #A0AEC0;
    font-size: 1.1rem;
    font-weight: 400;
    margin-top: 5px;
    margin-bottom: 25px;
}

.custom-card {
    background: rgba(30, 35, 41, 0.6);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    box-shadow: 0 4px 16px 0 rgba(0, 0, 0, 0.2);
    backdrop-filter: blur(8px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.custom-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px 0 rgba(0, 255, 174, 0.1);
    border: 1px solid rgba(0, 255, 174, 0.2);
}

.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #FAFAFA;
}

.metric-label {
    font-size: 0.9rem;
    color: #A0AEC0;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
}

.insight-box {
    background: rgba(0, 184, 255, 0.05);
    border-left: 3px solid #00B8FF;
    padding: 12px 16px;
    border-radius: 4px 8px 8px 4px;
    margin-top: 15px;
    font-size: 0.9rem;
    color: #E2E8F0;
    line-height: 1.5;
}

.badge {
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    display: inline-block;
}
.badge-success { background-color: rgba(0, 255, 174, 0.15); color: #00FFAE; }
.badge-warning { background-color: rgba(255, 193, 7, 0.15); color: #FFC107; }
.badge-danger { background-color: rgba(255, 71, 87, 0.15); color: #FF4757; }

hr { border-top: 1px solid rgba(255, 255, 255, 0.1); }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 3. DATA LOADING & PREPROCESSING
# ------------------------------------------------------------------------------
# Define mapping globally so API prediction module can securely do reverse lookups!
GLOBAL_AREA_MAP = {
    "Area-1": "Indiranagar", "Area-2": "Koramangala", "Area-3": "Whitefield", "Area-4": "HSR Layout",
    "Area-5": "Electronic City", "Area-6": "Jayanagar", "Area-7": "JP Nagar", "Area-8": "BTM Layout",
    "Area-9": "Marathahalli", "Area-10": "Bellandur", "Area-11": "MG Road", "Area-12": "Malleshwaram",
    "Area-13": "Hebbal", "Area-14": "Yelahanka", "Area-15": "Banashankari", "Area-16": "Basavanagudi",
    "Area-17": "Rajajinagar", "Area-18": "RT Nagar", "Area-19": "CV Raman Nagar", "Area-20": "Kengeri",
    "Area-21": "Peenya", "Area-22": "Yeshwanthpur", "Area-23": "Malleswaram", "Area-24": "Vijayanagar",
    "Area-25": "Sadashivanagar", "Area-26": "Frazer Town", "Area-27": "Ulsoor", "Area-28": "Richmond Town",
    "Area-29": "Sanjay Nagar", "Area-30": "Mathikere", "Area-31": "Mahadevapura", "Area-32": "KR Puram",
    "Area-33": "Devanahalli", "Area-34": "Banaskankari", "Area-35": "Kumaraswamy Layout", "Area-36": "Nagawara",
    "Area-37": "Hennur", "Area-38": "Kalyan Nagar", "Area-39": "Horamavu", "Area-40": "Ramamurthy Nagar",
    "Area-41": "Kasturi Nagar", "Area-42": "Domlur", "Area-43": "Ejipura", "Area-44": "Adugodi",
    "Area-45": "Benson Town", "Area-46": "Cox Town", "Area-47": "Shivajinagar", "Area-48": "Vasanth Nagar",
    "Area-49": "Shanthi Nagar", "Area-50": "Wilson Garden"
}

@st.cache_data
def load_data():
    # Adjust path if needed based on the run location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "data", "Bengaluru Ola.csv")
    
    df = pd.read_csv(data_path)

    # Reference global mapping
    area_map = GLOBAL_AREA_MAP
    df['Pickup Location'] = df['Pickup Location'].map(area_map).fillna(df['Pickup Location'])
    if 'Drop Location' in df.columns:
        df['Drop Location'] = df['Drop Location'].map(area_map).fillna(df['Drop Location'])

    
    # Ensure Date and Time are properly parsed
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
    df['Date_Obj'] = df['Datetime'].dt.date
    df['Hour'] = df['Datetime'].dt.hour
    df['Day_of_Week'] = df['Datetime'].dt.dayofweek
    df['Day_Name'] = df['Day_of_Week'].map({
        0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
        4: 'Friday', 5: 'Saturday', 6: 'Sunday'
    })
    df['Is_Weekend'] = df['Day_of_Week'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
    
    # Fill categorical nulls
    df['Payment Method'] = df['Payment Method'].fillna('Unknown')
    
    # Feature Engineering for mapping
    # Assuming 'Success', 'Cancelled by Driver', 'Cancelled by Customer', 'Incomplete'
    df['Is_Success'] = (df['Booking Status'] == 'Success').astype(int)
    
    return df

df = load_data()

# ------------------------------------------------------------------------------
# 4. CHART STYLING HELPER
# ------------------------------------------------------------------------------
def get_chart_layout(title=""):
    return dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20),
        title_font=dict(size=18, family="Inter", color="#FAFAFA"),
        font=dict(family="Inter", color="#A0AEC0"),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        hovermode="x unified",
        title=title
    )

COLOR_PALETTE = ["#00FFAE", "#00B8FF", "#A855F7", "#FF4757", "#FFC107", "#F97316"]

# ------------------------------------------------------------------------------
# 5. NAVIGATION (SIDEBAR)
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Mobility Intelligence")
    st.markdown("<p style='color:#64748B; font-size:0.85rem;'>Enterprise Analytics Platform</p>", unsafe_allow_html=True)
    st.markdown("<hr/>", unsafe_allow_html=True)
    
    page = st.radio(
        "Navigation",
        [
            "🏠 Executive Summary",
            "⏰ Time Intelligence",
            "📍 Location Intelligence",
            "🚗 Vehicle Intelligence",
            "💳 Payment Analysis",
            "❌ Cancellation Intelligence",
            "📍 Area Intelligence",
            "🔮 Predict Demand"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("<hr/>", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 6. PAGE ROUTING & RENDERING
# ------------------------------------------------------------------------------

if page == "🏠 Executive Summary":
    st.markdown("<h1>Platform Overview</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>High-level metrics and system health for the Bengaluru region.</p>", unsafe_allow_html=True)
    
    # 1. KPI row
    col1, col2, col3, col4 = st.columns(4)
    total_bookings = len(df)
    success_rate = (df['Booking Status'] == 'Success').mean() * 100
    avg_distance = df['Ride Distance'].mean()
    total_revenue = df[df['Booking Status'] == 'Success']['Booking Value'].sum()
    
    with col1:
        st.markdown(f"""
        <div class="custom-card">
            <div class="metric-label">Total Volume</div>
            <div class="metric-value">{total_bookings:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="custom-card">
            <div class="metric-label">Completion Rate</div>
            <div class="metric-value" style="color:#00FFAE;">{success_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="custom-card">
            <div class="metric-label">Avg Distance</div>
            <div class="metric-value">{avg_distance:.1f} km</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="custom-card">
            <div class="metric-label">Estimated Revenue</div>
            <div class="metric-value">₹{total_revenue/1000000:.1f}M</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br/>", unsafe_allow_html=True)
    
    c1, c2 = st.columns([1, 2])
    
    with c1:
        status_counts = df['Booking Status'].value_counts().reset_index()
        fig_donut = px.pie(
            status_counts, 
            values='count', 
            names='Booking Status',
            hole=0.6,
            color='Booking Status',
            color_discrete_map={
                'Success': '#00FFAE',
                'Cancelled by Driver': '#FF4757',
                'Cancelled by Customer': '#FFC107',
                'Incomplete': '#A855F7'
            }
        )
        fig_donut.update_layout(
            **get_chart_layout("Booking Outcomes"),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown("""
        <div class="insight-box">
            <b>Insight:</b> Over 66% of rides are successfully completed, while driver cancellations constitute the largest loss sector (19%).
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        daily_vol = df.groupby('Date_Obj').size().reset_index(name='Requests')
        fig_area = px.area(
            daily_vol, x='Date_Obj', y='Requests',
            color_discrete_sequence=['#00B8FF']
        )
        fig_area.update_layout(**get_chart_layout("Daily Volume Trend"))
        fig_area.update_traces(fill='tozeroy', fillcolor='rgba(0, 184, 255, 0.2)')
        st.plotly_chart(fig_area, use_container_width=True)
        st.markdown("""
        <div class="insight-box">
            <b>Insight:</b> Demand exhibits strong weekly seasonality, with distinct peaks forming every 7 days. Consistency in overall volume suggests stable market penetration.
        </div>
        """, unsafe_allow_html=True)


elif page == "⏰ Time Intelligence":
    st.markdown("<h1>Time Intelligence</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Temporal patterns of passenger demand and network utilization.</p>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    
    with c1:
        hourly_demand = df.groupby('Hour').size().reset_index(name='Demand')
        fig_hr = px.line(
            hourly_demand, x='Hour', y='Demand',
            markers=True,
            color_discrete_sequence=['#A855F7']
        )
        fig_hr.update_layout(**get_chart_layout("Demand vs Hour (24H)"))
        fig_hr.update_traces(fill='tozeroy', fillcolor='rgba(168, 85, 247, 0.2)')
        st.plotly_chart(fig_hr, use_container_width=True)
        st.markdown("""<div class="insight-box"><b>Insight:</b> Demand peaks twice daily—morning rush (8–10 AM) and evening return (5–8 PM), indicating a strong commuter-driven user base.</div>""", unsafe_allow_html=True)
        
    with c2:
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_demand = df.groupby('Day_Name').size().reindex(day_order).reset_index(name='Demand')
        fig_day = px.bar(
            day_demand, y='Day_Name', x='Demand', orientation='h',
            color='Demand', color_continuous_scale='Purp'
        )
        fig_day.update_layout(**get_chart_layout("Demand vs Day of Week"))
        fig_day.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_day, use_container_width=True)
        st.markdown("""<div class="insight-box"><b>Insight:</b> Weekends (Sat/Sun) show highest aggregate demand, suggesting significant leisure and social mobility outside standard working hours.</div>""", unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    
    with c3:
        daily_df = df.groupby('Date_Obj').size().reset_index(name='Demand')
        daily_df['Rolling 7-Day'] = daily_df['Demand'].rolling(window=7).mean()
        
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Bar(x=daily_df['Date_Obj'], y=daily_df['Demand'], name='Daily', marker_color='rgba(0, 184, 255, 0.4)'))
        fig_ts.add_trace(go.Scatter(x=daily_df['Date_Obj'], y=daily_df['Rolling 7-Day'], name='7-Day Avg', line=dict(color='#00FFAE', width=3)))
        
        fig_ts.update_layout(**get_chart_layout("Time Series Trend with Rolling Average"))
        st.plotly_chart(fig_ts, use_container_width=True)
        st.markdown("""<div class="insight-box"><b>Insight:</b> The 7-day rolling average remains remarkably stable, implying that driver supply should be allocated uniformly across weeks without major structural shifts.</div>""", unsafe_allow_html=True)

    with c4:
        heatmap_data = df.groupby(['Day_Name', 'Hour']).size().unstack().reindex(day_order)
        fig_hm = px.imshow(
            heatmap_data, 
            labels=dict(x="Hour of Day", y="Day of Week", color="Demand"),
            x=heatmap_data.columns,
            y=heatmap_data.index,
            aspect="auto",
            color_continuous_scale="Mint"
        )
        fig_hm.update_layout(**get_chart_layout("Heatmap: Hour vs Day"))
        st.plotly_chart(fig_hm, use_container_width=True)
        st.markdown("""<div class="insight-box"><b>Insight:</b> Hotspots appear distinctly on Friday and Saturday evenings, highlighting the optimal windows for surge pricing and driver incentives.</div>""", unsafe_allow_html=True)
        
    c_hist, c_wknd = st.columns(2)
    with c_hist:
        # Histogram of Demand Distribution
        st.markdown("**Demand Distribution (Requests/Hour)**")
        # To get requests per hour (aggregated) we already did groupby Date and Hour
        hx = df.groupby(['Date_Obj', 'Hour']).size().reset_index(name='Requests')
        fig_dist = px.histogram(
            hx, x='Requests', nbins=30,
            color_discrete_sequence=['#FFC107'],
            marginal='box'
        )
        fig_dist.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#A0AEC0"), yaxis_title="Frequency",
            height=300
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        st.markdown("""<p style="font-size:0.85rem;color:#A0AEC0;"><b>Insight:</b> Most base hours see ~50-80 requests. The long tail indicates extreme surge periods.</p>""", unsafe_allow_html=True)

    with c_wknd:
        st.markdown("**Weekend vs Weekday Demand Profile**")
        wknd_hr = df.groupby(['Hour', 'Is_Weekend']).size().reset_index(name='Requests')
        fig_wknd = px.line(
            wknd_hr, x='Hour', y='Requests', color='Is_Weekend',
            color_discrete_map={'Weekday':'#00B8FF', 'Weekend':'#00FFAE'},
            markers=True
        )
        fig_wknd.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#A0AEC0"), height=300,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_wknd, use_container_width=True)
        st.markdown("""<p style="font-size:0.85rem;color:#A0AEC0;"><b>Insight:</b> Weekdays have sharp morning peaks, while weekends show sustained, growing demand into the late night.</p>""", unsafe_allow_html=True)


elif page == "📍 Location Intelligence":
    st.markdown("<h1>Location Intelligence</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Spatial dynamics, geographical hotspots, and routing efficiency.</p>", unsafe_allow_html=True)
    
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        top_loc = df['Pickup Location'].value_counts().head(15).reset_index()
        top_loc.columns = ['Location', 'Requests']
        fig_loc = px.bar(
            top_loc, x='Requests', y='Location', orientation='h',
            color='Requests', color_continuous_scale='Blues'
        )
        fig_loc.update_layout(**get_chart_layout("Top 15 Demand Locations"))
        fig_loc.update_yaxes(categoryorder='total ascending')
        fig_loc.update_coloraxes(showscale=False)
        st.plotly_chart(fig_loc, use_container_width=True)
        st.markdown("""<div class="insight-box"><b>Insight:</b> Area-39 and Area-4 consistently drive the most traffic. Fleets should be pre-positioned nearby during shift starts.</div>""", unsafe_allow_html=True)

    with col_t2:
        # Calculate cancellation rate by location
        cancels = df[df['Booking Status'].isin(['Cancelled by Customer', 'Cancelled by Driver'])]
        cancel_rates = (cancels.groupby('Pickup Location').size() / df.groupby('Pickup Location').size() * 100).fillna(0).reset_index(name='Cancel_Rate_%')
        high_cancel_loc = cancel_rates.sort_values(by='Cancel_Rate_%', ascending=False).head(10)
        
        fig_c_loc = px.bar(
            high_cancel_loc, x='Pickup Location', y='Cancel_Rate_%',
            color='Cancel_Rate_%', color_continuous_scale='Reds',
            text_auto='.1f'
        )
        fig_c_loc.update_layout(**get_chart_layout("High Cancellation Locations (%)"))
        fig_c_loc.update_coloraxes(showscale=False)
        st.plotly_chart(fig_c_loc, use_container_width=True)
        st.markdown("""<div class="insight-box"><b>Insight:</b> Certain nodes experience higher friction. Interventions (e.g., driver mapping education, improved pickup zones) are required for these critical failure points.</div>""", unsafe_allow_html=True)
        
    st.markdown("### Location vs Peak Hour Density")
    # Finding the peak hour for top 20 locations
    top_20 = top_loc['Location'].head(20).tolist()
    loc_hr = df[df['Pickup Location'].isin(top_20)].groupby(['Pickup Location', 'Hour']).size().reset_index(name='Volume')
    idx = loc_hr.groupby(['Pickup Location'])['Volume'].transform(max) == loc_hr['Volume']
    peak_loc_hr = loc_hr[idx].drop_duplicates('Pickup Location')
    
    fig_bubble = px.scatter(
        peak_loc_hr, x='Hour', y='Pickup Location', size='Volume', color='Volume',
        color_continuous_scale='Viridis', size_max=30
    )
    fig_bubble.update_layout(**get_chart_layout(""))
    fig_bubble.update_xaxes(tickmode='linear', dtick=1, range=[-1, 24])
    st.plotly_chart(fig_bubble, use_container_width=True)
    st.markdown("""<div class="insight-box"><b>Insight:</b> Office parks peak strictly at evening hours (17:00-19:00), whereas residential hubs peak in the morning. Supply must physically shift across the grid mid-day.</div>""", unsafe_allow_html=True)


elif page == "🚗 Vehicle Intelligence":
    st.markdown("<h1>Vehicle Intelligence</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Fleet composition, product mix, and asset utilization.</p>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1.5])
    
    with c1:
        veh_dist = df['Vehicle Type'].value_counts().reset_index(name='Count')
        fig_veh_d = px.pie(
            veh_dist, values='Count', names='Vehicle Type', hole=0.5,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_veh_d.update_layout(**get_chart_layout("Vehicle Mix Distribution"))
        st.plotly_chart(fig_veh_d, use_container_width=True)
        st.markdown("""<div class="insight-box"><b>Insight:</b> A highly balanced fleet portfolio ensures resilience. Bikes and Autos form the dense short-trip backbone, while Prime caters to margin-heavy users.</div>""", unsafe_allow_html=True)
        
    with c2:
        v_h = df.groupby(['Hour', 'Vehicle Type']).size().reset_index(name='Demand')
        fig_vh = px.line(
            v_h, x='Hour', y='Demand', color='Vehicle Type', markers=True,
            line_shape='spline'
        )
        fig_vh.update_layout(**get_chart_layout("Demand by Vehicle Type per Hour"))
        st.plotly_chart(fig_vh, use_container_width=True)
        st.markdown("""<div class="insight-box"><b>Insight:</b> Bike demand climbs sharply during extreme peak hours due to traffic congestion evasion. Autos maintain steady baseline volume throughout the day.</div>""", unsafe_allow_html=True)
        
    st.markdown("### Vehicle Preference by Top Locations")
    top_10 = df['Pickup Location'].value_counts().head(10).index
    df_v_l = df[df['Pickup Location'].isin(top_10)].groupby(['Pickup Location', 'Vehicle Type']).size().reset_index(name='Count')
    
    fig_vl = px.bar(
        df_v_l, x='Pickup Location', y='Count', color='Vehicle Type',
        barmode='group'
    )
    fig_vl.update_layout(**get_chart_layout(""))
    st.plotly_chart(fig_vl, use_container_width=True)
    st.markdown("""<div class="insight-box"><b>Insight:</b> Micro-markets show specific affinities. High-density tech park zones prefer Prime/Mini configurations, while traditional markets heavily lean towards Autos and Bikes.</div>""", unsafe_allow_html=True)

elif page == "💳 Payment Analysis":
    st.markdown("<h1>Payment Analytics</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Financial interactions, frictionless checkout, and value flow.</p>", unsafe_allow_html=True)
    
    # Filter out unknowns for pristine analysis if necessary
    df_pay = df[df['Payment Method'] != 'Unknown']
    
    c1, c2 = st.columns(2)
    
    with c1:
        pay_dist = df_pay['Payment Method'].value_counts().reset_index(name='Count')
        fig_pd = px.pie(
            pay_dist, values='Count', names='Payment Method',
            color='Payment Method',
            color_discrete_map={'Cash':'#FFC107', 'UPI':'#00FFAE', 'Wallet':'#00B8FF', 'Card':'#A855F7'}
        )
        fig_pd.update_layout(**get_chart_layout("Payment Landscape"))
        st.plotly_chart(fig_pd, use_container_width=True)
        st.markdown("""<div class="insight-box"><b>Insight:</b> Cash and UPI dominate the transaction mix, emphasizing the need for immediate liquidity features for drivers and robust QR integrations.</div>""", unsafe_allow_html=True)
        
    with c2:
        pay_hr = df_pay.groupby(['Hour', 'Payment Method']).size().reset_index(name='Count')
        fig_ph = px.area(
            pay_hr, x='Hour', y='Count', color='Payment Method',
            color_discrete_map={'Cash':'#FFC107', 'UPI':'#00FFAE', 'Wallet':'#00B8FF', 'Card':'#A855F7'}
        )
        fig_ph.update_layout(**get_chart_layout("Payment Preferences Over Time"))
        st.plotly_chart(fig_ph, use_container_width=True)
        st.markdown("""<div class="insight-box"><b>Insight:</b> Digital payments (UPI/Wallet) spike heavily during morning rush hours (speed of checkout), while cash reliance grows during late-night localized trips.</div>""", unsafe_allow_html=True)

    st.markdown("### Average Booking Value vs Payment Method")
    success_rev = df_pay[df_pay['Booking Status'] == 'Success']
    pay_val = success_rev.groupby('Payment Method')['Booking Value'].mean().reset_index()
    
    fig_pv = px.bar(
        pay_val, x='Payment Method', y='Booking Value', color='Payment Method',
        text_auto='.0f', color_discrete_map={'Cash':'#FFC107', 'UPI':'#00FFAE', 'Wallet':'#00B8FF', 'Card':'#A855F7'}
    )
    fig_pv.update_layout(**get_chart_layout(""))
    st.plotly_chart(fig_pv, use_container_width=True)
    st.markdown("""<div class="insight-box"><b>Insight:</b> Card and Wallet users yield higher average order values (AOV). Shifting users to captive wallets via cashback lowers processing costs while increasing revenue.</div>""", unsafe_allow_html=True)

elif page == "❌ Cancellation Intelligence":
    st.markdown("<h1>Cancellation Intelligence</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Identifying friction, failure demand, and network leakage.</p>", unsafe_allow_html=True)
    
    cancellations = df[df['Booking Status'].str.contains('Cancel', na=False, case=False)]
    
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    c_rate = (len(cancellations) / len(df)) * 100
    driver_cancels = df['Cancelled Rides by Driver'].sum() if 'Cancelled Rides by Driver' in df.columns else len(df[df['Booking Status'] == 'Cancelled by Driver'])
    cust_cancels = df['Cancelled  by Customer'].sum() if 'Cancelled  by Customer' in df.columns else len(df[df['Booking Status'] == 'Cancelled by Customer'])
    
    with col_kpi1:
        st.markdown(f"""
        <div class="custom-card" style="text-align:center;">
            <div class="metric-label">Global Cancellation Rate</div>
            <div class="metric-value" style="color:#FF4757;">{c_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col_kpi2:
        st.markdown(f"""
        <div class="custom-card" style="text-align:center;">
            <div class="metric-label">Defected by Driver</div>
            <div class="metric-value" style="color:#F97316;">{driver_cancels:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_kpi3:
        st.markdown(f"""
        <div class="custom-card" style="text-align:center;">
            <div class="metric-label">Defected by Customer</div>
            <div class="metric-value" style="color:#FFC107;">{cust_cancels:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1.2])
    
    with c1:
        comparison = pd.DataFrame({
            "Source": ["Driver", "Customer"],
            "Cancellations": [driver_cancels, cust_cancels]
        })
        fig_cc = px.pie(
            comparison, values='Cancellations', names='Source', hole=0.6,
            color='Source', color_discrete_map={"Driver":"#F97316", "Customer":"#FFC107"}
        )
        fig_cc.update_layout(**get_chart_layout("Driver vs Customer Comparison"))
        st.plotly_chart(fig_cc, use_container_width=True)
        st.markdown("""<div class="insight-box"><b>Insight:</b> Drivers cancel significantly more often than customers. Core issue likely stems from dispatch distances, destination aversion, or upfront pricing opacity.</div>""", unsafe_allow_html=True)

    with c2:
        canc_hr = cancellations.groupby('Hour').size().reset_index(name='Drop-offs')
        fig_ch = px.bar(
            canc_hr, x='Hour', y='Drop-offs',
            color='Drop-offs', color_continuous_scale='Reds'
        )
        fig_ch.update_layout(**get_chart_layout("Cancellations by Hour"))
        fig_ch.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_ch, use_container_width=True)
        st.markdown("""<div class="insight-box"><b>Insight:</b> Cancellations correlate almost perfectly with peak demand hours. The system experiences high load and ETAs bloat, leading to immense user frustration and driver abandonment.</div>""", unsafe_allow_html=True)
        
    # Attempting to plot cancellation reasons if the column is well populated
    if 'Reason for Cancelling by Driver' in cancellations.columns and len(cancellations['Reason for Cancelling by Driver'].dropna()) > 0:
        top_reasons = cancellations['Reason for Cancelling by Driver'].value_counts().reset_index()
        top_reasons.columns = ['Reason', 'Count']
        # filter out 'na'
        top_reasons = top_reasons[top_reasons['Reason'].str.lower() != 'na'].head(6)
        
        fig_c_rsn = px.bar(top_reasons, y='Reason', x='Count', orientation='h', color='Count', color_continuous_scale='Sunsetdark')
        fig_c_rsn.update_layout(**get_chart_layout("Top Driver Cancellation Reasons"))
        fig_c_rsn.update_yaxes(categoryorder='total ascending')
        st.plotly_chart(fig_c_rsn, use_container_width=True)
        st.markdown("""<div class="insight-box"><b>Insight:</b> Investigating explicit reasons helps tailor penalty models and algorithmic dispatch rules.</div>""", unsafe_allow_html=True)


elif page == "📍 Area Intelligence":
    st.markdown("<h1>📍 Area Intelligence</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Deep-dive localization analytics for micro-markets.</p>", unsafe_allow_html=True)
    
    locations = sorted([loc for loc in df['Pickup Location'].unique() if pd.notna(loc)])
    if not locations:
        locations = ["Area-" + str(i) for i in range(1, 51)]
    
    selected_area = st.selectbox("Select Area to Analyze", locations, index=0)
    
    df_area = df[df['Pickup Location'] == selected_area]
    
    if len(df_area) == 0:
        st.warning(f"No data available for {selected_area}")
    else:
        # SECTION 1: AREA SUMMARY
        st.markdown("### Area Summary")
        c1, c2, c3, c4 = st.columns(4)
        
        total_rides = len(df_area)
        cancels = df_area[df_area['Booking Status'].str.contains('Cancel', na=False, case=False)]
        cancel_rate = (len(cancels) / total_rides * 100) if total_rides > 0 else 0
        
        hourly = df_area.groupby('Hour').size()
        peak_hour = hourly.idxmax() if not hourly.empty else "N/A"
        
        vehicles = df_area['Vehicle Type'].value_counts()
        most_used_veh = vehicles.idxmax() if not vehicles.empty else "N/A"
        
        with c1: st.metric("Total Rides", f"{total_rides:,}")
        with c2: st.metric("Cancellation Rate", f"{cancel_rate:.1f}%")
        with c3: st.metric("Peak Demand Hour", f"{peak_hour}:00")
        with c4: st.metric("Top Vehicle", most_used_veh)
            
        st.markdown("<br/>", unsafe_allow_html=True)
        
        # SECTION 2: DEMAND ANALYSIS
        st.markdown("### Demand Analysis")
        cd1, cd2 = st.columns(2)
        with cd1:
            hr_demand = df_area.groupby('Hour').size().reset_index(name='Demand')
            f_hr = px.line(hr_demand, x='Hour', y='Demand', markers=True, color_discrete_sequence=['#00FFAE'])
            f_hr.update_layout(**get_chart_layout("Demand vs Hour"))
            st.plotly_chart(f_hr, use_container_width=True)
        with cd2:
            day_demand = df_area.groupby('Day_Name').size().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']).reset_index(name='Demand')
            f_day = px.bar(day_demand, x='Day_Name', y='Demand', color='Demand', color_continuous_scale='Mint')
            f_day.update_layout(**get_chart_layout("Demand vs Day of Week"))
            f_day.update_coloraxes(showscale=False)
            st.plotly_chart(f_day, use_container_width=True)
            
        ts_demand = df_area.groupby('Date_Obj').size().reset_index(name='Demand')
        f_ts = px.line(ts_demand, x='Date_Obj', y='Demand', color_discrete_sequence=['#00B8FF'])
        f_ts.update_layout(**get_chart_layout("Time Series Trend"))
        st.plotly_chart(f_ts, use_container_width=True)
        
        st.markdown("<hr/>", unsafe_allow_html=True)
        
        # SECTION 3: VEHICLE INTELLIGENCE
        st.markdown("### Vehicle Intelligence")
        cv1, cv2 = st.columns(2)
        with cv1:
            vd = df_area['Vehicle Type'].value_counts().reset_index()
            vd.columns = ['Vehicle', 'Count']
            f_vd = px.pie(vd, names='Vehicle', values='Count', hole=0.5)
            f_vd.update_layout(**get_chart_layout("Vehicle Distribution"))
            st.plotly_chart(f_vd, use_container_width=True)
        with cv2:
            vh = df_area.groupby(['Hour', 'Vehicle Type']).size().reset_index(name='Demand')
            f_vh = px.line(vh, x='Hour', y='Demand', color='Vehicle Type', markers=True)
            f_vh.update_layout(**get_chart_layout("Vehicle Demand vs Hour"))
            st.plotly_chart(f_vh, use_container_width=True)
            
        st.markdown("<hr/>", unsafe_allow_html=True)
        
        # SECTION 4: CANCELLATION ANALYSIS
        st.markdown("### Cancellation Analysis")
        cc1, cc2 = st.columns(2)
        with cc1:
            driver_c = len(df_area[df_area['Booking Status'] == 'Cancelled by Driver'])
            cust_c = len(df_area[df_area['Booking Status'] == 'Cancelled by Customer'])
            f_cc = px.pie(names=['Driver', 'Customer'], values=[driver_c, cust_c], color_discrete_sequence=['#F97316', '#FFC107'])
            f_cc.update_layout(**get_chart_layout("Cancel Source"))
            st.plotly_chart(f_cc, use_container_width=True)
        with cc2:
            ch = cancels.groupby('Hour').size().reset_index(name='Cancellations')
            f_ch = px.bar(ch, x='Hour', y='Cancellations', color_discrete_sequence=['#FF4757'])
            f_ch.update_layout(**get_chart_layout("Cancellations vs Hour"))
            st.plotly_chart(f_ch, use_container_width=True)
            
        # Top Cancellation reasons
        cc3, cc4 = st.columns(2)
        with cc3:
            if 'Reason for Cancelling by Driver' in df_area.columns:
                dr_rsn = cancels['Reason for Cancelling by Driver'].dropna().value_counts().head(5).reset_index()
                dr_rsn.columns = ['Reason', 'Count']
                dr_rsn = dr_rsn[dr_rsn['Reason'].str.lower() != 'na']
                if not dr_rsn.empty:
                    f_dr = px.bar(dr_rsn, y='Reason', x='Count', orientation='h', color='Count', color_continuous_scale='Reds')
                    f_dr.update_layout(**get_chart_layout("Top Driver Reasons"))
                    f_dr.update_yaxes(categoryorder='total ascending')
                    f_dr.update_coloraxes(showscale=False)
                    st.plotly_chart(f_dr, use_container_width=True)
        with cc4:
            if 'Reason for Cancelling by Customer' in df_area.columns:
                cr_rsn = cancels['Reason for Cancelling by Customer'].dropna().value_counts().head(5).reset_index()
                cr_rsn.columns = ['Reason', 'Count']
                cr_rsn = cr_rsn[cr_rsn['Reason'].str.lower() != 'na']
                if not cr_rsn.empty:
                    f_cr = px.bar(cr_rsn, y='Reason', x='Count', orientation='h', color='Count', color_continuous_scale='Oranges')
                    f_cr.update_layout(**get_chart_layout("Top Customer Reasons"))
                    f_cr.update_yaxes(categoryorder='total ascending')
                    f_cr.update_coloraxes(showscale=False)
                    st.plotly_chart(f_cr, use_container_width=True)
                    
        st.markdown("<hr/>", unsafe_allow_html=True)
        
        # SECTION 5: PAYMENT ANALYSIS
        st.markdown("### Payment Analysis")
        cp1, cp2 = st.columns(2)
        df_pay_a = df_area[df_area['Payment Method'] != 'Unknown']
        with cp1:
            pd_area = df_pay_a['Payment Method'].value_counts().reset_index()
            pd_area.columns = ['Method', 'Count']
            f_pd = px.pie(pd_area, names='Method', values='Count')
            f_pd.update_layout(**get_chart_layout("Payment Distribution"))
            st.plotly_chart(f_pd, use_container_width=True)
        with cp2:
            pt = df_pay_a.groupby(['Date_Obj', 'Payment Method']).size().reset_index(name='Count')
            f_pt = px.area(pt, x='Date_Obj', y='Count', color='Payment Method')
            f_pt.update_layout(**get_chart_layout("Payment Trend Over Time"))
            st.plotly_chart(f_pt, use_container_width=True)
            
        success_rev = df_pay_a[df_pay_a['Booking Status'] == 'Success']
        if not success_rev.empty:
            pay_val = success_rev.groupby('Payment Method')['Booking Value'].mean().reset_index()
            f_pv = px.bar(pay_val, x='Payment Method', y='Booking Value', color='Payment Method', text_auto='.0f')
            f_pv.update_layout(**get_chart_layout("Average Booking Value vs Payment Method"))
            st.plotly_chart(f_pv, use_container_width=True)
            
        st.markdown("<hr/>", unsafe_allow_html=True)
        
        # SECTION 6: INSIGHT ENGINE
        st.markdown("### 🧠 Insight Engine (AI)")
        insights = []
        
        # Avoid division by zero
        avg_overall = len(df) / df['Pickup Location'].nunique() if df['Pickup Location'].nunique() > 0 else 0
        
        if total_rides > avg_overall * 1.5:
            insights.append("🔥 <b>High Demand Hub:</b> This area experiences significantly higher traffic than network average. Suggest allocating dedicated driver supply during peak hours.")
            
        if cancel_rate > 20:
            insights.append("⚠️ <b>Critical Friction:</b> Cancellation rates exceed 20%. Consider adjusting surge multipliers or investigating poor spatial pick-up points.")
            
        if most_used_veh == 'Bike':
            insights.append("🔄 <b>Fleet Optimization:</b> High Bike usage detected. Ensure micro-mobility partner density is maintained up to a 2km radius to satisfy sub-15 min ETAs.")
            
        if not insights:
            insights.append("✅ <b>Stable Market:</b> Operational metrics in this area are tracking safely within nominal boundaries. Continue standard operational rules.")
            
        for insight in insights:
            st.markdown(f"<div class='insight-box'>{insight}</div>", unsafe_allow_html=True)



elif page == "🔮 Predict Demand":
    # ------------------------------------------------------------------------------
    # PRESERVED PREDICTION FUNCTIONALITY (Upgraded UI wrapper)
    # ------------------------------------------------------------------------------
    st.markdown("<h1>🔮 Predict Engine</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Live ML-driven demand forecasting via FastAPI backend.</p>", unsafe_allow_html=True)
    
    API_URL = "https://urban-mobility-demand-prediction.onrender.com/predict"

    DAYS_MAP = {
        0: "Monday", 1: "Tuesday", 2: "Wednesday",
        3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"
    }
    
    # We can autogenerate Bangalore locations from the dataframe dynamically!
    locations = sorted([loc for loc in df['Pickup Location'].unique() if pd.notna(loc)])
    if not locations:
        locations = ["Area-" + str(i) for i in range(1, 51)]

    def format_hour(h):
        if h == 0: return "12 AM (Midnight)"
        if h < 12: return f"{h} AM"
        if h == 12: return "12 PM (Noon)"
        return f"{h-12} PM"

    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None

    c_input, c_output = st.columns([1, 1.2])
    
    with c_input:
        st.markdown("### ⚙️ Simulation Variables")
        
        hour = st.slider("Select Hour", min_value=0, max_value=23, value=17, format="%d", help="0 is Midnight, 23 is 11 PM")
        st.caption(f"**Selected Time:** {format_hour(hour)}")
        
        day_name = st.selectbox("Select Day of Week", list(DAYS_MAP.values()), index=4)
        day_idx = list(DAYS_MAP.keys())[list(DAYS_MAP.values()).index(day_name)]
        
        location_display = st.selectbox("Select Pickup Area", locations, index=0)
        
        st.markdown("<br/>", unsafe_allow_html=True)
        
        if st.button("🚀 Run Live Inference", use_container_width=True):
            with st.spinner("🧠 Connecting to API..."):
                try:
                    # Reverse map generic location back to 'Area-X' for backend model
                    REVERSE_AREA_MAP = {v: k for k, v in GLOBAL_AREA_MAP.items()}
                    backend_location = REVERSE_AREA_MAP.get(location_display, location_display)
                
                    payload = {
                        "hour": hour,
                        "day_of_week": day_idx,
                        "location": backend_location
                    }
                    response = requests.post(API_URL, json=payload, timeout=30)
                    
                    if response.status_code == 200:
                        st.session_state.prediction_result = response.json()
                        st.toast("✅ Inference completed!")
                    else:
                        st.error(f"⚠️ API Error: {response.text}")
                        st.session_state.prediction_result = None
                except requests.exceptions.ConnectionError:
                    st.error("🚫 Backend not responding. Check FastAPI server.")
                    st.session_state.prediction_result = None
        
    with c_output:
        if st.session_state.prediction_result:
            res = st.session_state.prediction_result
            demand_val = res.get("predicted_demand", 0)
            demand_str = res.get("demand_level", "Unknown")
            
            badge_class = "badge-warning"
            emoji = "🟡"
            if "low" in demand_str.lower():
                badge_class = "badge-success"
                emoji = "🟢"
            elif "high" in demand_str.lower():
                badge_class = "badge-danger"
                emoji = "🔴"
                
            st.markdown(f"""
                <div style="text-align:center;">
                    <h3 style="color:#A0AEC0; margin-bottom:10px;">Predicted Volume</h3>
                    <div style="font-size: 5rem; font-weight: 800; color: #FAFAFA; line-height:1; margin-bottom: 20px;">
                        {demand_val:,.1f}
                    </div>
                    <div style="margin-bottom: 20px;">
                        <span style="color:#A0AEC0; margin-right: 15px;">Network Status:</span>
                        <span class="badge {badge_class}" style="font-size: 1.2rem; padding: 8px 20px;">{emoji} {demand_str.upper()} DEMAND</span>
                    </div>
                    <p style="color: #64748B; font-size: 0.9rem;">Powered by Random Forest Regressor Matrix</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="text-align: center; color: #64748B;">
                    <h1 style="font-size: 4rem; opacity: 0.5;">📡</h1>
                    <p>Awaiting inference parameters...</p>
                </div>
            """, unsafe_allow_html=True)




# ------------------------------------------------------------------------------
# 7. FOOTER LOGIC
# ------------------------------------------------------------------------------
def render_footer():
    st.markdown("""
        <style>
        .custom-footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: rgba(14, 17, 23, 0.85); /* Matches dark UI with opacity */
            color: #64748B;
            text-align: center;
            padding: 12px 0;
            font-size: 0.85rem;
            border-top: 1px solid rgba(255, 255, 255, 0.05); /* Slight separation line */
            z-index: 9999;
            backdrop-filter: blur(8px);
            text-shadow: 0 0 5px rgba(255, 255, 255, 0.05); /* Glow effect */
        }
        
        /* Ensure streamlit content doesn't get hidden behind the sticky footer */
        .block-container {
            padding-bottom: 70px !important;
        }
        </style>
        
        <div class="custom-footer">
            🚖 Urban Mobility AI System • Built by Sanket Thakore & Mridul Goswami • Powered by Machine Learning & FastAPI
        </div>
    """, unsafe_allow_html=True)

# Call the footer at the very end of app execution
render_footer()
