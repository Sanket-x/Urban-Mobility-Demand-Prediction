import sys
import re

file_path = r'd:\College\DS\TS\frontend\app.py'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

nav_search = """            "❌ Cancellation Intelligence",
            "🔮 Predict Demand"
        ],"""
nav_replace = """            "❌ Cancellation Intelligence",
            "📍 Area Intelligence",
            "🔮 Predict Demand"
        ],"""
content = content.replace(nav_search, nav_replace)

predict_target = 'elif page == "🔮 Predict Demand":'

area_code = """elif page == "📍 Area Intelligence":
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

"""

content = content.replace(predict_target, area_code + "\n" + predict_target)

api_call_search = """                try:
                    payload = {
                        "hour": hour,
                        "day_of_week": day_idx,
                        "location": location_display
                    }
                    response = requests.post(API_URL, json=payload, timeout=5)"""
                    
api_call_replace = """                try:
                    # Reverse map generic location back to 'Area-X' for backend model
                    REVERSE_AREA_MAP = {v: k for k, v in area_map.items()} if 'area_map' in globals() else {}
                    backend_location = REVERSE_AREA_MAP.get(location_display, location_display)
                
                    payload = {
                        "hour": hour,
                        "day_of_week": day_idx,
                        "location": backend_location
                    }
                    response = requests.post(API_URL, json=payload, timeout=5)"""

content = content.replace(api_call_search, api_call_replace)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Insertion complete.")
