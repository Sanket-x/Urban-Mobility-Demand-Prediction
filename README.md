Urban-Mobility-Demand-Prediction

An AI-powered analytics and demand prediction platform designed for ride-hailing services such as Uber and Ola. The system focuses on understanding demand patterns, operational inefficiencies, and enabling data-driven decision making.

Features
Time series analysis of ride demand
Area-wise intelligence dashboard for location-specific insights
Vehicle demand and usage analysis
Cancellation analysis (driver vs customer behavior)
Payment method analytics
Machine learning-based demand prediction using FastAPI
Key Highlights
Provides area-level operational insights instead of only global trends
Includes a rule-based insight engine for generating actionable recommendations
Estimates supply-demand gaps for better fleet management
Designed with a business-focused analytics approach
Tech Stack
Python
Streamlit
FastAPI
Pandas, NumPy
Matplotlib, Seaborn, Plotly
Machine Learning (Random Forest, XGBoost)
How to Run
Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
Frontend
cd frontend
streamlit run app.py
Use Case

This system can help ride-hailing companies:

Optimize driver allocation based on demand patterns
Reduce ride cancellations by identifying problem areas
Improve overall customer experience
Enable data-driven operational decisions
