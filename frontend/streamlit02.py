"""
GreenRoute Analytics - Interactive Dashboard
Frontend for Logistics Optimization Platform
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="GreenRoute Analytics",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .recommendation-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2E7D32;
        margin-bottom: 1rem;
    }
    .priority-high {
        color: #d32f2f;
        font-weight: bold;
    }
    .priority-medium {
        color: #f57c00;
        font-weight: bold;
    }
    .priority-low {
        color: #388e3c;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Import backend modules (these would be your actual backend files)
# For this demo, I'll include the classes inline
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

class InefficencyAnalyzer:
    """Find operational inefficiencies"""
    
    def __init__(self, df):
        self.df = df.copy()
    
    def find_inefficient_routes(self):
        """Identify inefficient routes (top 25% by cost/km)"""
        threshold = self.df['cost_per_km'].quantile(0.75)
        inefficient = self.df[self.df['cost_per_km'] > threshold]
        
        return {
            'total_inefficient': len(inefficient),
            'percentage': round((len(inefficient) / len(self.df)) * 100, 2),
            'wasted_cost_total': float(inefficient['Delivery_Cost'].sum()) if 'Delivery_Cost' in self.df.columns else 0,
            'wasted_co2_total_kg': float(inefficient['co2_emissions_kg'].sum()),
            'saveable_cost_25pct': float(inefficient['Delivery_Cost'].sum() * 0.25) if 'Delivery_Cost' in self.df.columns else 0,
            'reduceable_co2_25pct_kg': float(inefficient['co2_emissions_kg'].sum() * 0.25)
        }
    
    def cluster_routes(self, n_clusters=8):
        """Group inefficient routes into clusters"""
        inefficient_indices = self.df[self.df['is_inefficient_route'] == 1].index
        
        if len(inefficient_indices) == 0:
            return []
        
        cost_col = 'Delivery_Cost' if 'Delivery_Cost' in self.df.columns else self.df.columns[0]
        distance_col = 'Distance' if 'Distance' in self.df.columns else self.df.columns[1]
        time_col = 'Delivery_Time_Hours' if 'Delivery_Time_Hours' in self.df.columns else self.df.columns[2]
        
        features = self.df.loc[inefficient_indices, [cost_col, distance_col, time_col]].values
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        kmeans = KMeans(n_clusters=min(n_clusters, len(inefficient_indices)), random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features_scaled)
        
        cluster_info = []
        for cluster_id in np.unique(clusters):
            cluster_mask = clusters == cluster_id
            cluster_data = self.df.loc[inefficient_indices[cluster_mask]]
            
            cluster_info.append({
                'cluster_id': int(cluster_id),
                'routes_in_cluster': len(cluster_data),
                'total_cost': float(cluster_data[cost_col].sum()),
                'total_co2_kg': float(cluster_data['co2_emissions_kg'].sum()),
                'avg_efficiency': float(cluster_data['efficiency_score'].mean())
            })
        
        return cluster_info
    
    def get_recommendations(self):
        """Generate recommendations"""
        inefficiencies = self.find_inefficient_routes()
        clusters = self.cluster_routes()
        
        return [
            {
                'rank': 1,
                'action': 'Consolidate Inefficient Routes',
                'problem': f"{inefficiencies['total_inefficient']:,} routes are inefficient ({inefficiencies['percentage']:.1f}%)",
                'solution': f"Group into {len(clusters)} clusters and consolidate",
                'impact_cost': f"Save ₹{inefficiencies['saveable_cost_25pct']:,.0f}/month",
                'impact_co2': f"Reduce {inefficiencies['reduceable_co2_25pct_kg']:,.0f} kg CO2/month",
                'priority': 'HIGH'
            }
        ]


class LastMileOptimizer:
    """Evaluate sustainable last-mile options"""
    
    def __init__(self, df):
        self.df = df
        self.baseline_cost = df['Delivery_Cost'].mean() if 'Delivery_Cost' in df.columns else df.iloc[:, 0].mean()
        self.baseline_time = df['Delivery_Time_Hours'].mean() if 'Delivery_Time_Hours' in df.columns else df.iloc[:, 1].mean()
        self.baseline_co2 = df['co2_emissions_kg'].mean()
    
    def evaluate_options(self):
        """Rank 4 delivery methods"""
        
        options = {
            'Home Delivery': {'cost': self.baseline_cost, 'co2': self.baseline_co2, 'time': self.baseline_time, 'satisfaction': 4.0},
            'Pickup Points': {'cost': self.baseline_cost * 0.80, 'co2': self.baseline_co2 * 0.60, 'time': self.baseline_time * 1.5, 'satisfaction': 3.8},
            'Electric Vehicles': {'cost': self.baseline_cost * 1.03, 'co2': self.baseline_co2 * 0.40, 'time': self.baseline_time * 0.95, 'satisfaction': 4.3},
            'Micro-Hubs': {'cost': self.baseline_cost * 0.70, 'co2': self.baseline_co2 * 0.45, 'time': self.baseline_time * 1.3, 'satisfaction': 3.9}
        }
        
        scores = {}
        for name, opt in options.items():
            co2_improvement = (self.baseline_co2 - opt['co2']) / self.baseline_co2
            cost_improvement = (self.baseline_cost - opt['cost']) / self.baseline_cost
            satisfaction_score = opt['satisfaction'] / 5.0
            
            total_score = (co2_improvement * 0.50) + (cost_improvement * 0.30) + (satisfaction_score * 0.20)
            
            scores[name] = {
                'score': total_score,
                'co2_reduction': co2_improvement * 100,
                'cost_change': (opt['cost'] - self.baseline_cost) / self.baseline_cost * 100,
                'satisfaction': opt['satisfaction'],
                'cost': opt['cost'],
                'co2': opt['co2'],
                'time': opt['time']
            }
        
        return sorted(scores.items(), key=lambda x: x[1]['score'], reverse=True)
    
    def get_recommendation(self):
        """Get best method"""
        ranking = self.evaluate_options()
        best = ranking[0]
        
        return {
            'primary_method': best[0],
            'primary_score': float(best[1]['score']),
            'co2_reduction_percent': float(best[1]['co2_reduction']),
            'full_ranking': [{'rank': i+1, 'method': name, 'score': float(scores['score'])} for i, (name, scores) in enumerate(ranking)]
        }


class PredictiveVisibilityService:
    """Predict disruptions"""
    
    def __init__(self, df):
        self.df = df
    
    def detect_anomalies(self):
        """Find unusual patterns"""
        features = self.df[['Delivery_Cost' if 'Delivery_Cost' in self.df.columns else self.df.columns[0], 
                            'Delivery_Time_Hours' if 'Delivery_Time_Hours' in self.df.columns else self.df.columns[1], 
                            'Distance' if 'Distance' in self.df.columns else self.df.columns[2]]].values
        
        iso = IsolationForest(contamination=0.05, random_state=42)
        anomalies = iso.fit_predict(features)
        
        return {
            'anomaly_count': int((anomalies == -1).sum()),
            'anomaly_percentage': float((anomalies == -1).sum() / len(self.df) * 100),
            'anomaly_mask': anomalies == -1
        }
    
    def get_disruption_actions(self):
        """Return prevention actions"""
        anomalies = self.detect_anomalies()
        
        return [
            {
                'disruption_type': 'Weather Impact',
                'problem': 'Bad weather causes delays',
                'action': 'Pre-position inventory before weather',
                'priority': 'HIGH'
            },
            {
                'disruption_type': 'Route Anomalies',
                'problem': f"{anomalies['anomaly_count']} routes are anomalous",
                'action': 'Monitor and flag unusual patterns',
                'priority': 'MEDIUM'
            }
        ]


# Helper functions for data generation
def generate_sample_data(n_records=500):
    """Generate sample delivery data"""
    np.random.seed(42)
    
    data = {
        'delivery_id': [f'DEL{str(i).zfill(5)}' for i in range(1, n_records + 1)],
        'Distance': np.random.uniform(5, 50, n_records),
        'Delivery_Time_Hours': np.random.uniform(0.5, 8, n_records),
        'Vehicle_Type': np.random.choice(['Small', 'Medium', 'Large'], n_records, p=[0.4, 0.4, 0.2]),
        'Delivery_Cost': np.random.uniform(100, 1500, n_records),
        'co2_emissions_kg': np.random.uniform(2, 25, n_records),
        'efficiency_score': np.random.uniform(0.4, 0.95, n_records),
    }
    
    df = pd.DataFrame(data)
    df['cost_per_km'] = df['Delivery_Cost'] / df['Distance']
    df['is_inefficient_route'] = (df['cost_per_km'] > df['cost_per_km'].quantile(0.75)).astype(int)
    
    return df


# Main Application
def main():
    # Header
    st.markdown('<div class="main-header">🚛 GreenRoute Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Data-Driven Route Optimization for Sustainable Logistics</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 Dashboard Controls")
    st.sidebar.markdown("---")
    
    # File upload or sample data
    data_source = st.sidebar.radio(
        "Select Data Source:",
        ["Use Sample Data", "Upload CSV"]
    )
    
    if data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Upload your delivery data", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"✓ Loaded {len(df)} records")
        else:
            st.info("👈 Please upload a CSV file or select 'Use Sample Data'")
            return
    else:
        n_records = st.sidebar.slider("Number of sample records:", 100, 1000, 500, 50)
        df = generate_sample_data(n_records)
        st.sidebar.success(f"✓ Generated {len(df)} sample records")
    
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["🏠 Overview Dashboard", "📈 Route Inefficiency Analysis", "🚚 Last-Mile Optimization", 
         "🔮 Predictive Analytics", "📋 Detailed Data View"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "**GreenRoute Analytics** helps logistics companies reduce costs and emissions "
        "through AI-powered route optimization and predictive analytics."
    )
    
    # Initialize analyzers
    inefficiency_analyzer = InefficencyAnalyzer(df)
    lastmile_optimizer = LastMileOptimizer(df)
    predictive_service = PredictiveVisibilityService(df)
    
    # Page routing
    if page == "🏠 Overview Dashboard":
        show_overview_dashboard(df, inefficiency_analyzer, lastmile_optimizer, predictive_service)
    elif page == "📈 Route Inefficiency Analysis":
        show_inefficiency_analysis(df, inefficiency_analyzer)
    elif page == "🚚 Last-Mile Optimization":
        show_lastmile_optimization(df, lastmile_optimizer)
    elif page == "🔮 Predictive Analytics":
        show_predictive_analytics(df, predictive_service)
    elif page == "📋 Detailed Data View":
        show_detailed_data(df)


def show_overview_dashboard(df, inefficiency_analyzer, lastmile_optimizer, predictive_service):
    """Overview dashboard with key metrics"""
    
    st.header("📊 Executive Dashboard")
    st.markdown("Real-time insights into your logistics operations")
    
    # Get analysis results
    inefficiency_results = inefficiency_analyzer.find_inefficient_routes()
    lastmile_results = lastmile_optimizer.get_recommendation()
    anomaly_results = predictive_service.detect_anomalies()
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Routes",
            value=f"{len(df):,}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Inefficient Routes",
            value=f"{inefficiency_results['total_inefficient']:,}",
            delta=f"-{inefficiency_results['percentage']}%",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="Monthly Savings Potential",
            value=f"₹{inefficiency_results['saveable_cost_25pct']:,.0f}",
            delta="25% optimization"
        )
    
    with col4:
        st.metric(
            label="CO₂ Reduction Potential",
            value=f"{inefficiency_results['reduceable_co2_25pct_kg']:,.0f} kg",
            delta="Per month"
        )
    
    st.markdown("---")
    
    # Charts Row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📉 Route Efficiency Distribution")
        
        # Efficiency score histogram
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df['efficiency_score'],
            nbinsx=30,
            marker_color='#2E7D32',
            name='Efficiency Score'
        ))
        fig.update_layout(
            xaxis_title="Efficiency Score",
            yaxis_title="Number of Routes",
            showlegend=False,
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💰 Cost vs Distance Analysis")
        
        # Scatter plot
        fig = px.scatter(
            df,
            x='Distance',
            y='Delivery_Cost',
            color='is_inefficient_route',
            color_discrete_map={0: '#4CAF50', 1: '#f44336'},
            labels={'is_inefficient_route': 'Route Type'},
            height=300
        )
        fig.update_layout(showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Top Recommendations
    st.subheader("🎯 Top Priority Actions")
    
    recommendations = inefficiency_analyzer.get_recommendations()
    
    for rec in recommendations:
        with st.container():
            st.markdown(f"""
            <div class="recommendation-card">
                <h4>🔴 {rec['action']}</h4>
                <p><strong>Problem:</strong> {rec['problem']}</p>
                <p><strong>Solution:</strong> {rec['solution']}</p>
                <p><strong>Impact:</strong> {rec['impact_cost']} • {rec['impact_co2']}</p>
                <span class="priority-high">Priority: {rec['priority']}</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Last Mile Recommendation
    st.markdown(f"""
    <div class="recommendation-card">
        <h4>🚚 Best Last-Mile Strategy: {lastmile_results['primary_method']}</h4>
        <p><strong>Optimization Score:</strong> {lastmile_results['primary_score']:.2f}</p>
        <p><strong>CO₂ Reduction:</strong> {lastmile_results['co2_reduction_percent']:.1f}%</p>
        <span class="priority-medium">Priority: MEDIUM</span>
    </div>
    """, unsafe_allow_html=True)


def show_inefficiency_analysis(df, analyzer):
    """Detailed inefficiency analysis page"""
    
    st.header("📈 Route Inefficiency Analysis")
    st.markdown("Identify and quantify operational inefficiencies")
    
    # Get results
    results = analyzer.find_inefficient_routes()
    clusters = analyzer.cluster_routes()
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Inefficient Routes",
            f"{results['total_inefficient']:,}",
            f"{results['percentage']:.1f}% of total"
        )
    
    with col2:
        st.metric(
            "Wasted Cost",
            f"₹{results['wasted_cost_total']:,.0f}",
            "Current monthly waste"
        )
    
    with col3:
        st.metric(
            "Excess CO₂",
            f"{results['wasted_co2_total_kg']:,.0f} kg",
            "Environmental impact"
        )
    
    st.markdown("---")
    
    # Cluster Analysis
    st.subheader("🎯 Route Clusters")
    st.markdown("Inefficient routes grouped by similarity for targeted optimization")
    
    if clusters:
        cluster_df = pd.DataFrame(clusters)
        
        # Cluster visualization
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"Cluster {c['cluster_id']}" for c in clusters],
            y=[c['routes_in_cluster'] for c in clusters],
            marker_color='#FF6B6B',
            name='Routes per Cluster'
        ))
        fig.update_layout(
            xaxis_title="Cluster ID",
            yaxis_title="Number of Routes",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Cluster details table
        st.subheader("📊 Cluster Details")
        display_df = cluster_df.copy()
        display_df['total_cost'] = display_df['total_cost'].apply(lambda x: f"₹{x:,.0f}")
        display_df['total_co2_kg'] = display_df['total_co2_kg'].apply(lambda x: f"{x:,.0f} kg")
        display_df['avg_efficiency'] = display_df['avg_efficiency'].apply(lambda x: f"{x:.2f}")
        st.dataframe(display_df, use_container_width=True)
    else:
        st.info("No inefficient routes detected. Your operations are running optimally!")
    
    st.markdown("---")
    
    # Cost per km distribution
    st.subheader("💸 Cost per Kilometer Distribution")
    
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=df['cost_per_km'],
        marker_color='#4A90E2',
        name='Cost/km'
    ))
    fig.update_layout(
        yaxis_title="Cost per Kilometer (₹)",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Inefficient routes table
    st.subheader("🔍 Most Inefficient Routes (Top 20)")
    inefficient_routes = df[df['is_inefficient_route'] == 1].nlargest(20, 'cost_per_km')
    st.dataframe(inefficient_routes[['delivery_id', 'Distance', 'Delivery_Cost', 'cost_per_km', 'co2_emissions_kg']], use_container_width=True)


def show_lastmile_optimization(df, optimizer):
    """Last-mile delivery optimization page"""
    
    st.header("🚚 Last-Mile Delivery Optimization")
    st.markdown("Evaluate and compare sustainable delivery methods")
    
    # Get evaluation results
    evaluation = optimizer.evaluate_options()
    recommendation = optimizer.get_recommendation()
    
    # Best method highlight
    st.success(f"✨ **Recommended Strategy:** {recommendation['primary_method']} (Score: {recommendation['primary_score']:.2f})")
    
    st.markdown("---")
    
    # Comparison metrics
    st.subheader("📊 Delivery Method Comparison")
    
    methods = []
    scores = []
    co2_reductions = []
    cost_changes = []
    satisfactions = []
    
    for method, metrics in evaluation:
        methods.append(method)
        scores.append(metrics['score'])
        co2_reductions.append(metrics['co2_reduction'])
        cost_changes.append(metrics['cost_change'])
        satisfactions.append(metrics['satisfaction'])
    
    # Score comparison bar chart
    fig = go.Figure()
    colors = ['#2E7D32' if method == recommendation['primary_method'] else '#757575' for method in methods]
    fig.add_trace(go.Bar(
        x=methods,
        y=scores,
        marker_color=colors,
        text=[f"{s:.2f}" for s in scores],
        textposition='outside'
    ))
    fig.update_layout(
        title="Overall Optimization Score",
        xaxis_title="Delivery Method",
        yaxis_title="Score",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Detailed metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌱 CO₂ Reduction %")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=methods,
            y=co2_reductions,
            marker_color='#4CAF50',
            text=[f"{cr:.1f}%" for cr in co2_reductions],
            textposition='outside'
        ))
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("😊 Customer Satisfaction")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=methods,
            y=satisfactions,
            marker_color='#2196F3',
            text=[f"{s:.1f}/5" for s in satisfactions],
            textposition='outside'
        ))
        fig.update_layout(height=350, yaxis_range=[0, 5])
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Detailed comparison table
    st.subheader("📋 Detailed Comparison")
    
    comparison_data = []
    for method, metrics in evaluation:
        comparison_data.append({
            'Delivery Method': method,
            'Score': f"{metrics['score']:.2f}",
            'CO₂ Reduction': f"{metrics['co2_reduction']:.1f}%",
            'Cost Change': f"{metrics['cost_change']:.1f}%",
            'Satisfaction': f"{metrics['satisfaction']:.1f}/5",
            'Cost': f"₹{metrics['cost']:.0f}",
            'CO₂': f"{metrics['co2']:.2f} kg",
            'Time': f"{metrics['time']:.2f} hrs"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Recommendations
    st.markdown("---")
    st.subheader("💡 Implementation Recommendations")
    
    best_method = recommendation['primary_method']
    
    recommendations = {
        'Electric Vehicles': [
            "Invest in EV charging infrastructure at distribution centers",
            "Start with 20% fleet conversion in urban areas",
            "Partner with local governments for charging station access",
            "Monitor battery performance and range optimization"
        ],
        'Pickup Points': [
            "Identify high-density delivery zones for pickup points",
            "Partner with retail stores and convenience stores",
            "Implement user-friendly pickup notification system",
            "Offer incentives for customers choosing pickup options"
        ],
        'Micro-Hubs': [
            "Establish micro-hubs in strategic urban locations",
            "Use smaller vehicles for final-mile from hubs",
            "Optimize inventory distribution across hubs",
            "Implement real-time tracking between hubs"
        ],
        'Home Delivery': [
            "Optimize route planning algorithms",
            "Implement time-slot delivery options",
            "Train drivers on eco-driving techniques",
            "Use data analytics to predict delivery patterns"
        ]
    }
    
    for i, rec in enumerate(recommendations.get(best_method, []), 1):
        st.markdown(f"**{i}.** {rec}")


def show_predictive_analytics(df, service):
    """Predictive analytics and anomaly detection page"""
    
    st.header("🔮 Predictive Analytics")
    st.markdown("Detect anomalies and predict potential disruptions")
    
    # Get anomaly results
    anomaly_results = service.detect_anomalies()
    disruption_actions = service.get_disruption_actions()
    
    # Anomaly metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Anomalous Routes",
            f"{anomaly_results['anomaly_count']:,}",
            f"{anomaly_results['anomaly_percentage']:.1f}% of total"
        )
    
    with col2:
        st.metric(
            "Normal Routes",
            f"{len(df) - anomaly_results['anomaly_count']:,}",
            f"{100 - anomaly_results['anomaly_percentage']:.1f}%"
        )
    
    with col3:
        risk_level = "HIGH" if anomaly_results['anomaly_percentage'] > 7 else "MEDIUM" if anomaly_results['anomaly_percentage'] > 4 else "LOW"
        st.metric(
            "Risk Level",
            risk_level,
            "Based on anomaly rate"
        )
    
    st.markdown("---")
    
    # Anomaly visualization
    st.subheader("🎯 Anomaly Detection Results")
    
    df_viz = df.copy()
    df_viz['is_anomaly'] = anomaly_results['anomaly_mask']
    
    fig = px.scatter(
        df_viz,
        x='Distance',
        y='Delivery_Cost',
        color='is_anomaly',
        color_discrete_map={False: '#4CAF50', True: '#f44336'},
        labels={'is_anomaly': 'Anomaly Status'},
        title="Cost vs Distance with Anomaly Detection",
        height=500
    )
    fig.update_layout(
        xaxis_title="Distance (km)",
        yaxis_title="Delivery Cost (₹)"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Disruption actions
    st.subheader("⚠️ Potential Disruptions & Actions")
    
    for action in disruption_actions:
        priority_class = f"priority-{action['priority'].lower()}"
        st.markdown(f"""
        <div class="recommendation-card">
            <h4>🔔 {action['disruption_type']}</h4>
            <p><strong>Problem:</strong> {action['problem']}</p>
            <p><strong>Recommended Action:</strong> {action['action']}</p>
            <span class="{priority_class}">Priority: {action['priority']}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Time series anomaly pattern (simulated)
    st.subheader("📈 Anomaly Trends Over Time")
    
    # Generate time series data
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    anomaly_counts = np.random.poisson(lam=anomaly_results['anomaly_count']/10, size=30)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=anomaly_counts,
        mode='lines+markers',
        name='Anomalies Detected',
        line=dict(color='#f44336', width=2),
        marker=dict(size=8)
    ))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of Anomalies",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Anomalous routes table
    st.subheader("🔍 Detected Anomalous Routes")
    anomalous_routes = df[anomaly_results['anomaly_mask']]
    if len(anomalous_routes) > 0:
        st.dataframe(
            anomalous_routes[['delivery_id', 'Distance', 'Delivery_Cost', 'Delivery_Time_Hours', 'co2_emissions_kg']].head(20),
            use_container_width=True
        )
    else:
        st.info("No anomalous routes detected!")


def show_detailed_data(df):
    """Detailed data view page"""
    
    st.header("📋 Detailed Data View")
    st.markdown("Explore and filter your delivery data")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        vehicle_filter = st.multiselect(
            "Vehicle Type",
            options=df['Vehicle_Type'].unique(),
            default=df['Vehicle_Type'].unique()
        )
    
    with col2:
        efficiency_filter = st.slider(
            "Efficiency Score Range",
            float(df['efficiency_score'].min()),
            float(df['efficiency_score'].max()),
            (float(df['efficiency_score'].min()), float(df['efficiency_score'].max()))
        )
    
    with col3:
        route_type = st.radio(
            "Route Type",
            ["All Routes", "Efficient Only", "Inefficient Only"]
        )
    
    # Apply filters
    filtered_df = df[df['Vehicle_Type'].isin(vehicle_filter)]
    filtered_df = filtered_df[
        (filtered_df['efficiency_score'] >= efficiency_filter[0]) &
        (filtered_df['efficiency_score'] <= efficiency_filter[1])
    ]
    
    if route_type == "Efficient Only":
        filtered_df = filtered_df[filtered_df['is_inefficient_route'] == 0]
    elif route_type == "Inefficient Only":
        filtered_df = filtered_df[filtered_df['is_inefficient_route'] == 1]
    
    st.markdown(f"**Showing {len(filtered_df)} of {len(df)} routes**")
    
    # Display data
    st.dataframe(filtered_df, use_container_width=True, height=600)
    
    st.markdown("---")
    
    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name=f"greenroute_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    # Summary statistics
    st.subheader("📊 Summary Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Numerical Summary**")
        st.dataframe(filtered_df.describe(), use_container_width=True)
    
    with col2:
        st.write("**Categorical Summary**")
        st.write(f"**Vehicle Types:**")
        st.dataframe(filtered_df['Vehicle_Type'].value_counts(), use_container_width=True)


# Run the app
if __name__ == "__main__":
    main()