"""
 Route Optimization
Identify and quantify inefficiencies
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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
            'wasted_cost_total': float(inefficient['delivery_cost'].sum()) if 'delivery_cost' in self.df.columns else 0,
            'wasted_co2_total_kg': float(inefficient['co2_emissions_kg'].sum()),
            'saveable_cost_25pct': float(inefficient['delivery_cost'].sum() * 0.25) if 'delivery_cost' in self.df.columns else 0,
            'reduceable_co2_25pct_kg': float(inefficient['co2_emissions_kg'].sum() * 0.25)
        }
    
    def cluster_routes(self, n_clusters=8):
        """Group inefficient routes into clusters"""
        inefficient_indices = self.df[self.df['is_inefficient_route'] == 1].index
        
        if len(inefficient_indices) == 0:
            return []
        
        cost_col = 'delivery_cost' if 'delivery_cost' in self.df.columns else self.df.columns[0]
        distance_col = 'distance_km' if 'distance_km' in self.df.columns else self.df.columns[1]
        time_col = 'delivery_time_hours' if 'delivery_time_hours' in self.df.columns else self.df.columns[2]
        
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


if __name__ == "__main__":
    df = pd.read_csv('data/processed/cleaned_data.csv')
    analyzer = InefficencyAnalyzer(df)
    print(analyzer.get_recommendations())