"""
PILLAR 3: Predictive Visibility
"""

import pandas as pd
from sklearn.ensemble import IsolationForest

class PredictiveVisibilityService:
    """Predict disruptions"""
    
    def __init__(self, df):
        self.df = df
    
    def detect_anomalies(self):
        """Find unusual patterns"""
        features = self.df[['delivery_cost' if 'delivery_cost' in self.df.columns else self.df.columns[0], 
                            'delivery_time_hours' if 'delivery_time_hours' in self.df.columns else self.df.columns[1], 
                            'distance_km' if 'distance_km' in self.df.columns else self.df.columns[2]]].values
        
        iso = IsolationForest(contamination=0.05, random_state=42)
        anomalies = iso.fit_predict(features)
        
        return {
            'anomaly_count': int((anomalies == -1).sum()),
            'anomaly_percentage': float((anomalies == -1).sum() / len(self.df) * 100)
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


if __name__ == "__main__":
    df = pd.read_csv('data/processed/cleaned_data.csv')
    predictive = PredictiveVisibilityService(df)
    print(predictive.get_disruption_actions())
    