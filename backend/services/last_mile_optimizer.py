"""
PILLAR 2: Last-Mile Delivery Optimization
"""

import pandas as pd

class LastMileOptimizer:
    """Evaluate sustainable last-mile options"""
    
    def __init__(self, df):
        self.df = df
        self.baseline_cost = df['delivery_cost'].mean() if 'delivery_cost' in df.columns else df.iloc[:, 0].mean()
        self.baseline_time = df['delivery_time_hours'].mean() if 'delivery_time_hours' in df.columns else df.iloc[:, 1].mean()
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
                'satisfaction': opt['satisfaction']
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


if __name__ == "__main__":
    df = pd.read_csv('data/processed/cleaned_data.csv')
    optimizer = LastMileOptimizer(df)
    print(optimizer.get_recommendation())