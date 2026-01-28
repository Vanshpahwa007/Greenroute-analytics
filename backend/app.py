

from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import os

from services.inefficiency_analyzer import InefficencyAnalyzer
from services.last_mile_optimizer import LastMileOptimizer
from services.predictive_service import PredictiveVisibilityService

app = Flask(__name__)
CORS(app)

DATA_PATH = 'data/processed/cleaned_data.csv'

df = pd.read_csv(DATA_PATH) if os.path.exists(DATA_PATH) else None

if df is not None:
    print(f"✅ Loaded {len(df):,} deliveries")
    analyzer = InefficencyAnalyzer(df)
    optimizer = LastMileOptimizer(df)
    predictive = PredictiveVisibilityService(df)
else:
    print("❌ Data not found")

# ============ ENDPOINTS ============

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'data_loaded': df is not None}), 200

@app.route('/api/inefficiencies', methods=['GET'])
def get_inefficiencies():
    if df is None:
        return jsonify({'error': 'Data not loaded'}), 500
    return jsonify({
        'inefficiencies': analyzer.find_inefficient_routes(),
        'recommendations': analyzer.get_recommendations()
    }), 200

@app.route('/api/last-mile-comparison', methods=['GET'])
def get_last_mile():
    if df is None:
        return jsonify({'error': 'Data not loaded'}), 500
    return jsonify(optimizer.get_recommendation()), 200

@app.route('/api/disruption-forecast', methods=['GET'])
def get_disruptions():
    if df is None:
        return jsonify({'error': 'Data not loaded'}), 500
    return jsonify({'actions': predictive.get_disruption_actions()}), 200

@app.route('/api/summary', methods=['GET'])
def get_summary():
    if df is None:
        return jsonify({'error': 'Data not loaded'}), 500
    return jsonify({
        'total_deliveries': len(df),
        'inefficiencies': analyzer.find_inefficient_routes(),
        'best_delivery_method': optimizer.get_recommendation()
    }), 200

if __name__ == '__main__':
    print("\n🚀 GreenRoute API running on http://localhost:5000\n")
    app.run(debug=True, port=5000)