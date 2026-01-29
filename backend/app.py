"""
Flask backend for GreenRoute Analytics
Provides REST API endpoints for data preprocessing and analysis
"""

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd
import os
import json
from datetime import datetime
from pathlib import Path

# Import our data processing classes
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
from data_cleaner import DataCleaner
from utils.preprocessing import DataPreprocessor

app = Flask(__name__)
CORS(app)

# Configuration
RAW_DATA_PATH = 'data/raw/Delivery_data.csv'
CLEANED_DATA_PATH = 'data/raw/Delivery_data_cleaned.csv'
PROCESSED_DATA_PATH = 'data/processed'

# Global state
processing_status = {
    'status': 'idle',
    'progress': 0,
    'message': '',
    'timestamp': None
}


# ============================================================================
# HEALTH & STATUS ENDPOINTS
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'GreenRoute Analytics Backend',
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current processing status"""
    return jsonify(processing_status), 200


# ============================================================================
# DATA CLEANING ENDPOINTS
# ============================================================================

@app.route('/api/clean-data', methods=['POST'])
def clean_data():
    """
    Clean raw delivery data by:
    - Fixing corrupted delivery partner column
    - Validating numeric columns
    - Removing outliers
    """
    global processing_status
    
    try:
        processing_status['status'] = 'cleaning'
        processing_status['message'] = 'Starting data cleaning...'
        processing_status['progress'] = 0
        processing_status['timestamp'] = datetime.now().isoformat()
        
        # Check if raw data exists
        if not os.path.exists(RAW_DATA_PATH):
            return jsonify({
                'error': f'Raw data file not found: {RAW_DATA_PATH}',
                'hint': 'Please ensure Delivery_data.csv exists in data/raw/ directory'
            }), 400
        
        # Run data cleaner
        cleaner = DataCleaner(RAW_DATA_PATH)
        cleaned_df = cleaner.clean()
        
        if cleaned_df is None:
            processing_status['status'] = 'error'
            processing_status['message'] = 'Failed to clean data'
            return jsonify({'error': 'Data cleaning failed'}), 500
        
        # Save cleaned data
        output_path = cleaner.save_cleaned_data(CLEANED_DATA_PATH)
        
        processing_status['status'] = 'completed'
        processing_status['progress'] = 100
        processing_status['message'] = f'Cleaned {len(cleaned_df):,} records'
        
        return jsonify({
            'success': True,
            'message': f'Successfully cleaned {len(cleaned_df):,} delivery records',
            'rows_processed': len(cleaned_df),
            'output_file': output_path,
            'report': cleaner.get_report()
        }), 200
    
    except Exception as e:
        processing_status['status'] = 'error'
        processing_status['message'] = str(e)
        return jsonify({'error': str(e)}), 500


# ============================================================================
# DATA PREPROCESSING ENDPOINTS
# ============================================================================

@app.route('/api/preprocess-data', methods=['POST'])
def preprocess_data():
    """
    Preprocess cleaned data by:
    - Engineering core metrics (cost_per_km, efficiency_score, etc.)
    - Validating data quality
    - Generating summary statistics
    """
    global processing_status
    
    try:
        processing_status['status'] = 'preprocessing'
        processing_status['message'] = 'Starting data preprocessing...'
        processing_status['progress'] = 0
        processing_status['timestamp'] = datetime.now().isoformat()
        
        # Use cleaned data if it exists, otherwise use raw data
        input_file = CLEANED_DATA_PATH if os.path.exists(CLEANED_DATA_PATH) else RAW_DATA_PATH
        
        if not os.path.exists(input_file):
            return jsonify({
                'error': 'Data file not found',
                'hint': 'Run /api/clean-data endpoint first'
            }), 400
        
        # Run preprocessor
        preprocessor = DataPreprocessor(input_file)
        
        # Run pipeline
        if preprocessor.load_data() is None:
            return jsonify({'error': 'Failed to load data'}), 500
        
        processing_status['progress'] = 25
        processing_status['message'] = 'Data loaded, cleaning...'
        
        preprocessor.clean_data()
        processing_status['progress'] = 50
        processing_status['message'] = 'Cleaning complete, engineering metrics...'
        
        if preprocessor.engineer_core_metrics() is None:
            return jsonify({'error': 'Failed to engineer metrics'}), 500
        
        processing_status['progress'] = 75
        processing_status['message'] = 'Metrics engineered, validating...'
        
        if not preprocessor.validate_data():
            return jsonify({'error': 'Data validation failed'}), 500
        
        processing_status['progress'] = 90
        processing_status['message'] = 'Validation passed, generating summary...'
        
        stats = preprocessor.get_summary_statistics()
        
        processing_status['progress'] = 95
        processing_status['message'] = 'Saving outputs...'
        
        # Save all outputs
        outputs = preprocessor.save_all_outputs(PROCESSED_DATA_PATH)
        
        processing_status['status'] = 'completed'
        processing_status['progress'] = 100
        processing_status['message'] = 'Preprocessing complete'
        
        return jsonify({
            'success': True,
            'message': f'Successfully preprocessed {len(preprocessor.df):,} deliveries',
            'records_processed': len(preprocessor.df),
            'summary_statistics': {
                'total_deliveries': stats['total_deliveries'],
                'inefficient_routes': stats['inefficient_routes'],
                'delayed_deliveries': stats['delayed_deliveries'],
                'avg_efficiency_score': round(stats['avg_efficiency'], 1),
                'total_co2_emissions_kg': round(stats['total_co2'], 0),
                'total_delivery_cost_inr': round(stats['total_cost'], 0),
                'potential_savings_inr': round(stats['wasted_cost'] * 0.25, 0),
                'reduceable_co2_kg': round(stats['reduceable_co2'], 0)
            },
            'output_files': outputs
        }), 200
    
    except Exception as e:
        processing_status['status'] = 'error'
        processing_status['message'] = str(e)
        return jsonify({'error': str(e)}), 500


# ============================================================================
# ANALYSIS ENDPOINTS
# ============================================================================

@app.route('/api/inefficient-routes', methods=['GET'])
def get_inefficient_routes():
    """Get list of inefficient routes"""
    try:
        processed_file = os.path.join(PROCESSED_DATA_PATH, 'cleaned_data.csv')
        
        if not os.path.exists(processed_file):
            return jsonify({
                'error': 'Processed data not found',
                'hint': 'Run /api/preprocess-data endpoint first'
            }), 400
        
        df = pd.read_csv(processed_file)
        
        # Filter inefficient routes
        inefficient = df[df['is_inefficient_route'] == 1].copy()
        inefficient = inefficient.sort_values('cost_per_km', ascending=False)
        
        return jsonify({
            'total_inefficient': len(inefficient),
            'percentage': round(len(inefficient) / len(df) * 100, 1),
            'top_10_routes': inefficient[
                ['Delivery_Partner', 'Distance', 'Delivery_Cost', 'cost_per_km', 'efficiency_score']
            ].head(10).to_dict('records')
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/delayed-deliveries', methods=['GET'])
def get_delayed_deliveries():
    """Get list of delayed deliveries"""
    try:
        processed_file = os.path.join(PROCESSED_DATA_PATH, 'cleaned_data.csv')
        
        if not os.path.exists(processed_file):
            return jsonify({
                'error': 'Processed data not found',
                'hint': 'Run /api/preprocess-data endpoint first'
            }), 400
        
        df = pd.read_csv(processed_file)
        
        # Filter delayed deliveries
        delayed = df[df['is_delayed'] == 1].copy()
        delayed = delayed.sort_values('delay_hours', ascending=False)
        
        return jsonify({
            'total_delayed': len(delayed),
            'percentage': round(len(delayed) / len(df) * 100, 1),
            'top_10_delayed': delayed[
                ['Delivery_Partner', 'Delivery_Time_Hours', 'Expected_Time_Hours', 'delay_hours']
            ].head(10).to_dict('records')
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/partner-performance', methods=['GET'])
def get_partner_performance():
    """Get performance metrics by delivery partner"""
    try:
        processed_file = os.path.join(PROCESSED_DATA_PATH, 'cleaned_data.csv')
        
        if not os.path.exists(processed_file):
            return jsonify({
                'error': 'Processed data not found',
                'hint': 'Run /api/preprocess-data endpoint first'
            }), 400
        
        df = pd.read_csv(processed_file)
        
        # Group by partner
        partner_stats = []
        for partner in df['Delivery_Partner'].unique():
            partner_data = df[df['Delivery_Partner'] == partner]
            
            partner_stats.append({
                'partner': partner,
                'total_deliveries': len(partner_data),
                'avg_efficiency_score': round(partner_data['efficiency_score'].mean(), 1),
                'inefficient_rate': round(partner_data['is_inefficient_route'].mean() * 100, 1),
                'delayed_rate': round(partner_data['is_delayed'].mean() * 100, 1),
                'avg_cost_per_km': round(partner_data['cost_per_km'].mean(), 2),
                'avg_co2_emissions': round(partner_data['co2_emissions_kg'].mean(), 2)
            })
        
        # Sort by efficiency
        partner_stats.sort(key=lambda x: x['avg_efficiency_score'], reverse=True)
        
        return jsonify({
            'partners': partner_stats
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# FILE DOWNLOAD ENDPOINTS
# ============================================================================

@app.route('/api/download/<file_type>', methods=['GET'])
def download_file(file_type):
    """Download processed data files"""
    try:
        file_mapping = {
            'csv': os.path.join(PROCESSED_DATA_PATH, 'cleaned_data.csv'),
            'excel': os.path.join(PROCESSED_DATA_PATH, 'cleaned_data.xlsx'),
            'report': os.path.join(PROCESSED_DATA_PATH, 'processing_report.json')
        }
        
        if file_type not in file_mapping:
            return jsonify({'error': 'Invalid file type'}), 400
        
        file_path = file_mapping[file_type]
        
        if not os.path.exists(file_path):
            return jsonify({'error': f'File not found: {file_path}'}), 404
        
        return send_file(file_path, as_attachment=True)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# FULL PIPELINE ENDPOINT
# ============================================================================

@app.route('/api/process-complete', methods=['POST'])
def process_complete():
    """
    Run the complete pipeline:
    1. Clean data
    2. Preprocess data
    3. Generate reports
    """
    global processing_status
    
    try:
        processing_status['status'] = 'processing'
        processing_status['message'] = 'Starting complete pipeline...'
        processing_status['timestamp'] = datetime.now().isoformat()
        
        # Step 1: Clean
        print("Step 1: Cleaning data...")
        cleaner = DataCleaner(RAW_DATA_PATH)
        cleaned_df = cleaner.clean()
        
        if cleaned_df is None:
            raise Exception('Data cleaning failed')
        
        cleaner.save_cleaned_data(CLEANED_DATA_PATH)
        
        # Step 2: Preprocess
        print("Step 2: Preprocessing data...")
        preprocessor = DataPreprocessor(CLEANED_DATA_PATH)
        
        if preprocessor.load_data() is None:
            raise Exception('Failed to load cleaned data')
        
        preprocessor.clean_data()
        
        if preprocessor.engineer_core_metrics() is None:
            raise Exception('Failed to engineer metrics')
        
        if not preprocessor.validate_data():
            raise Exception('Data validation failed')
        
        stats = preprocessor.get_summary_statistics()
        outputs = preprocessor.save_all_outputs(PROCESSED_DATA_PATH)
        
        processing_status['status'] = 'completed'
        processing_status['progress'] = 100
        processing_status['message'] = 'Pipeline complete'
        
        return jsonify({
            'success': True,
            'message': 'Complete pipeline executed successfully',
            'records_processed': len(preprocessor.df),
            'summary_statistics': {
                'total_deliveries': stats['total_deliveries'],
                'inefficient_routes': stats['inefficient_routes'],
                'delayed_deliveries': stats['delayed_deliveries'],
                'avg_efficiency_score': round(stats['avg_efficiency'], 1),
                'total_co2_emissions_kg': round(stats['total_co2'], 0),
                'total_delivery_cost_inr': round(stats['total_cost'], 0),
                'potential_savings_inr': round(stats['wasted_cost'] * 0.25, 0),
                'reduceable_co2_kg': round(stats['reduceable_co2'], 0)
            },
            'output_files': outputs
        }), 200
    
    except Exception as e:
        processing_status['status'] = 'error'
        processing_status['message'] = str(e)
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': {
            'Health': 'GET /api/health',
            'Status': 'GET /api/status',
            'Clean Data': 'POST /api/clean-data',
            'Preprocess Data': 'POST /api/preprocess-data',
            'Inefficient Routes': 'GET /api/inefficient-routes',
            'Delayed Deliveries': 'GET /api/delayed-deliveries',
            'Partner Performance': 'GET /api/partner-performance',
            'Download': 'GET /api/download/<file_type>',
            'Complete Pipeline': 'POST /api/process-complete'
        }
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'error': 'Internal server error',
        'message': str(error)
    }), 500


# ============================================================================
# STARTUP
# ============================================================================

if __name__ == '__main__':
    # Create necessary directories
    Path('data/raw').mkdir(parents=True, exist_ok=True)
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("🚀 GreenRoute Analytics Backend")
    print("="*70)
    print("\n📡 Available API Endpoints:")
    print("   • GET  /api/health                    - Health check")
    print("   • GET  /api/status                    - Processing status")
    print("   • POST /api/clean-data                - Clean raw data")
    print("   • POST /api/preprocess-data           - Preprocess cleaned data")
    print("   • GET  /api/inefficient-routes        - Get inefficient routes")
    print("   • GET  /api/delayed-deliveries        - Get delayed deliveries")
    print("   • GET  /api/partner-performance       - Get partner statistics")
    print("   • GET  /api/download/<file_type>      - Download processed data")
    print("   • POST /api/process-complete          - Run complete pipeline")
    print("\n📊 Start with: POST /api/process-complete")
    print("="*70 + "\n")
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)