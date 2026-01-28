"""
Data preprocessing for GreenRoute Analytics
Cleans and engineers features from raw delivery data
"""

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from datetime import datetime

class DataPreprocessor:
    """Clean and engineer features from raw delivery data"""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.original_count = 0
        self.processing_log = {}
    
    def load_data(self):
        """Load raw CSV data"""
        print("📥 Loading data...")
        try:
            self.df = pd.read_csv(self.filepath)
            self.original_count = len(self.df)
            print(f"   ✅ Loaded {len(self.df):,} deliveries")
            print(f"   Columns: {', '.join(self.df.columns)}")
            return self.df
        except Exception as e:
            print(f"   ❌ Error loading data: {e}")
            return None
    
    def inspect_data(self):
        """Show data structure"""
        print("\n📋 Data Structure:")
        print(self.df.info())
        print(f"\n📊 First few rows:")
        print(self.df.head())
        return self.df
    
    def clean_data(self):
        """Remove duplicates and handle missing values"""
        print("\n🧹 Cleaning data...")
        
        # Remove duplicates
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates()
        removed_dupes = initial_count - len(self.df)
        if removed_dupes > 0:
            print(f"   Removed {removed_dupes} duplicates")
            self.processing_log['duplicates_removed'] = removed_dupes
        
        # Handle missing values
        missing_cols = self.df.isnull().sum()
        if missing_cols.sum() > 0:
            print(f"   Found {missing_cols.sum()} missing values")
            
            # For numeric columns: fill with median
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if self.df[col].isnull().sum() > 0:
                    median = self.df[col].median()
                    self.df[col].fillna(median, inplace=True)
                    print(f"   Filled {col} with median: {median}")
            
            # For categorical columns: fill with mode
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if self.df[col].isnull().sum() > 0:
                    mode = self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else "Unknown"
                    self.df[col].fillna(mode, inplace=True)
                    print(f"   Filled {col} with mode: {mode}")
        
        print(f"   ✅ Cleaned, {len(self.df):,} deliveries remain")
        self.processing_log['records_after_cleaning'] = len(self.df)
        return self.df
    
    def engineer_core_metrics(self):
        """Create metrics that show inefficiencies"""
        print("\n⚙️  Engineering core metrics...")
        
        try:
            # 1. COST PER KM (proxy for fuel efficiency)
            # Assume delivery_cost is in rupees, distance in km
            self.df['cost_per_km'] = self.df['Delivery_Cost'] / (self.df['Distance'] + 0.01)
            
            # 2. CO2 EMISSIONS (2% of delivery cost = CO2 equivalent)
            self.df['co2_emissions_kg'] = self.df['Delivery_Cost'] * 0.02
            
            # 3. DELIVERY DELAY (actual vs expected time)
            self.df['delay_hours'] = self.df['Delivery_Time_Hours'] - self.df['Expected_Time_Hours']
            self.df['is_delayed'] = (self.df['delay_hours'] > 0).astype(int)
            
            # 4. INEFFICIENCY FLAG (top 25% by cost/km)
            cost_per_km_threshold = self.df['cost_per_km'].quantile(0.75)
            self.df['is_inefficient_route'] = (self.df['cost_per_km'] > cost_per_km_threshold).astype(int)
            
            # 5. EFFICIENCY SCORE (0-100, higher is better)
            cost_score = 100 * (1 - (self.df['cost_per_km'] / self.df['cost_per_km'].max()).clip(0, 1))
            time_score = 100 * (1 - (self.df['delay_hours'] / (self.df['delay_hours'].max() + 1)).clip(0, 1))
            self.df['efficiency_score'] = (cost_score * 0.6 + time_score * 0.4).round(1)
            
            print("   ✅ Created metrics:")
            print(f"      - cost_per_km")
            print(f"      - co2_emissions_kg")
            print(f"      - delay_hours")
            print(f"      - is_inefficient_route")
            print(f"      - efficiency_score")
            
            self.processing_log['metrics_engineered'] = [
                'cost_per_km', 'co2_emissions_kg', 'delay_hours', 
                'is_inefficient_route', 'efficiency_score'
            ]
            
            return self.df
        
        except KeyError as e:
            print(f"   ❌ Column not found: {e}")
            print(f"   Available columns: {', '.join(self.df.columns)}")
            return None
    
    def validate_data(self):
        """Check data quality"""
        print("\n✔️  Validating data...")
        
        try:
            assert len(self.df) > 0, "Empty dataframe!"
            assert self.df['Distance'].min() >= 0, "Negative distances found!"
            assert self.df['Delivery_Cost'].min() >= 0, "Negative costs found!"
            assert self.df['efficiency_score'].between(0, 100).all(), "Scores out of range!"
            
            print("   ✅ All validations passed")
            self.processing_log['validation_status'] = 'PASSED'
            return True
        except AssertionError as e:
            print(f"   ❌ Validation failed: {e}")
            self.processing_log['validation_status'] = 'FAILED'
            return False
    
    def get_summary_statistics(self):
        """Generate summary of inefficiencies"""
        print("\n📊 INEFFICIENCY SUMMARY:")
        
        total = len(self.df)
        inefficient = self.df['is_inefficient_route'].sum()
        delayed = self.df['is_delayed'].sum()
        
        print(f"   Total deliveries: {total:,}")
        print(f"   Inefficient routes: {inefficient:,} ({inefficient/total*100:.1f}%)")
        print(f"   Delayed deliveries: {delayed:,} ({delayed/total*100:.1f}%)")
        print(f"   Avg efficiency score: {self.df['efficiency_score'].mean():.1f}/100")
        print(f"   Total CO2 emissions: {self.df['co2_emissions_kg'].sum():,.0f} kg")
        print(f"   Total delivery cost: ₹{self.df['Delivery_Cost'].sum():,.0f}")
        
        wasted_routes = self.df[self.df['is_inefficient_route'] == 1]
        if len(wasted_routes) > 0:
            print(f"\n   💸 COST WASTE from inefficiencies:")
            print(f"      Current cost (inefficient): ₹{wasted_routes['Delivery_Cost'].sum():,.0f}")
            print(f"      Saveable (25% optimization): ₹{wasted_routes['Delivery_Cost'].sum() * 0.25:,.0f}")
            
            print(f"\n   🌍 CO2 WASTE from inefficiencies:")
            print(f"      Current CO2 (inefficient): {wasted_routes['co2_emissions_kg'].sum():,.0f} kg")
            print(f"      Reduceable (25% optimization): {wasted_routes['co2_emissions_kg'].sum() * 0.25:,.0f} kg")
        
        stats = {
            'total_deliveries': total,
            'inefficient_routes': inefficient,
            'delayed_deliveries': delayed,
            'avg_efficiency': self.df['efficiency_score'].mean(),
            'total_co2': self.df['co2_emissions_kg'].sum(),
            'total_cost': self.df['Delivery_Cost'].sum(),
            'wasted_cost': wasted_routes['Delivery_Cost'].sum() if len(wasted_routes) > 0 else 0,
            'reduceable_co2': wasted_routes['co2_emissions_kg'].sum() * 0.25 if len(wasted_routes) > 0 else 0
        }
        
        self.processing_log['summary_statistics'] = stats
        return stats
    
    def save_processed_data(self, output_path='data/processed/cleaned_data.csv'):
        """Save cleaned and engineered data in CSV format"""
        print(f"\n💾 Saving to {output_path}...")
        
        try:
            # Create directory if it doesn't exist
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save CSV
            self.df.to_csv(output_path, index=False)
            print(f"   ✅ Saved {len(self.df):,} deliveries to CSV")
            self.processing_log['output_csv'] = output_path
            
            return output_path
        except Exception as e:
            print(f"   ❌ Error saving CSV: {e}")
            return None
    
    def save_processed_data_excel(self, output_path='data/processed/cleaned_data.xlsx'):
        """Save cleaned and engineered data in Excel format with multiple sheets"""
        print(f"\n💾 Saving to {output_path} (Excel)...")
        
        try:
            # Create directory if it doesn't exist
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Create Excel writer object
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Sheet 1: All cleaned data
                self.df.to_excel(writer, sheet_name='All Deliveries', index=False)
                
                # Sheet 2: Inefficient routes only
                inefficient_df = self.df[self.df['is_inefficient_route'] == 1]
                if len(inefficient_df) > 0:
                    inefficient_df.to_excel(writer, sheet_name='Inefficient Routes', index=False)
                
                # Sheet 3: Delayed deliveries only
                delayed_df = self.df[self.df['is_delayed'] == 1]
                if len(delayed_df) > 0:
                    delayed_df.to_excel(writer, sheet_name='Delayed Deliveries', index=False)
                
                # Sheet 4: Summary statistics
                stats = self.processing_log.get('summary_statistics', {})
                if stats:
                    stats_df = pd.DataFrame(list(stats.items()), columns=['Metric', 'Value'])
                    stats_df.to_excel(writer, sheet_name='Summary', index=False)
            
            print(f"   ✅ Saved {len(self.df):,} deliveries to Excel with {4} sheets")
            self.processing_log['output_excel'] = output_path
            
            return output_path
        except Exception as e:
            print(f"   ❌ Error saving Excel: {e}")
            return None
    
    def save_processing_report(self, output_path='data/processed/processing_report.json'):
        """Save detailed processing log as JSON"""
        print(f"\n📄 Saving processing report to {output_path}...")
        
        try:
            # Create directory if it doesn't exist
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Add timestamp
            self.processing_log['timestamp'] = datetime.now().isoformat()
            self.processing_log['original_records'] = self.original_count
            self.processing_log['final_records'] = len(self.df)
            
            # Save as JSON
            with open(output_path, 'w') as f:
                json.dump(self.processing_log, f, indent=2, default=str)
            
            print(f"   ✅ Processing report saved")
            
            return output_path
        except Exception as e:
            print(f"   ❌ Error saving report: {e}")
            return None
    
    def save_all_outputs(self, output_dir='data/processed'):
        """Save all output formats in one step"""
        print(f"\n🚀 Saving all outputs to {output_dir}/...")
        
        csv_path = self.save_processed_data(f'{output_dir}/cleaned_data.csv')
        excel_path = self.save_processed_data_excel(f'{output_dir}/cleaned_data.xlsx')
        report_path = self.save_processing_report(f'{output_dir}/processing_report.json')
        
        print(f"\n✅ All outputs saved successfully!")
        print(f"   CSV: {csv_path}")
        print(f"   Excel: {excel_path}")
        print(f"   Report: {report_path}")
        
        return {
            'csv': csv_path,
            'excel': excel_path,
            'report': report_path
        }


# USAGE
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = DataPreprocessor('data/raw/Delivery_data.csv')
    
    # Run pipeline
    if preprocessor.load_data() is not None:
        preprocessor.inspect_data()
        preprocessor.clean_data()
        if preprocessor.engineer_core_metrics() is not None:
            if preprocessor.validate_data():
                stats = preprocessor.get_summary_statistics()
                
                # Save all output formats
                preprocessor.save_all_outputs('data/processed')
                
                print("\n✅ Data preprocessing complete!")