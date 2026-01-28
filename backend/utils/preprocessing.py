"""
Data preprocessing for GreenRoute Analytics
Cleans and engineers features from raw delivery data
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

class DataPreprocessor:
    """Clean and engineer features from raw delivery data"""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.original_count = 0
    
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
        
        # Convert time columns: extract nanoseconds as hours
        # Timestamps are in format '1970-01-01 00:00:00.XXXXXXXXX' where X is nanoseconds
        def extract_hours_from_timestamp(ts_str):
            if pd.isna(ts_str):
                return np.nan
            try:
                # Extract nanoseconds part (after the last period)
                nanos = int(str(ts_str).split('.')[-1])
                # Convert nanoseconds to hours (there are 3.6e12 nanoseconds in an hour)
                return nanos / 3.6e12
            except:
                return np.nan
        
        self.df['delivery_time_hours'] = self.df['delivery_time_hours'].apply(extract_hours_from_timestamp)
        self.df['expected_time_hours'] = self.df['expected_time_hours'].apply(extract_hours_from_timestamp)
        
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
        return self.df
    
    def engineer_core_metrics(self):
        """Create metrics that show inefficiencies"""
        print("\n⚙️  Engineering core metrics...")
        
        try:
            # 1. COST PER KM (proxy for fuel efficiency)
            # Assume delivery_cost is in rupees, distance in km
            self.df['cost_per_km'] = self.df['delivery_cost'] / (self.df['distance_km'] + 0.01)
            
            # 2. CO2 EMISSIONS (2% of delivery cost = CO2 equivalent)
            self.df['co2_emissions_kg'] = self.df['delivery_cost'] * 0.02
            
            # 3. DELIVERY DELAY (actual vs expected time)
            self.df['delay_hours'] = self.df['delivery_time_hours'] - self.df['expected_time_hours']
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
            assert self.df['distance_km'].min() >= 0, "Negative distances found!"
            assert self.df['delivery_cost'].min() >= 0, "Negative costs found!"
            assert self.df['efficiency_score'].between(0, 100).all(), "Scores out of range!"
            
            print("   ✅ All validations passed")
            return True
        except AssertionError as e:
            print(f"   ❌ Validation failed: {e}")
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
        print(f"   Total delivery cost: ₹{self.df['delivery_cost'].sum():,.0f}")
        
        wasted_routes = self.df[self.df['is_inefficient_route'] == 1]
        if len(wasted_routes) > 0:
            print(f"\n   💸 COST WASTE from inefficiencies:")
            print(f"      Current cost (inefficient): ₹{wasted_routes['delivery_cost'].sum():,.0f}")
            print(f"      Saveable (25% optimization): ₹{wasted_routes['delivery_cost'].sum() * 0.25:,.0f}")
            
            print(f"\n   🌍 CO2 WASTE from inefficiencies:")
            print(f"      Current CO2 (inefficient): {wasted_routes['co2_emissions_kg'].sum():,.0f} kg")
            print(f"      Reduceable (25% optimization): {wasted_routes['co2_emissions_kg'].sum() * 0.25:,.0f} kg")
        
        return {
            'total_deliveries': total,
            'inefficient_routes': inefficient,
            'delayed_deliveries': delayed,
            'avg_efficiency': self.df['efficiency_score'].mean(),
            'total_co2': self.df['co2_emissions_kg'].sum(),
            'total_cost': self.df['delivery_cost'].sum(),
            'wasted_cost': wasted_routes['delivery_cost'].sum() if len(wasted_routes) > 0 else 0,
            'reduceable_co2': wasted_routes['co2_emissions_kg'].sum() * 0.25 if len(wasted_routes) > 0 else 0
        }
    
    def save_processed_data(self, output_path='data/processed/cleaned_data.csv'):
        """Save cleaned and engineered data"""
        print(f"\n💾 Saving to {output_path}...")
        
        # Create directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.df.to_csv(output_path, index=False)
        print(f"   ✅ Saved {len(self.df):,} deliveries")
        return output_path


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
                preprocessor.save_processed_data()
                print("\n✅ Data preprocessing complete!")