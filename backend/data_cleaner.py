"""
Data cleaner for GreenRoute Analytics - fixes corrupted data columns
Handles the concatenated delivery partner issue and other data problems
"""

import pandas as pd
import numpy as np
from pathlib import Path

class DataCleaner:
    """Fix corrupted and malformed data before preprocessing"""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.issues_found = []
    
    def load_raw_data(self):
        """Load raw CSV data"""
        print("📥 Loading raw data...")
        try:
            self.df = pd.read_csv(self.filepath)
            print(f"   ✅ Loaded {len(self.df):,} rows")
            print(f"   Columns: {', '.join(self.df.columns)}")
            return self.df
        except Exception as e:
            print(f"   ❌ Error loading data: {e}")
            return None
    
    def detect_concatenated_strings(self):
        """Detect columns with concatenated string values (data corruption)"""
        print("\n🔍 Detecting concatenated string columns...")
        
        for col in self.df.columns:
            if self.df[col].dtype == 'object':  # String column
                # Check if values are unusually long (concatenated data)
                avg_length = self.df[col].astype(str).str.len().mean()
                max_length = self.df[col].astype(str).str.len().max()
                
                if avg_length > 200 or max_length > 1000:
                    self.issues_found.append({
                        'column': col,
                        'issue': 'Likely concatenated/corrupted data',
                        'avg_length': round(avg_length, 1),
                        'max_length': max_length
                    })
                    print(f"   ⚠️  Column '{col}': avg length {avg_length:.0f}, max {max_length}")
        
        return self.issues_found
    
    def fix_delivery_partner(self):
        """Extract valid delivery partner from corrupted concatenated string"""
        print("\n🔧 Fixing corrupted Delivery_Partner column...")
        
        if 'Delivery_Partner' not in self.df.columns:
            print("   ℹ️  Delivery_Partner column not found, skipping fix")
            return self.df
        
        # List of known delivery partners
        known_partners = [
            'delhivery', 'xpressbees', 'shadow fax', 'dhl', 'ecom express',
            'blue dart', 'fedex', 'ekart', 'amazon logistics'
        ]
        
        def extract_partner(text):
            """Extract first valid partner name from corrupted string"""
            text_lower = str(text).lower()
            
            for partner in known_partners:
                if partner in text_lower:
                    return partner.title()
            
            return 'Other'
        
        # Apply fix
        self.df['Delivery_Partner'] = self.df['Delivery_Partner'].apply(extract_partner)
        
        print("   ✅ Fixed Delivery_Partner column")
        print(f"   Partner distribution:\n{self.df['Delivery_Partner'].value_counts()}")
        
        return self.df
    
    def validate_numeric_columns(self):
        """Ensure numeric columns contain valid numbers"""
        print("\n✔️  Validating numeric columns...")
        
        numeric_cols = ['Distance', 'Delivery_Cost', 'Delivery_Time_Hours', 'Expected_Time_Hours']
        
        for col in numeric_cols:
            if col not in self.df.columns:
                continue
            
            # Convert to numeric, coercing errors
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
            # Check for NaN values created by coercion
            nan_count = self.df[col].isna().sum()
            if nan_count > 0:
                print(f"   ⚠️  Column '{col}': {nan_count} invalid values found")
                self.issues_found.append({
                    'column': col,
                    'issue': f'{nan_count} non-numeric values',
                    'action': 'Filled with median'
                })
                
                # Fill with median
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
            else:
                print(f"   ✅ Column '{col}': All values valid")
        
        return self.df
    
    def remove_outliers(self):
        """Remove extreme outliers that indicate data corruption"""
        print("\n📊 Removing outliers...")
        
        initial_count = len(self.df)
        
        # Remove rows with distance > 500 km (likely data corruption)
        if 'Distance' in self.df.columns:
            self.df = self.df[self.df['Distance'] <= 500]
            removed = initial_count - len(self.df)
            if removed > 0:
                print(f"   ✅ Removed {removed} rows with distance > 500 km")
        
        # Remove rows with delivery cost > 50000 rupees (unlikely)
        if 'Delivery_Cost' in self.df.columns:
            initial_count = len(self.df)
            self.df = self.df[self.df['Delivery_Cost'] <= 50000]
            removed = initial_count - len(self.df)
            if removed > 0:
                print(f"   ✅ Removed {removed} rows with cost > ₹50,000")
        
        return self.df
    
    def clean(self):
        """Run full cleaning pipeline"""
        print("\n" + "="*60)
        print("🧹 STARTING DATA CLEANING PIPELINE")
        print("="*60)
        
        if self.load_raw_data() is None:
            return None
        
        self.detect_concatenated_strings()
        self.fix_delivery_partner()
        self.validate_numeric_columns()
        self.remove_outliers()
        
        print("\n" + "="*60)
        print(f"✅ CLEANING COMPLETE: {len(self.df):,} valid rows")
        print("="*60)
        
        return self.df
    
    def save_cleaned_data(self, output_path='data/raw/Delivery_data_cleaned.csv'):
        """Save cleaned data to file"""
        print(f"\n💾 Saving cleaned data to {output_path}...")
        
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            self.df.to_csv(output_path, index=False)
            print(f"   ✅ Saved {len(self.df):,} rows")
            return output_path
        except Exception as e:
            print(f"   ❌ Error saving: {e}")
            return None
    
    def get_report(self):
        """Get cleaning report"""
        return {
            'rows_processed': len(self.df),
            'issues_found': self.issues_found
        }


# USAGE
if __name__ == "__main__":
    # Clean the raw data
    cleaner = DataCleaner('data/raw/Delivery_data.csv')
    cleaned_df = cleaner.clean()
    
    if cleaned_df is not None:
        output_file = cleaner.save_cleaned_data()
        report = cleaner.get_report()
        
        print("\n📋 CLEANING REPORT:")
        print(f"Rows processed: {report['rows_processed']:,}")
        if report['issues_found']:
            print("\nIssues found:")
            for issue in report['issues_found']:
                print(f"  - {issue}")