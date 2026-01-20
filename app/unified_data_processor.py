"""
Unified Data Processor for Multi-Country Real Estate Data
==========================================================

This script loads and standardizes real estate datasets from multiple countries
into a unified format for machine learning model training.

Features:
- Load datasets from Bangladesh, India, Japan, Poland, USA
- Standardize column names and data types
- Convert currencies to USD
- Generate synthetic data for missing cities
- Handle missing values
"""

import pandas as pd
import numpy as np
import os
import glob
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Currency conversion rates to USD (approximate)
CURRENCY_RATES = {
    'BDT': 110,   # Bangladesh Taka
    'INR': 83,    # Indian Rupee
    'PLN': 4,     # Polish Zloty
    'JPY': 150,   # Japanese Yen
    'USD': 1,     # US Dollar
}

# Top 20 cities per country for synthetic data generation
TOP_CITIES = {
    'Bangladesh': ['Dhaka', 'Chittagong', 'Khulna', 'Rajshahi', 'Sylhet', 
                   'Rangpur', 'Comilla', 'Gazipur', 'Narayanganj', 'Mymensingh',
                   'Barisal', 'Bogra', 'Cox\'s Bazar', 'Jessore', 'Dinajpur',
                   'Brahmanbaria', 'Savar', 'Tongi', 'Narsingdi', 'Tangail'],
    'India': ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 
              'Kolkata', 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow',
              'Surat', 'Kanpur', 'Nagpur', 'Indore', 'Thane',
              'Bhopal', 'Visakhapatnam', 'Patna', 'Vadodara', 'Ghaziabad'],
    'Japan': ['Tokyo', 'Osaka', 'Yokohama', 'Nagoya', 'Sapporo',
              'Kobe', 'Kyoto', 'Fukuoka', 'Kawasaki', 'Saitama',
              'Hiroshima', 'Sendai', 'Chiba', 'Kitakyushu', 'Sakai',
              'Niigata', 'Hamamatsu', 'Kumamoto', 'Sagamihara', 'Okayama'],
    'Poland': ['Warsaw', 'Krakow', 'Lodz', 'Wroclaw', 'Poznan',
               'Gdansk', 'Szczecin', 'Bydgoszcz', 'Lublin', 'Katowice',
               'Bialystok', 'Gdynia', 'Czestochowa', 'Radom', 'Sosnowiec',
               'Torun', 'Kielce', 'Rzeszow', 'Gliwice', 'Zabrze'],
    'USA': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
            'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose',
            'Austin', 'Jacksonville', 'Fort Worth', 'Columbus', 'Charlotte',
            'Seattle', 'Denver', 'Washington', 'Boston', 'Nashville']
}

class UnifiedDataProcessor:
    """Process and unify real estate data from multiple countries."""
    
    def __init__(self, data_dir: str = 'datasets'):
        """Initialize the data processor.
        
        Args:
            data_dir: Directory containing country-specific dataset folders
        """
        self.data_dir = data_dir
        self.unified_df = None
        self.country_stats = {}
        
        # Define standard column names
        self.standard_columns = [
            'country', 'city', 'rooms', 'area_sqm', 'balcony',
            'building_age', 'furnishing_status', 'price_usd'
        ]
    
    def load_bangladesh_data(self) -> pd.DataFrame:
        """Load and standardize Bangladesh dataset."""
        filepath = os.path.join(self.data_dir, 'Bangladesh', 'house_price_bd.csv')
        if not os.path.exists(filepath):
            print(f"Warning: Bangladesh data not found at {filepath}")
            return pd.DataFrame()
        
        df = pd.read_csv(filepath)
        print(f"Loaded Bangladesh data: {len(df)} records")
        
        # Standardize columns - using correct column names
        standardized = pd.DataFrame({
            'country': 'Bangladesh',
            'city': df['City'],
            'rooms': pd.to_numeric(df['Bedrooms'], errors='coerce'),
            'area_sqm': pd.to_numeric(df['Floor_area'], errors='coerce') * 0.0929,  # sqft to sqm
            'balcony': 0,  # Not available in dataset, default to 0
            'building_age': 5,  # Not available, estimate median
            'furnishing_status': 'Unknown',
            'price_usd': pd.to_numeric(df['Price_in_taka'].astype(str).str.replace(',', '').str.replace('৳', ''), errors='coerce') / CURRENCY_RATES['BDT']
        })
        
        return standardized.dropna(subset=['price_usd', 'rooms', 'area_sqm'])
    
    def load_india_data(self) -> pd.DataFrame:
        """Load and standardize India datasets."""
        india_dir = os.path.join(self.data_dir, 'INDIA')
        all_data = []
        
        # Load Bangalore dataset
        bangalore_path = os.path.join(india_dir, 'Bangalore.csv')
        if os.path.exists(bangalore_path):
            df = pd.read_csv(bangalore_path)
            print(f"Loaded India Bangalore: {len(df)} records")
            
            # Handle balcony column
            if 'Balcony' in df.columns:
                balcony = pd.to_numeric(df['Balcony'], errors='coerce').fillna(0).apply(lambda x: 1 if x > 0 else 0)
            else:
                balcony = 0
            
            standardized = pd.DataFrame({
                'country': 'India',
                'city': 'Bangalore',
                'rooms': pd.to_numeric(df['No. of Bedrooms'], errors='coerce'),
                'area_sqm': pd.to_numeric(df['Area'], errors='coerce') * 0.0929,
                'balcony': balcony,
                'building_age': 10,  # Estimate
                'furnishing_status': df.get('Furnishing', 'Unknown') if 'Furnishing' in df.columns else 'Unknown',
                'price_usd': pd.to_numeric(df['Price'], errors='coerce') / CURRENCY_RATES['INR']
            })
            all_data.append(standardized)
        
        # Load Delhi dataset
        delhi_path = os.path.join(india_dir, 'Delhi_v2.csv')
        if os.path.exists(delhi_path):
            df = pd.read_csv(delhi_path, nrows=50000)  # Limit for memory
            print(f"Loaded India Delhi: {len(df)} records")
            
            standardized = pd.DataFrame({
                'country': 'India',
                'city': 'Delhi',
                'rooms': pd.to_numeric(df['Bedrooms'], errors='coerce'),
                'area_sqm': pd.to_numeric(df['area'], errors='coerce') * 0.0929,
                'balcony': pd.to_numeric(df['Balcony'], errors='coerce').fillna(0).apply(lambda x: 1 if x > 0 else 0) if 'Balcony' in df.columns else 0,
                'building_age': 10,
                'furnishing_status': df.get('Furnished_status', 'Unknown'),
                'price_usd': pd.to_numeric(df['price'], errors='coerce') / CURRENCY_RATES['INR']
            })
            all_data.append(standardized)
        
        # Load House Rent dataset (for city and furnishing diversity)
        rent_path = os.path.join(india_dir, 'House_Rent_Dataset.csv')
        if os.path.exists(rent_path):
            df = pd.read_csv(rent_path)
            print(f"Loaded India House Rent: {len(df)} records")
            
            # Convert rent to approximate sale price (multiply by 200 as rough estimate)
            standardized = pd.DataFrame({
                'country': 'India',
                'city': df['City'],
                'rooms': pd.to_numeric(df['BHK'], errors='coerce'),
                'area_sqm': pd.to_numeric(df['Size'], errors='coerce') * 0.0929,
                'balcony': 0,
                'building_age': 10,
                'furnishing_status': df['Furnishing Status'],
                'price_usd': pd.to_numeric(df['Rent'], errors='coerce') * 200 / CURRENCY_RATES['INR']
            })
            all_data.append(standardized)
        
        # Load House Price India dataset
        price_path = os.path.join(india_dir, 'House Price India.csv')
        if os.path.exists(price_path):
            df = pd.read_csv(price_path)
            print(f"Loaded India House Price: {len(df)} records")
            
            standardized = pd.DataFrame({
                'country': 'India',
                'city': 'India',  # Generic city
                'rooms': pd.to_numeric(df.get('number of bedrooms', 2), errors='coerce'),
                'area_sqm': pd.to_numeric(df.get('living area', 1000), errors='coerce') * 0.0929,
                'balcony': 0,
                'building_age': 2024 - pd.to_numeric(df.get('Built Year', 2015), errors='coerce'),
                'furnishing_status': 'Unknown',
                'price_usd': pd.to_numeric(df['Price'], errors='coerce') / CURRENCY_RATES['INR']
            })
            all_data.append(standardized)
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            return result.dropna(subset=['price_usd', 'rooms', 'area_sqm'])
        return pd.DataFrame()
    
    def load_poland_data(self) -> pd.DataFrame:
        """Load and standardize Poland datasets."""
        poland_dir = os.path.join(self.data_dir, 'POLAND')
        all_data = []
        
        # Load all monthly apartment files (limit to a few for efficiency)
        files = glob.glob(os.path.join(poland_dir, 'apartments_pl_*.csv'))[:6]
        
        for filepath in files:
            df = pd.read_csv(filepath)
            print(f"Loaded Poland {os.path.basename(filepath)}: {len(df)} records")
            
            standardized = pd.DataFrame({
                'country': 'Poland',
                'city': df['city'],
                'rooms': pd.to_numeric(df['rooms'], errors='coerce'),
                'area_sqm': pd.to_numeric(df['squareMeters'], errors='coerce'),
                'balcony': df['hasParkingSpace'].apply(lambda x: 1 if x else 0) if 'hasParkingSpace' in df.columns else 0,
                'building_age': 2024 - pd.to_numeric(df.get('buildYear', 2000), errors='coerce'),
                'furnishing_status': df.get('condition', 'Unknown') if 'condition' in df.columns else 'Unknown',
                'price_usd': pd.to_numeric(df['price'], errors='coerce') / CURRENCY_RATES['PLN']
            })
            all_data.append(standardized)
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            return result.dropna(subset=['price_usd', 'rooms', 'area_sqm'])
        return pd.DataFrame()
    
    def load_usa_data(self) -> pd.DataFrame:
        """Load and standardize USA datasets."""
        usa_dir = os.path.join(self.data_dir, 'USA')
        all_data = []
        
        # Load KC House data
        kc_path = os.path.join(usa_dir, 'kc_house_data.csv')
        if os.path.exists(kc_path):
            df = pd.read_csv(kc_path)
            print(f"Loaded USA KC House: {len(df)} records")
            
            standardized = pd.DataFrame({
                'country': 'USA',
                'city': 'Seattle',  # King County, Washington
                'rooms': pd.to_numeric(df['bedrooms'], errors='coerce'),
                'area_sqm': pd.to_numeric(df['sqft_living'], errors='coerce') * 0.0929,
                'balcony': df['view'].apply(lambda x: 1 if x > 0 else 0) if 'view' in df.columns else 0,
                'building_age': 2024 - pd.to_numeric(df.get('yr_built', 1990), errors='coerce'),
                'furnishing_status': 'Unknown',
                'price_usd': pd.to_numeric(df['price'], errors='coerce')
            })
            all_data.append(standardized)
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            return result.dropna(subset=['price_usd', 'rooms', 'area_sqm'])
        return pd.DataFrame()
    
    def load_japan_data(self) -> pd.DataFrame:
        """Load and standardize Japan datasets."""
        japan_dir = os.path.join(self.data_dir, 'JAPAN')
        all_data = []
        
        # Load first 5 prefecture files only (to limit memory usage)
        files = sorted(glob.glob(os.path.join(japan_dir, '*.csv')))[:5]
        
        prefecture_names = ['Hokkaido', 'Aomori', 'Iwate', 'Miyagi', 'Akita', 
                           'Yamagata', 'Fukushima', 'Ibaraki', 'Tochigi', 'Gunma']
        
        for i, filepath in enumerate(files):
            df = pd.read_csv(filepath, nrows=10000)  # Limit rows per file
            prefecture = prefecture_names[i] if i < len(prefecture_names) else f'Prefecture_{i}'
            print(f"Loaded Japan {prefecture}: {len(df)} records")
            
            # Get city column
            if 'Municipality' in df.columns:
                city = df['Municipality']
            else:
                city = prefecture
            
            # Get area column
            if 'Area' in df.columns:
                area_sqm = pd.to_numeric(df['Area'], errors='coerce')
            else:
                area_sqm = 70  # Default
            
            # Get building age from BuildingYear
            if 'BuildingYear' in df.columns:
                building_age = 2024 - pd.to_numeric(df['BuildingYear'], errors='coerce')
                building_age = building_age.fillna(10)
            else:
                building_age = 10
            
            # Get price from TradePrice
            if 'TradePrice' in df.columns:
                price_usd = pd.to_numeric(df['TradePrice'], errors='coerce') / CURRENCY_RATES['JPY']
            else:
                price_usd = np.nan
            
            standardized = pd.DataFrame({
                'country': 'Japan',
                'city': city,
                'rooms': 3,  # Default for Japan apartments
                'area_sqm': area_sqm,
                'balcony': 1,  # Most Japanese apartments have balconies
                'building_age': building_age,
                'furnishing_status': 'Unfurnished',  # Standard in Japan
                'price_usd': price_usd
            })
            all_data.append(standardized)
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            return result.dropna(subset=['price_usd', 'area_sqm'])
        return pd.DataFrame()
    
    def generate_synthetic_data(self, country: str, existing_cities: List[str], 
                                stats: Dict) -> pd.DataFrame:
        """Generate synthetic data for missing cities based on existing statistics.
        
        Args:
            country: Country name
            existing_cities: List of cities already in the dataset
            stats: Dictionary with mean/std for numerical features
            
        Returns:
            DataFrame with synthetic data for missing cities
        """
        if country not in TOP_CITIES:
            return pd.DataFrame()
        
        missing_cities = [c for c in TOP_CITIES[country] if c not in existing_cities]
        if not missing_cities:
            return pd.DataFrame()
        
        synthetic_records = []
        samples_per_city = 100  # Generate 100 samples per missing city
        
        for city in missing_cities:
            for _ in range(samples_per_city):
                record = {
                    'country': country,
                    'city': city,
                    'rooms': max(1, int(np.random.normal(stats['rooms_mean'], stats['rooms_std']))),
                    'area_sqm': max(20, np.random.normal(stats['area_mean'], stats['area_std'])),
                    'balcony': np.random.choice([0, 1], p=[0.3, 0.7]),
                    'building_age': max(0, int(np.random.normal(stats['age_mean'], stats['age_std']))),
                    'furnishing_status': np.random.choice(['Furnished', 'Semi-Furnished', 'Unfurnished']),
                    'price_usd': max(1000, np.random.normal(stats['price_mean'], stats['price_std']))
                }
                synthetic_records.append(record)
        
        print(f"Generated {len(synthetic_records)} synthetic records for {len(missing_cities)} cities in {country}")
        return pd.DataFrame(synthetic_records)
    
    def load_all_datasets(self) -> pd.DataFrame:
        """Load and combine all country datasets."""
        print("=" * 60)
        print("Loading all datasets...")
        print("=" * 60)
        
        datasets = []
        
        # Load each country
        bangladesh_df = self.load_bangladesh_data()
        if not bangladesh_df.empty:
            datasets.append(bangladesh_df)
            self.country_stats['Bangladesh'] = self._compute_stats(bangladesh_df)
        
        india_df = self.load_india_data()
        if not india_df.empty:
            datasets.append(india_df)
            self.country_stats['India'] = self._compute_stats(india_df)
        
        poland_df = self.load_poland_data()
        if not poland_df.empty:
            datasets.append(poland_df)
            self.country_stats['Poland'] = self._compute_stats(poland_df)
        
        usa_df = self.load_usa_data()
        if not usa_df.empty:
            datasets.append(usa_df)
            self.country_stats['USA'] = self._compute_stats(usa_df)
        
        japan_df = self.load_japan_data()
        if not japan_df.empty:
            datasets.append(japan_df)
            self.country_stats['Japan'] = self._compute_stats(japan_df)
        
        # Combine all datasets
        if datasets:
            self.unified_df = pd.concat(datasets, ignore_index=True)
            print(f"\n{'=' * 60}")
            print(f"Total unified records: {len(self.unified_df)}")
            print(f"Countries: {self.unified_df['country'].unique().tolist()}")
            print(f"Cities: {self.unified_df['city'].nunique()}")
        else:
            print("Warning: No datasets loaded!")
            self.unified_df = pd.DataFrame(columns=self.standard_columns)
        
        return self.unified_df
    
    def _compute_stats(self, df: pd.DataFrame) -> Dict:
        """Compute statistics for synthetic data generation."""
        return {
            'rooms_mean': df['rooms'].mean(),
            'rooms_std': df['rooms'].std(),
            'area_mean': df['area_sqm'].mean(),
            'area_std': df['area_sqm'].std(),
            'age_mean': df['building_age'].mean() if 'building_age' in df.columns else 10,
            'age_std': df['building_age'].std() if 'building_age' in df.columns else 5,
            'price_mean': df['price_usd'].mean(),
            'price_std': df['price_usd'].std()
        }
    
    def add_synthetic_cities(self) -> pd.DataFrame:
        """Add synthetic data for missing cities in each country."""
        if self.unified_df is None or self.unified_df.empty:
            print("No data loaded. Call load_all_datasets() first.")
            return pd.DataFrame()
        
        synthetic_dfs = []
        
        for country in self.unified_df['country'].unique():
            if country not in self.country_stats:
                continue
            
            existing_cities = self.unified_df[
                self.unified_df['country'] == country
            ]['city'].unique().tolist()
            
            synthetic = self.generate_synthetic_data(
                country, existing_cities, self.country_stats[country]
            )
            if not synthetic.empty:
                synthetic_dfs.append(synthetic)
        
        if synthetic_dfs:
            synthetic_combined = pd.concat(synthetic_dfs, ignore_index=True)
            self.unified_df = pd.concat([self.unified_df, synthetic_combined], ignore_index=True)
            print(f"\nTotal records after synthetic data: {len(self.unified_df)}")
        
        return self.unified_df
    
    def clean_data(self) -> pd.DataFrame:
        """Clean the unified dataset."""
        if self.unified_df is None or self.unified_df.empty:
            return pd.DataFrame()
        
        print("\nCleaning data...")
        initial_count = len(self.unified_df)
        
        # Remove extreme outliers (prices outside 1st-99th percentile)
        q01 = self.unified_df['price_usd'].quantile(0.01)
        q99 = self.unified_df['price_usd'].quantile(0.99)
        self.unified_df = self.unified_df[
            (self.unified_df['price_usd'] >= q01) & 
            (self.unified_df['price_usd'] <= q99)
        ]
        
        # Clean rooms (1-20 range)
        self.unified_df = self.unified_df[
            (self.unified_df['rooms'] >= 1) & 
            (self.unified_df['rooms'] <= 20)
        ]
        
        # Clean area (10-2000 sqm range)
        self.unified_df = self.unified_df[
            (self.unified_df['area_sqm'] >= 10) & 
            (self.unified_df['area_sqm'] <= 2000)
        ]
        
        # Clean building age (0-150 years)
        self.unified_df['building_age'] = self.unified_df['building_age'].clip(0, 150)
        
        # Standardize furnishing status
        furnishing_map = {
            'furnished': 'Furnished',
            'semi-furnished': 'Semi-Furnished', 
            'semi furnished': 'Semi-Furnished',
            'semifurnished': 'Semi-Furnished',
            'unfurnished': 'Unfurnished',
            'unknown': 'Unknown',
            '': 'Unknown'
        }
        self.unified_df['furnishing_status'] = self.unified_df['furnishing_status'].str.lower().map(
            furnishing_map
        ).fillna('Unknown')
        
        final_count = len(self.unified_df)
        print(f"Cleaned {initial_count - final_count} outlier records")
        print(f"Final record count: {final_count}")
        
        return self.unified_df
    
    def save_processed_data(self, output_path: str = 'outputs/unified_property_data.csv') -> str:
        """Save processed data to CSV."""
        if self.unified_df is None or self.unified_df.empty:
            print("No data to save!")
            return ""
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.unified_df.to_csv(output_path, index=False)
        print(f"\nSaved unified data to: {output_path}")
        return output_path
    
    def get_summary(self) -> Dict:
        """Get summary statistics of the unified dataset."""
        if self.unified_df is None or self.unified_df.empty:
            return {}
        
        return {
            'total_records': len(self.unified_df),
            'countries': self.unified_df['country'].nunique(),
            'cities': self.unified_df['city'].nunique(),
            'country_distribution': self.unified_df['country'].value_counts().to_dict(),
            'price_stats': {
                'mean': self.unified_df['price_usd'].mean(),
                'median': self.unified_df['price_usd'].median(),
                'min': self.unified_df['price_usd'].min(),
                'max': self.unified_df['price_usd'].max()
            },
            'rooms_stats': {
                'mean': self.unified_df['rooms'].mean(),
                'min': self.unified_df['rooms'].min(),
                'max': self.unified_df['rooms'].max()
            },
            'area_stats': {
                'mean': self.unified_df['area_sqm'].mean(),
                'min': self.unified_df['area_sqm'].min(),
                'max': self.unified_df['area_sqm'].max()
            }
        }
    
    def run_pipeline(self, include_synthetic: bool = True) -> pd.DataFrame:
        """Run the complete data processing pipeline.
        
        Args:
            include_synthetic: Whether to generate synthetic data for missing cities
            
        Returns:
            Processed and unified DataFrame
        """
        # Load all datasets
        self.load_all_datasets()
        
        # Add synthetic data if requested
        if include_synthetic:
            self.add_synthetic_cities()
        
        # Clean data
        self.clean_data()
        
        # Save processed data
        self.save_processed_data()
        
        # Print summary
        summary = self.get_summary()
        print("\n" + "=" * 60)
        print("DATA PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Total Records: {summary.get('total_records', 0):,}")
        print(f"Countries: {summary.get('countries', 0)}")
        print(f"Cities: {summary.get('cities', 0)}")
        print("\nCountry Distribution:")
        for country, count in summary.get('country_distribution', {}).items():
            print(f"  {country}: {count:,}")
        
        return self.unified_df


def main():
    """Main function to run data processing."""
    processor = UnifiedDataProcessor(data_dir='datasets')
    df = processor.run_pipeline(include_synthetic=True)
    
    if not df.empty:
        print("\n" + "=" * 60)
        print("Sample of processed data:")
        print("=" * 60)
        print(df.head(10).to_string())


if __name__ == "__main__":
    main()
