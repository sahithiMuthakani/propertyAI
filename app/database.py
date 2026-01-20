"""
Comprehensive Data Processing and Exploration Script
=====================================================

This script provides extensive data analysis capabilities including:
- Data loading and structure exploration
- Data type identification (categorical vs numerical)
- Missing value analysis and handling
- Descriptive statistics generation
- Automated visualizations
- Insight extraction and reporting

Author: AI Agent
Date: 2025-12-04
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class DataExplorer:
    """Comprehensive data exploration and analysis class"""
    
    def __init__(self, file_path):
        """
        Initialize the DataExplorer with a dataset file path
        
        Args:
            file_path (str): Path to the CSV file
        """
        self.file_path = file_path
        self.df = None
        self.numerical_cols = []
        self.categorical_cols = []
        self.insights = []
        self.output_dir = 'outputs'
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_data(self):
        """Load the dataset from CSV file"""
        try:
            print(f"\n{'='*80}")
            print(f"LOADING DATA FROM: {self.file_path}")
            print(f"{'='*80}\n")
            
            self.df = pd.read_csv(self.file_path)
            print(f"✓ Dataset loaded successfully!")
            print(f"  Rows: {self.df.shape[0]:,}")
            print(f"  Columns: {self.df.shape[1]}")
            
            self.insights.append(f"Dataset contains {self.df.shape[0]:,} rows and {self.df.shape[1]} columns")
            return True
            
        except Exception as e:
            print(f"✗ Error loading data: {str(e)}")
            return False
    
    def explore_structure(self):
        """Explore the basic structure of the dataset"""
        print(f"\n{'='*80}")
        print("DATA STRUCTURE EXPLORATION")
        print(f"{'='*80}\n")
        
        # Display first few rows
        print("First 5 rows:")
        print(self.df.head())
        
        # Display basic info
        print("\n\nDataset Information:")
        print("-" * 80)
        self.df.info()
        
        # Column names
        print(f"\n\nColumn Names ({len(self.df.columns)}):")
        print("-" * 80)
        for i, col in enumerate(self.df.columns, 1):
            print(f"{i:2d}. {col}")
    
    def identify_data_types(self):
        """Identify and categorize columns into numerical and categorical"""
        print(f"\n{'='*80}")
        print("DATA TYPE IDENTIFICATION")
        print(f"{'='*80}\n")
        
        # Numerical columns
        self.numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Categorical columns (including object and low-cardinality numerical)
        self.categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # Check for numerical columns with low cardinality (could be categorical)
        for col in self.numerical_cols.copy():
            if self.df[col].nunique() < 10:
                print(f"  Note: '{col}' is numerical but has only {self.df[col].nunique()} unique values - might be categorical")
        
        print(f"Numerical Columns ({len(self.numerical_cols)}):")
        for col in self.numerical_cols:
            print(f"  • {col} ({self.df[col].dtype})")
        
        print(f"\nCategorical Columns ({len(self.categorical_cols)}):")
        for col in self.categorical_cols:
            unique_count = self.df[col].nunique()
            print(f"  • {col} ({self.df[col].dtype}) - {unique_count} unique values")
        
        self.insights.append(f"Dataset has {len(self.numerical_cols)} numerical and {len(self.categorical_cols)} categorical features")
    
    def analyze_missing_values(self):
        """Analyze missing values in the dataset"""
        print(f"\n{'='*80}")
        print("MISSING VALUE ANALYSIS")
        print(f"{'='*80}\n")
        
        missing_data = pd.DataFrame({
            'Column': self.df.columns,
            'Missing_Count': self.df.isnull().sum(),
            'Missing_Percentage': (self.df.isnull().sum() / len(self.df)) * 100
        })
        
        missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)
        
        if len(missing_data) == 0:
            print("✓ No missing values found in the dataset!")
            self.insights.append("Dataset has no missing values")
        else:
            print(f"Found missing values in {len(missing_data)} columns:\n")
            print(missing_data.to_string(index=False))
            
            # Visualize missing values
            if len(missing_data) > 0:
                plt.figure(figsize=(12, max(6, len(missing_data) * 0.4)))
                plt.barh(missing_data['Column'], missing_data['Missing_Percentage'], color='#e74c3c')
                plt.xlabel('Missing Percentage (%)', fontsize=12, fontweight='bold')
                plt.ylabel('Columns', fontsize=12, fontweight='bold')
                plt.title('Missing Values Distribution', fontsize=14, fontweight='bold', pad=20)
                plt.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                plt.savefig(f'{self.output_dir}/missing_values.png', dpi=300, bbox_inches='tight')
                plt.close()
                print(f"\n✓ Missing values visualization saved to '{self.output_dir}/missing_values.png'")
            
            total_missing_pct = (self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1])) * 100
            self.insights.append(f"{len(missing_data)} columns have missing values ({total_missing_pct:.2f}% of total data)")
    
    def handle_missing_values(self):
        """Handle missing values with appropriate strategies"""
        print(f"\n{'='*80}")
        print("HANDLING MISSING VALUES")
        print(f"{'='*80}\n")
        
        original_shape = self.df.shape
        
        # Strategy 1: Remove columns with >50% missing data
        high_missing_cols = self.df.columns[self.df.isnull().mean() > 0.5].tolist()
        if high_missing_cols:
            print(f"Removing {len(high_missing_cols)} columns with >50% missing data:")
            for col in high_missing_cols:
                print(f"  • {col}")
            self.df = self.df.drop(columns=high_missing_cols)
        
        # Strategy 2: Impute numerical columns with median
        numerical_missing = [col for col in self.numerical_cols if col in self.df.columns and self.df[col].isnull().any()]
        if numerical_missing:
            print(f"\nImputing {len(numerical_missing)} numerical columns with median:")
            for col in numerical_missing:
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
                print(f"  • {col} (median: {median_val:.2f})")
        
        # Strategy 3: Impute categorical columns with mode
        categorical_missing = [col for col in self.categorical_cols if col in self.df.columns and self.df[col].isnull().any()]
        if categorical_missing:
            print(f"\nImputing {len(categorical_missing)} categorical columns with mode:")
            for col in categorical_missing:
                mode_val = self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown'
                self.df[col].fillna(mode_val, inplace=True)
                print(f"  • {col} (mode: {mode_val})")
        
        # Update column lists
        self.numerical_cols = [col for col in self.numerical_cols if col in self.df.columns]
        self.categorical_cols = [col for col in self.categorical_cols if col in self.df.columns]
        
        print(f"\n✓ Missing value handling complete!")
        print(f"  Original shape: {original_shape}")
        print(f"  Final shape: {self.df.shape}")
        
        if original_shape != self.df.shape:
            self.insights.append(f"Removed {original_shape[1] - self.df.shape[1]} columns with excessive missing data")
    
    def generate_descriptive_statistics(self):
        """Generate comprehensive descriptive statistics"""
        print(f"\n{'='*80}")
        print("DESCRIPTIVE STATISTICS")
        print(f"{'='*80}\n")
        
        # Numerical statistics
        if self.numerical_cols:
            print("Numerical Features Statistics:")
            print("-" * 80)
            stats = self.df[self.numerical_cols].describe()
            print(stats)
            
            # Save to CSV
            stats.to_csv(f'{self.output_dir}/numerical_statistics.csv')
            print(f"\n✓ Numerical statistics saved to '{self.output_dir}/numerical_statistics.csv'")
        
        # Categorical statistics
        if self.categorical_cols:
            print(f"\n\nCategorical Features Statistics:")
            print("-" * 80)
            for col in self.categorical_cols[:5]:  # Show top 5
                print(f"\n{col}:")
                print(f"  Unique values: {self.df[col].nunique()}")
                print(f"  Most common: {self.df[col].mode()[0] if not self.df[col].mode().empty else 'N/A'}")
                top_5 = self.df[col].value_counts().head(5)
                print("  Top 5 values:")
                for val, count in top_5.items():
                    print(f"    • {val}: {count} ({count/len(self.df)*100:.1f}%)")
    
    def visualize_data(self):
        """Create comprehensive visualizations based on data types"""
        print(f"\n{'='*80}")
        print("DATA VISUALIZATION")
        print(f"{'='*80}\n")
        
        # Visualize numerical features
        if self.numerical_cols:
            print(f"Creating visualizations for {len(self.numerical_cols)} numerical features...")
            
            # Histograms
            num_cols_to_plot = min(9, len(self.numerical_cols))
            fig, axes = plt.subplots(3, 3, figsize=(15, 12))
            fig.suptitle('Numerical Features Distribution', fontsize=16, fontweight='bold', y=0.995)
            axes = axes.flatten()
            
            for idx, col in enumerate(self.numerical_cols[:num_cols_to_plot]):
                self.df[col].hist(bins=30, ax=axes[idx], color='#3498db', edgecolor='black', alpha=0.7)
                axes[idx].set_title(col, fontweight='bold')
                axes[idx].set_xlabel('Value')
                axes[idx].set_ylabel('Frequency')
                axes[idx].grid(alpha=0.3)
            
            # Hide unused subplots
            for idx in range(num_cols_to_plot, 9):
                axes[idx].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/numerical_distributions.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Histograms saved to '{self.output_dir}/numerical_distributions.png'")
            
            # Box plots
            fig, axes = plt.subplots(3, 3, figsize=(15, 12))
            fig.suptitle('Numerical Features Box Plots (Outlier Detection)', fontsize=16, fontweight='bold', y=0.995)
            axes = axes.flatten()
            
            for idx, col in enumerate(self.numerical_cols[:num_cols_to_plot]):
                self.df.boxplot(column=col, ax=axes[idx], patch_artist=True,
                               boxprops=dict(facecolor='#2ecc71', alpha=0.7),
                               medianprops=dict(color='#e74c3c', linewidth=2))
                axes[idx].set_title(col, fontweight='bold')
                axes[idx].set_ylabel('Value')
                axes[idx].grid(alpha=0.3)
            
            for idx in range(num_cols_to_plot, 9):
                axes[idx].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/numerical_boxplots.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Box plots saved to '{self.output_dir}/numerical_boxplots.png'")
            
            # Correlation heatmap
            if len(self.numerical_cols) > 1:
                plt.figure(figsize=(12, 10))
                correlation = self.df[self.numerical_cols].corr()
                mask = np.triu(np.ones_like(correlation, dtype=bool))
                sns.heatmap(correlation, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                           center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
                plt.title('Correlation Heatmap - Numerical Features', fontsize=14, fontweight='bold', pad=20)
                plt.tight_layout()
                plt.savefig(f'{self.output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Correlation heatmap saved to '{self.output_dir}/correlation_heatmap.png'")
                
                # Find high correlations
                high_corr = []
                for i in range(len(correlation.columns)):
                    for j in range(i+1, len(correlation.columns)):
                        if abs(correlation.iloc[i, j]) > 0.7:
                            high_corr.append((correlation.columns[i], correlation.columns[j], correlation.iloc[i, j]))
                
                if high_corr:
                    self.insights.append(f"Found {len(high_corr)} highly correlated feature pairs (|r| > 0.7)")
        
        # Visualize categorical features
        if self.categorical_cols:
            print(f"\nCreating visualizations for {len(self.categorical_cols)} categorical features...")
            
            cat_cols_to_plot = min(6, len(self.categorical_cols))
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle('Categorical Features Distribution', fontsize=16, fontweight='bold', y=0.995)
            axes = axes.flatten()
            
            for idx, col in enumerate(self.categorical_cols[:cat_cols_to_plot]):
                top_values = self.df[col].value_counts().head(10)
                top_values.plot(kind='bar', ax=axes[idx], color='#9b59b6', edgecolor='black', alpha=0.8)
                axes[idx].set_title(col, fontweight='bold')
                axes[idx].set_xlabel('Category')
                axes[idx].set_ylabel('Count')
                axes[idx].tick_params(axis='x', rotation=45)
                axes[idx].grid(alpha=0.3)
            
            for idx in range(cat_cols_to_plot, 6):
                axes[idx].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/categorical_distributions.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Categorical distributions saved to '{self.output_dir}/categorical_distributions.png'")
    
    def extract_insights(self):
        """Extract and compile key insights from the data"""
        print(f"\n{'='*80}")
        print("INSIGHT EXTRACTION")
        print(f"{'='*80}\n")
        
        # Data quality insights
        duplicate_rows = self.df.duplicated().sum()
        if duplicate_rows > 0:
            self.insights.append(f"Warning: {duplicate_rows} duplicate rows found ({duplicate_rows/len(self.df)*100:.2f}%)")
        
        # Numerical insights
        for col in self.numerical_cols[:5]:  # Analyze top 5
            skewness = self.df[col].skew()
            if abs(skewness) > 1:
                self.insights.append(f"'{col}' is highly skewed ({skewness:.2f}) - consider transformation")
            
            # Outlier detection using IQR
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))).sum()
            if outliers > len(self.df) * 0.05:  # More than 5% outliers
                self.insights.append(f"'{col}' has {outliers} outliers ({outliers/len(self.df)*100:.1f}%)")
        
        # Categorical insights
        for col in self.categorical_cols[:3]:
            cardinality = self.df[col].nunique()
            if cardinality > len(self.df) * 0.5:
                self.insights.append(f"'{col}' has very high cardinality ({cardinality}) - might need encoding strategy")
            elif cardinality == 2:
                self.insights.append(f"'{col}' is binary - suitable for direct encoding")
        
        # Print all insights
        print("Key Insights for Model Training:")
        print("-" * 80)
        for i, insight in enumerate(self.insights, 1):
            print(f"{i:2d}. {insight}")
    
    def generate_report(self):
        """Generate comprehensive markdown report"""
        report_path = f'{self.output_dir}/data_analysis_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Data Analysis Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Dataset:** {os.path.basename(self.file_path)}\n\n")
            f.write("---\n\n")
            
            f.write("## Dataset Summary\n\n")
            f.write(f"- **Total Rows:** {self.df.shape[0]:,}\n")
            f.write(f"- **Total Columns:** {self.df.shape[1]}\n")
            f.write(f"- **Numerical Features:** {len(self.numerical_cols)}\n")
            f.write(f"- **Categorical Features:** {len(self.categorical_cols)}\n")
            f.write(f"- **Missing Values:** {self.df.isnull().sum().sum():,}\n")
            f.write(f"- **Duplicate Rows:** {self.df.duplicated().sum():,}\n\n")
            
            f.write("## Column Information\n\n")
            f.write("### Numerical Columns\n\n")
            for col in self.numerical_cols:
                f.write(f"- `{col}` ({self.df[col].dtype})\n")
            
            f.write("\n### Categorical Columns\n\n")
            for col in self.categorical_cols:
                f.write(f"- `{col}` ({self.df[col].dtype}) - {self.df[col].nunique()} unique values\n")
            
            f.write("\n## Key Insights\n\n")
            for i, insight in enumerate(self.insights, 1):
                f.write(f"{i}. {insight}\n")
            
            f.write("\n## Recommendations for Model Training\n\n")
            f.write("### Data Preprocessing\n")
            f.write("1. **Feature Scaling:** Apply StandardScaler or MinMaxScaler to numerical features\n")
            f.write("2. **Encoding:** Use OneHotEncoding for categorical features with low cardinality\n")
            f.write("3. **Outlier Handling:** Consider removing or capping outliers in skewed features\n")
            f.write("4. **Feature Engineering:** Create interaction features from highly correlated variables\n\n")
            
            f.write("### Model Selection\n")
            f.write("Based on the data characteristics:\n")
            f.write("- **Regression Task:** Random Forest Regressor, Gradient Boosting, or XGBoost\n")
            f.write("- **Classification Task:** Random Forest Classifier, Logistic Regression, or SVM\n")
            f.write("- **Validation Strategy:** 5-fold cross-validation recommended\n\n")
            
            f.write("## Visualizations\n\n")
            f.write("All visualizations have been saved to the `outputs/` folder:\n\n")
            if os.path.exists(f'{self.output_dir}/numerical_distributions.png'):
                f.write("- ![Numerical Distributions](numerical_distributions.png)\n")
            if os.path.exists(f'{self.output_dir}/correlation_heatmap.png'):
                f.write("- ![Correlation Heatmap](correlation_heatmap.png)\n")
            if os.path.exists(f'{self.output_dir}/categorical_distributions.png'):
                f.write("- ![Categorical Distributions](categorical_distributions.png)\n")
        
        print(f"\n✓ Comprehensive report saved to '{report_path}'")
    
    def save_processed_data(self):
        """Save the processed dataset"""
        output_path = f'{self.output_dir}/processed_data.csv'
        self.df.to_csv(output_path, index=False)
        print(f"✓ Processed dataset saved to '{output_path}'")
        
        # Also save a pickled version for faster loading
        pickle_path = f'{self.output_dir}/processed_data.pkl'
        self.df.to_pickle(pickle_path)
        print(f"✓ Pickled dataset saved to '{pickle_path}'")
    
    def run_complete_analysis(self):
        """Run the complete data analysis pipeline"""
        if not self.load_data():
            return False
        
        self.explore_structure()
        self.identify_data_types()
        self.analyze_missing_values()
        self.handle_missing_values()
        self.generate_descriptive_statistics()
        self.visualize_data()
        self.extract_insights()
        self.generate_report()
        self.save_processed_data()
        
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE!")
        print(f"{'='*80}\n")
        print(f"All outputs saved to: {os.path.abspath(self.output_dir)}/")
        print("\nGenerated files:")
        for file in os.listdir(self.output_dir):
            file_path = os.path.join(self.output_dir, file)
            size = os.path.getsize(file_path)
            size_str = f"{size/1024:.1f} KB" if size < 1024*1024 else f"{size/(1024*1024):.1f} MB"
            print(f"  • {file} ({size_str})")
        
        return True


def main():
    """Main function to run data processing"""
    print("\n" + "="*80)
    print(" "*20 + "DATA PROCESSING & EXPLORATION TOOL")
    print("="*80 + "\n")
    
    # Check if dataset path is provided
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        # List available datasets
        dataset_dir = 'datasets'
        if os.path.exists(dataset_dir):
            print("Available datasets in 'datasets/' folder:")
            print("-" * 80)
            datasets = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]
            for i, dataset in enumerate(datasets, 1):
                size = os.path.getsize(os.path.join(dataset_dir, dataset))
                size_mb = size / (1024 * 1024)
                print(f"{i:2d}. {dataset:<40} ({size_mb:.2f} MB)")
            
            print("\n" + "-" * 80)
            choice = input("\nEnter dataset number (or full path): ").strip()
            
            if choice.isdigit() and 1 <= int(choice) <= len(datasets):
                dataset_path = os.path.join(dataset_dir, datasets[int(choice) - 1])
            else:
                dataset_path = choice
        else:
            dataset_path = input("Enter dataset path: ").strip()
    
    # Check if file exists
    if not os.path.exists(dataset_path):
        print(f"\n✗ Error: File not found - {dataset_path}")
        return
    
    # Run analysis
    explorer = DataExplorer(dataset_path)
    success = explorer.run_complete_analysis()
    
    if success:
        print("\n" + "="*80)
        print("Next Steps:")
        print("-" * 80)
        print("1. Review the generated visualizations in the 'outputs/' folder")
        print("2. Read the data_analysis_report.md for detailed insights")
        print("3. Use processed_data.csv for model training in model.py")
        print("4. Consider the recommendations for feature engineering and model selection")
        print("="*80 + "\n")


if __name__ == "__main__":
    main()
