import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import argparse
from matplotlib.colors import LinearSegmentedColormap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("india_climate_analysis.log"),
        logging.StreamHandler()
    ]
)

# Directories
DATA_DIR = "Weather_Dataset_Collection/india_climate_watch_data"
ANALYSIS_DIR = os.path.join(DATA_DIR, "analysis")

os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Set Seaborn style for prettier visualizations
sns.set(style="whitegrid")

class WeatherDataAnalyzer:
    """Class to analyze and visualize weather data"""
    
    def __init__(self, data_file=None):
        """
        Initialize with data file
        
        Args:
            data_file (str): Path to CSV file with weather data
        """
        self.data_file = data_file or os.path.join(DATA_DIR, "india_weather_master.csv")
        self.df = None
        self.load_data()
        
    def load_data(self):
        """Load and preprocess the weather data"""
        if not os.path.exists(self.data_file):
            logging.error(f"Data file not found: {self.data_file}")
            return False
        
        try:
            # Load the CSV file
            self.df = pd.read_csv(self.data_file)
            
            # Convert timestamp to datetime
            self.df['collection_time'] = pd.to_datetime(self.df['collection_time'])
            
            # Add date and hour columns for easier analysis
            self.df['date'] = self.df['collection_time'].dt.date
            self.df['hour'] = self.df['collection_time'].dt.hour
            self.df['day_of_week'] = self.df['collection_time'].dt.day_name()
            
            # Add region column based on geographical location
            self.df['region'] = self.df['city'].apply(self.assign_region)
            
            logging.info(f"Loaded {len(self.df)} records from {self.data_file}")
            logging.info(f"Date range: {self.df['date'].min()} to {self.df['date'].max()}")
            logging.info(f"Cities: {', '.join(sorted(self.df['city'].unique()))}")
            
            return True
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            return False
    
    def assign_region(self, city):
        """
        Assign a region to each city
        
        Args:
            city (str): City name
            
        Returns:
            str: Region name
        """
        north = ["Delhi", "Lucknow", "Kanpur", "Patna"]
        south = ["Bangalore", "Hyderabad", "Chennai", "Visakhapatnam"]
        east = ["Kolkata", "Guwahati"]
        west = ["Mumbai", "Ahmedabad", "Pune", "Surat", "Vadodara", "Thane"]
        central = ["Indore", "Bhopal", "Nagpur", "Jaipur"]
        
        if city in north:
            return "North"
        elif city in south:
            return "South"
        elif city in east:
            return "East"
        elif city in west:
            return "West"
        elif city in central:
            return "Central"
        else:
            return "Other"
    
    def generate_summary_statistics(self):
        """
        Generate summary statistics for the dataset
        
        Returns:
            pd.DataFrame: Summary statistics
        """
        if self.df is None:
            return None
        
        # Calculate summary statistics by city
        city_summary = self.df.groupby('city').agg({
            'temp': ['mean', 'min', 'max', 'std'],
            'humidity': ['mean', 'min', 'max'],
            'wind_speed': ['mean', 'max'],
            'pressure': ['mean']
        })
        
        # Calculate weather modes separately to avoid errors
        try:
            weather_modes = self.df.groupby('city')['weather_main'].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
            for city, mode in weather_modes.items():
                city_summary.loc[city, ('weather_main', 'mode')] = mode
        except:
            # If this fails, we'll skip the weather mode
            pass
        
        # Use direct column calculations for overall statistics
        avg_temp = self.df['temp'].mean()
        min_temp = self.df['temp'].min()
        max_temp = self.df['temp'].max()
        std_temp = self.df['temp'].std()
        avg_humidity = self.df['humidity'].mean()
        min_humidity = self.df['humidity'].min()
        max_humidity = self.df['humidity'].max()
        avg_wind = self.df['wind_speed'].mean()
        max_wind = self.df['wind_speed'].max()
        avg_pressure = self.df['pressure'].mean()
        
        # Save summaries to CSV
        city_summary.to_csv(os.path.join(ANALYSIS_DIR, "city_summary_statistics.csv"))
        
        # Generate text summary
        with open(os.path.join(ANALYSIS_DIR, "weather_summary_report.txt"), 'w') as f:
            f.write("IndiaClimateWatch - Weather Summary Report\n")
            f.write("=======================================\n\n")
            
            f.write(f"Analysis Period: {self.df['date'].min()} to {self.df['date'].max()}\n")
            f.write(f"Number of Records: {len(self.df)}\n")
            f.write(f"Cities Covered: {len(self.df['city'].unique())}\n\n")
            
            f.write("Overall Weather Statistics:\n")
            f.write(f"Average Temperature: {avg_temp:.1f}°C\n")
            f.write(f"Temperature Range: {min_temp:.1f}°C to {max_temp:.1f}°C\n")
            f.write(f"Average Humidity: {avg_humidity:.1f}%\n")
            f.write(f"Average Wind Speed: {avg_wind:.1f} m/s\n")
            f.write(f"Average Pressure: {avg_pressure:.1f} hPa\n\n")
            
            f.write("City-wise Temperature Extremes:\n")
            
            # Use a different approach to find the extremes based on direct city data
            city_temp_data = self.df.groupby('city')['temp'].agg(['max', 'min']).reset_index()
            
            # Only add std if we have enough data points
            if len(self.df) > len(self.df['city'].unique()):
                try:
                    # Only calculate std if we have multiple data points per city
                    city_temp_data['std'] = self.df.groupby('city')['temp'].std().values
                except:
                    # If this fails, create a dummy std column
                    city_temp_data['std'] = 0
            else:
                # If we only have one data point per city, we can't calculate std
                city_temp_data['std'] = 0
                
            # Handle NaN values in std column
            city_temp_data['std'] = city_temp_data['std'].fillna(0)
            
            # Find extremes safely
            hottest_city = city_temp_data.loc[city_temp_data['max'].idxmax(), 'city']
            hottest_temp = city_temp_data.loc[city_temp_data['max'].idxmax(), 'max']
            coolest_city = city_temp_data.loc[city_temp_data['min'].idxmin(), 'city']
            coolest_temp = city_temp_data.loc[city_temp_data['min'].idxmin(), 'min']
            
            # Only report most variable city if we have meaningful std values
            f.write(f"Hottest City: {hottest_city} ({hottest_temp:.1f}°C)\n")
            f.write(f"Coolest City: {coolest_city} ({coolest_temp:.1f}°C)\n")
            
            # Only add variability info if we have enough data
            if city_temp_data['std'].max() > 0:
                most_variable_city = city_temp_data.loc[city_temp_data['std'].idxmax(), 'city']
                most_variable_std = city_temp_data.loc[city_temp_data['std'].idxmax(), 'std'] 
                f.write(f"Most Variable Temperature: {most_variable_city} (std: {most_variable_std:.1f}°C)\n")
            else:
                f.write("Temperature Variability: Not enough data points to calculate\n")
            f.write("\n")
            
            f.write("Weather Conditions Summary:\n")
            weather_counts = self.df['weather_main'].value_counts()
            for weather, count in weather_counts.items():
                percentage = 100 * count / len(self.df)
                f.write(f"{weather}: {percentage:.1f}% of observations\n")
        
        logging.info(f"Summary statistics generated and saved to {ANALYSIS_DIR}")
        return city_summary
    
    def plot_temperature_trends(self):
        """Plot temperature trends over time for different cities"""
        if self.df is None:
            return
        
        # Ensure we have date data
        if 'date' not in self.df.columns:
            logging.error("Date column not available for trend analysis")
            return
        
        # Get daily average temperatures by city
        daily_temps = self.df.groupby(['date', 'city'])['temp'].mean().reset_index()
        
        # Select top 5 metropolitan cities for clarity
        metro_cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata']
        city_data = daily_temps[daily_temps['city'].isin(metro_cities)]
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot each city
        for city in metro_cities:
            city_df = city_data[city_data['city'] == city]
            if not city_df.empty:
                plt.plot(city_df['date'], city_df['temp'], marker='o', linestyle='-', label=city)
        
        plt.title('Temperature Trends in Major Indian Metropolitan Cities', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Temperature (°C)', fontsize=12)
        plt.legend()
        plt.grid(True)
        
        # Rotate date labels for readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(ANALYSIS_DIR, 'temperature_trends.png'))
        plt.close()
        
        # Now plot regional average temperatures
        regional_temps = self.df.groupby(['date', 'region'])['temp'].mean().reset_index()
        
        plt.figure(figsize=(12, 8))
        
        for region in regional_temps['region'].unique():
            region_df = regional_temps[regional_temps['region'] == region]
            plt.plot(region_df['date'], region_df['temp'], marker='o', linestyle='-', label=region)
        
        plt.title('Temperature Trends by Region in India', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Temperature (°C)', fontsize=12)
        plt.legend()
        plt.grid(True)
        
        # Rotate date labels for readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(ANALYSIS_DIR, 'regional_temperature_trends.png'))
        plt.close()
        
        logging.info("Temperature trend plots generated")
    
    def plot_temperature_heatmap(self):
        """Create a heatmap of average temperatures across cities"""
        if self.df is None:
            return
        
        # Calculate average temperature by city and date
        pivot_data = self.df.pivot_table(
            index='city', 
            columns='date', 
            values='temp',
            aggfunc='mean'
        )
        
        # Create a custom colormap from cool to hot
        colors = ["#4575B4", "#91BFDB", "#E0F3F8", "#FFFFBF", "#FEE090", "#FC8D59", "#D73027"]
        cmap = LinearSegmentedColormap.from_list("temperature_cmap", colors)
        
        plt.figure(figsize=(16, 10))
        sns.heatmap(
            pivot_data, 
            cmap=cmap,
            linewidths=0.5,
            annot=False,
            center=30,
            vmin=20,
            vmax=40
        )
        
        plt.title('Temperature Heatmap Across Indian Cities', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('City', fontsize=12)
        
        # Rotate date labels for readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(ANALYSIS_DIR, 'temperature_heatmap.png'))
        plt.close()
        
        logging.info("Temperature heatmap generated")
    
    def plot_humidity_comparison(self):
        """Create a box plot comparing humidity levels across cities"""
        if self.df is None:
            return
        
        plt.figure(figsize=(14, 8))
        
        # Create box plot
        sns.boxplot(x='city', y='humidity', data=self.df, palette='viridis')
        
        plt.title('Humidity Comparison Across Indian Cities', fontsize=16)
        plt.xlabel('City', fontsize=12)
        plt.ylabel('Relative Humidity (%)', fontsize=12)
        
        # Rotate city labels for readability
        plt.xticks(rotation=90)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(ANALYSIS_DIR, 'humidity_comparison.png'))
        plt.close()
        
        # Create regional humidity comparison
        plt.figure(figsize=(10, 6))
        
        # Create box plot by region
        sns.boxplot(x='region', y='humidity', data=self.df, palette='viridis')
        
        plt.title('Humidity Comparison by Region', fontsize=16)
        plt.xlabel('Region', fontsize=12)
        plt.ylabel('Relative Humidity (%)', fontsize=12)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(ANALYSIS_DIR, 'regional_humidity_comparison.png'))
        plt.close()
        
        logging.info("Humidity comparison plots generated")
    
    def plot_weather_conditions(self):
        """Create a bar chart of weather conditions frequency"""
        if self.df is None:
            return
        
        # Count frequency of each weather condition
        weather_counts = self.df['weather_main'].value_counts()
        
        plt.figure(figsize=(12, 6))
        
        # Create bar chart
        sns.barplot(x=weather_counts.index, y=weather_counts.values, palette='viridis')
        
        plt.title('Frequency of Weather Conditions', fontsize=16)
        plt.xlabel('Weather Condition', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(ANALYSIS_DIR, 'weather_conditions.png'))
        plt.close()
        
        # Create pie chart of weather conditions
        plt.figure(figsize=(10, 10))
        
        # Create pie chart
        plt.pie(
            weather_counts.values, 
            labels=weather_counts.index, 
            autopct='%1.1f%%',
            colors=sns.color_palette('viridis', len(weather_counts)),
            startangle=90,
            shadow=True
        )
        
        plt.title('Distribution of Weather Conditions', fontsize=16)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(ANALYSIS_DIR, 'weather_conditions_pie.png'))
        plt.close()
        
        # Weather conditions by region
        plt.figure(figsize=(14, 8))
        
        # Create a crosstab of region and weather condition
        region_weather = pd.crosstab(
            self.df['region'], 
            self.df['weather_main'], 
            normalize='index'
        ) * 100  # Convert to percentage
        
        # Create a stacked bar chart
        region_weather.plot(kind='bar', stacked=True, colormap='viridis', figsize=(14, 8))
        
        plt.title('Weather Conditions by Region', fontsize=16)
        plt.xlabel('Region', fontsize=12)
        plt.ylabel('Percentage', fontsize=12)
        plt.legend(title='Weather Condition')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(ANALYSIS_DIR, 'weather_conditions_by_region.png'))
        plt.close()
        
        logging.info("Weather conditions plots generated")
    
    def plot_wind_analysis(self):
        """Create visualizations for wind speed analysis"""
        if self.df is None:
            return
        
        # Wind speed by city
        plt.figure(figsize=(14, 8))
        
        # Create box plot
        sns.boxplot(x='city', y='wind_speed', data=self.df, palette='Blues')
        
        plt.title('Wind Speed Comparison Across Indian Cities', fontsize=16)
        plt.xlabel('City', fontsize=12)
        plt.ylabel('Wind Speed (m/s)', fontsize=12)
        
        # Rotate city labels for readability
        plt.xticks(rotation=90)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(ANALYSIS_DIR, 'wind_speed_comparison.png'))
        plt.close()
        
        # Wind rose chart (simplified version showing directional distribution)
        if 'wind_direction' in self.df.columns:
            # Create directional categories
            self.df['wind_direction_cat'] = pd.cut(
                self.df['wind_direction'],
                bins=[-1, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 360],
                labels=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N']
            )
            
            # Count frequency of each direction
            direction_counts = self.df['wind_direction_cat'].value_counts().sort_index()
            
            plt.figure(figsize=(10, 10), subplot_kw={'projection': 'polar'})
            
            # Convert to radians for polar plot
            angles = np.linspace(0, 2*np.pi, len(direction_counts), endpoint=False)
            
            # Plot wind rose
            plt.bar(
                angles, 
                direction_counts.values,
                width=0.5,
                alpha=0.8,
                color=sns.color_palette('Blues', len(direction_counts))
            )
            
            # Set direction labels
            plt.xticks(angles, direction_counts.index)
            
            plt.title('Wind Direction Distribution', fontsize=16)
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(os.path.join(ANALYSIS_DIR, 'wind_direction_rose.png'))
            plt.close()
        
        logging.info("Wind analysis plots generated")
    
    def plot_correlation_analysis(self):
        """Create correlation heatmap for weather parameters"""
        if self.df is None:
            return
        
        # Select numerical columns for correlation analysis
        numerical_cols = ['temp', 'feels_like', 'humidity', 'pressure', 'wind_speed', 'clouds']
        
        # Calculate correlation matrix
        corr_matrix = self.df[numerical_cols].corr()
        
        plt.figure(figsize=(10, 8))
        
        # Create correlation heatmap
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap='coolwarm',
            linewidths=0.5,
            vmin=-1,
            vmax=1,
            square=True,
            fmt='.2f'
        )
        
        plt.title('Correlation Between Weather Parameters', fontsize=16)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(ANALYSIS_DIR, 'weather_correlation.png'))
        plt.close()
        
        # Scatter plot of temperature vs. humidity
        plt.figure(figsize=(10, 8))
        
        sns.scatterplot(
            x='temp',
            y='humidity',
            data=self.df,
            hue='region',
            palette='viridis',
            alpha=0.7
        )
        
        plt.title('Temperature vs. Humidity Relationship', fontsize=16)
        plt.xlabel('Temperature (°C)', fontsize=12)
        plt.ylabel('Relative Humidity (%)', fontsize=12)
        plt.legend(title='Region')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(ANALYSIS_DIR, 'temp_humidity_relationship.png'))
        plt.close()
        
        logging.info("Correlation analysis plots generated")
    
    def plot_temperature_distribution(self):
        """Create histogram and KDE of temperature distribution"""
        if self.df is None:
            return
        
        plt.figure(figsize=(12, 8))
        
        # Create distribution plot (histogram with KDE)
        sns.histplot(
            data=self.df,
            x='temp',
            hue='region',
            kde=True,
            bins=20,
            palette='viridis',
            alpha=0.6
        )
        
        plt.title('Temperature Distribution by Region', fontsize=16)
        plt.xlabel('Temperature (°C)', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.legend(title='Region')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(ANALYSIS_DIR, 'temperature_distribution.png'))
        plt.close()
        
        # Create violin plot for temperature by city
        plt.figure(figsize=(14, 10))
        
        # Sort cities by average temperature
        city_order = self.df.groupby('city')['temp'].mean().sort_values().index
        
        sns.violinplot(
            x='city',
            y='temp',
            data=self.df,
            order=city_order,
            palette='viridis',
            inner='quartile'
        )
        
        plt.title('Temperature Distribution Across Cities', fontsize=16)
        plt.xlabel('City', fontsize=12)
        plt.ylabel('Temperature (°C)', fontsize=12)
        plt.xticks(rotation=90)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(ANALYSIS_DIR, 'temperature_distribution_by_city.png'))
        plt.close()
        
        logging.info("Temperature distribution plots generated")
    
    def generate_city_profile(self, city_name):
        """
        Generate a detailed profile for a specific city
        
        Args:
            city_name (str): Name of the city
        """
        if self.df is None:
            return
        
        # Filter data for the specified city
        city_data = self.df[self.df['city'] == city_name]
        
        if city_data.empty:
            logging.warning(f"No data found for city: {city_name}")
            return
        
        # Create a directory for city profiles
        city_dir = os.path.join(ANALYSIS_DIR, "city_profiles")
        os.makedirs(city_dir, exist_ok=True)
        
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(15, 12))
        
        # 1. Temperature trend over time
        ax1 = plt.subplot(2, 2, 1)
        daily_temp = city_data.groupby('date')['temp'].mean()
        ax1.plot(daily_temp.index, daily_temp.values, marker='o', linestyle='-', color='#FF5733')
        ax1.set_title(f'Temperature Trend for {city_name}', fontsize=14)
        ax1.set_xlabel('Date', fontsize=10)
        ax1.set_ylabel('Temperature (°C)', fontsize=10)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 2. Weather conditions pie chart
        ax2 = plt.subplot(2, 2, 2)
        weather_counts = city_data['weather_main'].value_counts()
        ax2.pie(
            weather_counts.values, 
            labels=weather_counts.index, 
            autopct='%1.1f%%',
            colors=sns.color_palette('viridis', len(weather_counts)),
            startangle=90
        )
        ax2.set_title(f'Weather Conditions for {city_name}', fontsize=14)
        ax2.axis('equal')
        
        # 3. Temperature vs. humidity scatter plot
        ax3 = plt.subplot(2, 2, 3)
        sns.scatterplot(
            x='temp',
            y='humidity',
            data=city_data,
            hue='weather_main',
            palette='viridis',
            alpha=0.7,
            ax=ax3
        )
        ax3.set_title(f'Temperature vs. Humidity in {city_name}', fontsize=14)
        ax3.set_xlabel('Temperature (°C)', fontsize=10)
        ax3.set_ylabel('Relative Humidity (%)', fontsize=10)
        ax3.legend(title='Weather')
        
        # 4. Daily temperature range
        ax4 = plt.subplot(2, 2, 4)
        daily_min = city_data.groupby('date')['temp_min'].min()
        daily_max = city_data.groupby('date')['temp_max'].max()
        
        ax4.fill_between(
            daily_min.index,
            daily_min.values,
            daily_max.values,
            alpha=0.3,
            color='#3399FF'
        )
        ax4.plot(daily_min.index, daily_min.values, color='blue', label='Min')
        ax4.plot(daily_max.index, daily_max.values, color='red', label='Max')
        
        ax4.set_title(f'Daily Temperature Range for {city_name}', fontsize=14)
        ax4.set_xlabel('Date', fontsize=10)
        ax4.set_ylabel('Temperature (°C)', fontsize=10)
        ax4.tick_params(axis='x', rotation=45)
        ax4.legend()
        ax4.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(city_dir, f'{city_name.lower().replace(" ", "_")}_profile.png'))
        plt.close()
        
        # Generate text summary
        with open(os.path.join(city_dir, f'{city_name.lower().replace(" ", "_")}_summary.txt'), 'w') as f:
            f.write(f"Weather Profile for {city_name}\n")
            f.write("=" * (19 + len(city_name)) + "\n\n")
            
            f.write(f"Analysis Period: {city_data['date'].min()} to {city_data['date'].max()}\n")
            f.write(f"Number of Records: {len(city_data)}\n\n")
            
            f.write("Temperature Statistics:\n")
            f.write(f"  Average: {city_data['temp'].mean():.1f}°C\n")
            f.write(f"  Minimum: {city_data['temp_min'].min():.1f}°C\n")
            f.write(f"  Maximum: {city_data['temp_max'].max():.1f}°C\n")
            f.write(f"  Standard Deviation: {city_data['temp'].std():.1f}°C\n\n")
            
            f.write("Humidity Statistics:\n")
            f.write(f"  Average: {city_data['humidity'].mean():.1f}%\n")
            f.write(f"  Minimum: {city_data['humidity'].min():.1f}%\n")
            f.write(f"  Maximum: {city_data['humidity'].max():.1f}%\n\n")
            
            f.write("Wind Statistics:\n")
            f.write(f"  Average Speed: {city_data['wind_speed'].mean():.1f} m/s\n")
            f.write(f"  Maximum Speed: {city_data['wind_speed'].max():.1f} m/s\n\n")
            
            f.write("Predominant Weather Conditions:\n")
            for condition, count in weather_counts.items():
                percentage = 100 * count / len(city_data)
                f.write(f"  {condition}: {percentage:.1f}%\n")
        
        logging.info(f"City profile generated for {city_name}")
    
    def generate_all_analyses(self):
        """Generate all available analyses and visualizations"""
        logging.info("Generating all analyses...")
        
        self.generate_summary_statistics()
        self.plot_temperature_trends()
        self.plot_temperature_heatmap()
        self.plot_humidity_comparison()
        self.plot_weather_conditions()
        self.plot_wind_analysis()
        self.plot_correlation_analysis()
        self.plot_temperature_distribution()
        
        # Generate profiles for metropolitan cities
        for city in ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata']:
            self.generate_city_profile(city)
        
        logging.info("All analyses completed")

def main():
    """Main function to run the weather data analyzer"""
    parser = argparse.ArgumentParser(description="IndiaClimateWatch Weather Data Analyzer")
    parser.add_argument("--data-file", help="Path to the weather data CSV file")
    parser.add_argument("--city", help="Generate profile for a specific city")
    
    args = parser.parse_args()
    
    analyzer = WeatherDataAnalyzer(args.data_file)
    
    if args.city:
        analyzer.generate_city_profile(args.city)
    else:
        analyzer.generate_all_analyses()
    
    logging.info(f"Analysis completed. Results saved to {os.path.abspath(ANALYSIS_DIR)}")

if __name__ == "__main__":
    main()