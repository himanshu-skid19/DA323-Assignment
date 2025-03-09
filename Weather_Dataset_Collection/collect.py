"""
IndiaClimateWatch - Weather Data Collector
-----------------------------------------
This script collects weather data from OpenWeatherMap API for 20 Indian cities
and saves it to CSV files. It can be run as a scheduled task to collect data
over a period of time.
"""

import os
import time
import json
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("india_climate_watch.log"),
        logging.StreamHandler()
    ]
)

# Create data directory
DATA_DIR = "india_climate_watch_data"
os.makedirs(DATA_DIR, exist_ok=True)

# List of 20 Indian cities (representing different regions)
INDIAN_CITIES = [
    {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777},
    {"name": "Delhi", "lat": 28.6139, "lon": 77.2090},
    {"name": "Bangalore", "lat": 12.9716, "lon": 77.5946},
    {"name": "Hyderabad", "lat": 17.3850, "lon": 78.4867},
    {"name": "Chennai", "lat": 13.0827, "lon": 80.2707},
    {"name": "Kolkata", "lat": 22.5726, "lon": 88.3639},
    {"name": "Jaipur", "lat": 26.9124, "lon": 75.7873},
    {"name": "Ahmedabad", "lat": 23.0225, "lon": 72.5714},
    {"name": "Pune", "lat": 18.5204, "lon": 73.8567},
    {"name": "Surat", "lat": 21.1702, "lon": 72.8311},
    {"name": "Lucknow", "lat": 26.8467, "lon": 80.9462},
    {"name": "Kanpur", "lat": 26.4499, "lon": 80.3319},
    {"name": "Nagpur", "lat": 21.1458, "lon": 79.0882},
    {"name": "Indore", "lat": 22.7196, "lon": 75.8577},
    {"name": "Thane", "lat": 19.2183, "lon": 72.9781},
    {"name": "Bhopal", "lat": 23.2599, "lon": 77.4126},
    {"name": "Visakhapatnam", "lat": 17.6868, "lon": 83.2185},
    {"name": "Patna", "lat": 25.5941, "lon": 85.1376},
    {"name": "Vadodara", "lat": 22.3072, "lon": 73.1812},
    {"name": "Guwahati", "lat": 26.1158, "lon": 91.7086}
]

class WeatherDataCollector:
    """Class to handle OpenWeatherMap API data collection"""
    
    def __init__(self, api_key):
        """Initialize with API key"""
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
        
    def get_current_weather(self, lat, lon):
        """
        Get current weather data for a specific location
        
        Args:
            lat (float): Latitude
            lon (float): Longitude
            
        Returns:
            dict: Weather data or None if request failed
        """
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "metric"  # Use metric units (Celsius, m/s, etc.)
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()  # Raise exception for HTTP errors
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching weather data: {e}")
            return None
            
    def get_forecast(self, lat, lon):
        """
        Get 5-day weather forecast for a specific location
        
        Args:
            lat (float): Latitude
            lon (float): Longitude
            
        Returns:
            dict: Forecast data or None if request failed
        """
        forecast_url = "https://api.openweathermap.org/data/2.5/forecast"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "metric"  # Use metric units (Celsius, m/s, etc.)
        }
        
        try:
            response = requests.get(forecast_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching forecast data: {e}")
            return None
    
    def collect_data_for_cities(self, cities=INDIAN_CITIES):
        """
        Collect current weather data for a list of cities
        
        Args:
            cities (list): List of city dictionaries with name, lat, lon
            
        Returns:
            list: List of weather data dictionaries
        """
        weather_data = []
        collection_time = datetime.now()
        
        for city in cities:
            logging.info(f"Collecting data for {city['name']}...")
            data = self.get_current_weather(city['lat'], city['lon'])
            
            if data:
                # Extract and format the weather data
                weather_info = {
                    "city": city['name'],
                    "collection_time": collection_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "timestamp": data.get("dt", 0),
                    "temp": data.get("main", {}).get("temp"),
                    "temp_min": data.get("main", {}).get("temp_min"),
                    "temp_max": data.get("main", {}).get("temp_max"),
                    "feels_like": data.get("main", {}).get("feels_like"),
                    "humidity": data.get("main", {}).get("humidity"),
                    "pressure": data.get("main", {}).get("pressure"),
                    "wind_speed": data.get("wind", {}).get("speed"),
                    "wind_direction": data.get("wind", {}).get("deg"),
                    "clouds": data.get("clouds", {}).get("all"),
                    "weather_main": data.get("weather", [{}])[0].get("main"),
                    "weather_description": data.get("weather", [{}])[0].get("description")
                }
                
                # Add rain and snow if available
                if "rain" in data:
                    weather_info["rain_1h"] = data["rain"].get("1h", 0)
                else:
                    weather_info["rain_1h"] = 0
                    
                if "snow" in data:
                    weather_info["snow_1h"] = data["snow"].get("1h", 0)
                else:
                    weather_info["snow_1h"] = 0
                
                weather_data.append(weather_info)
                logging.info(f"Successfully collected data for {city['name']}")
            else:
                logging.warning(f"Failed to collect data for {city['name']}")
            
            # Add a small delay to avoid hitting API rate limits
            time.sleep(1)
        
        return weather_data

    def save_to_csv(self, weather_data, filename=None):
        """
        Save weather data to CSV file
        
        Args:
            weather_data (list): List of weather data dictionaries
            filename (str): Optional filename, defaults to current date
            
        Returns:
            str: Path to the saved CSV file
        """
        if not weather_data:
            logging.warning("No weather data to save")
            return None
        
        # Create DataFrame from weather data
        df = pd.DataFrame(weather_data)
        
        # Generate filename based on current date if not provided
        if not filename:
            current_date = datetime.now().strftime("%Y%m%d")
            filename = f"india_weather_{current_date}.csv"
        
        # Save to CSV
        csv_path = os.path.join(DATA_DIR, filename)
        
        # If file exists, append without headers
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode='a', header=False, index=False)
            logging.info(f"Appended data to {csv_path}")
        else:
            df.to_csv(csv_path, index=False)
            logging.info(f"Created new file {csv_path}")
            
        return csv_path
    
    def append_to_master_csv(self, weather_data):
        """
        Append weather data to master CSV file containing all data
        
        Args:
            weather_data (list): List of weather data dictionaries
            
        Returns:
            str: Path to the master CSV file
        """
        master_csv = os.path.join(DATA_DIR, "india_weather_master.csv")
        df = pd.DataFrame(weather_data)
        
        if os.path.exists(master_csv):
            df.to_csv(master_csv, mode='a', header=False, index=False)
        else:
            df.to_csv(master_csv, index=False)
            
        return master_csv

def collect_weather_data(api_key):
    """
    Run a single collection of weather data for all cities
    
    Args:
        api_key (str): OpenWeatherMap API key
        
    Returns:
        tuple: (data, csv_path)
    """
    collector = WeatherDataCollector(api_key)
    weather_data = collector.collect_data_for_cities()
    
    if weather_data:
        # Save to date-specific CSV
        csv_path = collector.save_to_csv(weather_data)
        
        # Append to master CSV
        master_csv = collector.append_to_master_csv(weather_data)
        
        logging.info(f"Weather data collection completed. Saved to {csv_path} and {master_csv}")
        return weather_data, csv_path
    else:
        logging.error("Failed to collect weather data")
        return None, None

def collect_historical_data(api_key, days_back=30):
    """
    Retrieve historical weather data for the past X days
    Note: This requires a paid API subscription to OpenWeatherMap.
    This is provided as a reference but might not work with a free API key.
    
    Args:
        api_key (str): OpenWeatherMap API key
        days_back (int): Number of days of historical data to retrieve
        
    Returns:
        dict: Historical data by city
    """
    collector = WeatherDataCollector(api_key)
    historical_url = "https://api.openweathermap.org/data/2.5/onecall/timemachine"
    
    historical_data = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    # Loop through each day
    for day_offset in range(days_back):
        target_date = start_date + timedelta(days=day_offset)
        unix_time = int(target_date.timestamp())
        
        for city in INDIAN_CITIES:
            params = {
                "lat": city['lat'],
                "lon": city['lon'],
                "dt": unix_time,
                "appid": api_key,
                "units": "metric"
            }
            
            try:
                response = requests.get(historical_url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    for hourly_data in data.get("hourly", []):
                        weather_info = {
                            "city": city['name'],
                            "collection_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "timestamp": hourly_data.get("dt", 0),
                            "date": datetime.fromtimestamp(hourly_data.get("dt", 0)).strftime("%Y-%m-%d"),
                            "hour": datetime.fromtimestamp(hourly_data.get("dt", 0)).strftime("%H"),
                            "temp": hourly_data.get("temp"),
                            "feels_like": hourly_data.get("feels_like"),
                            "humidity": hourly_data.get("humidity"),
                            "pressure": hourly_data.get("pressure"),
                            "wind_speed": hourly_data.get("wind_speed"),
                            "wind_direction": hourly_data.get("wind_deg"),
                            "weather_main": hourly_data.get("weather", [{}])[0].get("main"),
                            "weather_description": hourly_data.get("weather", [{}])[0].get("description")
                        }
                        historical_data.append(weather_info)
                else:
                    logging.warning(f"Failed to get historical data for {city['name']} on {target_date.date()}: {response.status_code}")
            except Exception as e:
                logging.error(f"Error fetching historical data: {e}")
            
            # Add delay to avoid rate limiting
            time.sleep(2)
    
    # Save historical data if we have any
    if historical_data:
        df = pd.DataFrame(historical_data)
        historical_csv = os.path.join(DATA_DIR, "india_weather_historical.csv")
        df.to_csv(historical_csv, index=False)
        logging.info(f"Historical data saved to {historical_csv}")
    
    return historical_data

def setup_scheduler(api_key, interval_hours=6):
    """
    Set up a simple scheduler to collect data at regular intervals
    For a production scenario, consider using more robust schedulers like cron or Airflow
    
    Args:
        api_key (str): OpenWeatherMap API key
        interval_hours (int): Collection interval in hours
        
    Returns:
        None
    """
    logging.info(f"Starting weather data collection scheduler (interval: {interval_hours} hours)")
    
    interval_seconds = interval_hours * 60 * 60
    
    try:
        while True:
            # Collect and save weather data
            collect_weather_data(api_key)
            
            # Wait for the next collection interval
            next_collection = datetime.now() + timedelta(seconds=interval_seconds)
            logging.info(f"Next collection scheduled for: {next_collection}")
            
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        logging.info("Scheduler stopped by user")
    except Exception as e:
        logging.error(f"Scheduler error: {e}")

def main():
    """Main function to run the weather data collector"""
    parser = argparse.ArgumentParser(description="IndiaClimateWatch Weather Data Collector")
    parser.add_argument("--api-key", required=True, help="OpenWeatherMap API key")
    parser.add_argument("--schedule", action="store_true", help="Run as a scheduled task")
    parser.add_argument("--interval", type=int, default=6, help="Collection interval in hours (default: 6)")
    parser.add_argument("--historical", action="store_true", help="Attempt to collect historical data (requires paid API)")
    parser.add_argument("--days", type=int, default=30, help="Days of historical data to collect (default: 30)")
    
    args = parser.parse_args()
    
    # Validate API key
    if not args.api_key:
        logging.error("API key is required. Get one from https://openweathermap.org/api")
        return
    
    # Print collection information
    logging.info("IndiaClimateWatch - Weather Data Collector")
    logging.info(f"Collecting data for {len(INDIAN_CITIES)} Indian cities")
    
    if args.historical:
        logging.info(f"Collecting historical data for the past {args.days} days")
        collect_historical_data(args.api_key, args.days)
    
    if args.schedule:
        logging.info(f"Running as scheduled task with {args.interval}-hour intervals")
        setup_scheduler(args.api_key, args.interval)
    else:
        # Run a single collection
        collect_weather_data(args.api_key)

if __name__ == "__main__":
    main()