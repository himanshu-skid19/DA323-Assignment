import os
import json
import time
import random
import requests
import subprocess
import datetime
import argparse
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

# Directory structure
BASE_DIR = 'radiowave_dataset'
AUDIO_DIR = os.path.join(BASE_DIR, 'audio')
METADATA_DIR = os.path.join(BASE_DIR, 'metadata')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Create directories if they don't exist
for directory in [BASE_DIR, AUDIO_DIR, METADATA_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'radio_collector.log')),
        logging.StreamHandler()
    ]
)

def get_radio_browser_stations(limit=100, has_extended_info=True):
    """
    Fetch radio stations from the Radio Browser API.
    
    Args:
        limit (int): Maximum number of stations to retrieve
        has_extended_info (bool): Whether to only include stations with extended info
        
    Returns:
        list: List of radio station information dictionaries
    """
    logging.info(f"Fetching {limit} radio stations from Radio Browser API")
    
    # Radio Browser API endpoints have multiple servers for load balancing
    # First, get a random API endpoint
    try:
        dns_response = requests.get('https://all.api.radio-browser.info/json/servers')
        if dns_response.status_code != 200:
            logging.error(f"Failed to get Radio Browser servers: {dns_response.status_code}")
            return []
            
        servers = dns_response.json()
        if not servers:
            logging.error("No Radio Browser servers found")
            return []
            
        # Select a random server
        server = random.choice(servers)
        base_url = f"https://{server['name']}/json/stations"
        
        # Parameters for the API request
        params = {
            'limit': limit,
            'hidebroken': 'true',        # Hide broken stations
            'has_extended_info': 'true' if has_extended_info else 'false',
            'order': 'random',           # Get random stations
            'codec': 'MP3'               # Only MP3 streams for compatibility
        }
        
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            logging.error(f"Failed to get stations: {response.status_code}")
            return []
            
        stations = response.json()
        logging.info(f"Successfully retrieved {len(stations)} stations")
        return stations
        
    except Exception as e:
        logging.error(f"Error fetching radio stations: {e}")
        return []

def get_icecast_directory_stations(limit=100):
    """
    Fetch radio stations from the Icecast directory.
    
    Args:
        limit (int): Maximum number of stations to retrieve
        
    Returns:
        list: List of radio station information dictionaries
    """
    logging.info(f"Fetching Icecast directory stations")
    
    try:
        # Icecast directory URL
        url = "https://dir.xiph.org/yp.xml"
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            logging.error(f"Failed to get Icecast directory: {response.status_code}")
            return []
        
        # Parse XML response
        root = ET.fromstring(response.text)
        stations = []
        
        for entry in root.findall('./entry')[:limit]:
            station = {
                'name': entry.findtext('server_name', 'Unknown'),
                'url': entry.findtext('listen_url', ''),
                'bitrate': entry.findtext('bitrate', '0'),
                'genre': entry.findtext('genre', 'Unknown'),
                'codec': entry.findtext('server_type', 'Unknown'),
                'country': 'Unknown',  # Icecast doesn't provide country info
                'tags': entry.findtext('genre', '').split(),
                'homepage': entry.findtext('server_url', '')
            }
            
            if station['url']:  # Only add if we have a URL
                stations.append(station)
        
        logging.info(f"Successfully retrieved {len(stations)} Icecast stations")
        return stations
        
    except Exception as e:
        logging.error(f"Error fetching Icecast stations: {e}")
        return []

def get_hardcoded_stations():
    """
    Return a list of hardcoded reliable radio station streams.
    This serves as a fallback if the API methods fail.
    
    Returns:
        list: List of radio station information dictionaries
    """
    stations = [
        {
            "name": "NPR Program Stream",
            "url": "https://npr-ice.streamguys1.com/live.mp3",
            "codec": "MP3",
            "bitrate": "128",
            "country": "USA",
            "tags": ["news", "talk", "public radio"],
            "homepage": "https://www.npr.org/"
        },
        {
            "name": "BBC World Service",
            "url": "https://stream.live.vc.bbcmedia.co.uk/bbc_world_service",
            "codec": "MP3",
            "bitrate": "96",
            "country": "UK",
            "tags": ["news", "talk", "international"],
            "homepage": "https://www.bbc.co.uk/worldserviceradio"
        },
        {
            "name": "KEXP Seattle",
            "url": "https://kexp.streamguys1.com/kexp160.aac",
            "codec": "AAC",
            "bitrate": "160",
            "country": "USA",
            "tags": ["alternative", "indie", "variety"],
            "homepage": "https://www.kexp.org/"
        },
        {
            "name": "Classic FM UK",
            "url": "https://media-ice.musicradio.com/ClassicFMMP3",
            "codec": "MP3",
            "bitrate": "128",
            "country": "UK",
            "tags": ["classical", "orchestral"],
            "homepage": "https://www.classicfm.com/"
        },
        {
            "name": "WNYC Public Radio",
            "url": "https://stream.wqxr.org/wqxr-web",
            "codec": "MP3",
            "bitrate": "128",
            "country": "USA",
            "tags": ["classical", "public radio"],
            "homepage": "https://www.wqxr.org/"
        }
    ]
    
    logging.info(f"Using {len(stations)} hardcoded reliable stations")
    return stations

def validate_station(station):
    """
    Check if a station has all the required information and a valid URL.
    
    Args:
        station (dict): Station information
        
    Returns:
        bool: True if the station is valid, False otherwise
    """
    # Check if the required fields are present
    required_fields = ['name', 'url']
    if not all(field in station and station[field] for field in required_fields):
        return False
    
    # Check if the URL is valid
    try:
        result = urlparse(station['url'])
        if not all([result.scheme, result.netloc]):
            return False
            
        # Check if URL ends with common audio stream extensions
        audio_extensions = ['.mp3', '.aac', '.ogg', '.m3u', '.pls']
        if not any(station['url'].lower().endswith(ext) for ext in audio_extensions) and 'stream' not in station['url'].lower():
            # Try a HEAD request to check content type
            try:
                headers = requests.head(station['url'], timeout=5).headers
                content_type = headers.get('Content-Type', '')
                if not any(media_type in content_type.lower() for media_type in ['audio', 'mpegurl', 'playlist']):
                    return False
            except:
                # If HEAD request fails, we'll still try the station
                pass
                
        return True
    except:
        return False

def select_stations(count=30):
    """
    Select a diverse set of radio stations to record.
    
    Args:
        count (int): Number of stations to select
        
    Returns:
        list: Selected radio station information
    """
    # First try Radio Browser API
    stations = get_radio_browser_stations(limit=150)
    
    # If not enough stations, try Icecast directory
    if len(stations) < count:
        icecast_stations = get_icecast_directory_stations(limit=100)
        stations.extend(icecast_stations)
    
    # Add hardcoded stations as a fallback
    if len(stations) < count:
        hardcoded_stations = get_hardcoded_stations()
        stations.extend(hardcoded_stations)
    
    # Filter out invalid stations
    valid_stations = [station for station in stations if validate_station(station)]
    logging.info(f"Found {len(valid_stations)} valid stations out of {len(stations)}")
    
    # If we have more stations than needed, select a diverse set
    if len(valid_stations) > count:
        # Try to get a diverse set of stations by country and genre
        selected = []
        countries = {}
        genres = {}
        
        for station in valid_stations:
            country = station.get('country', 'Unknown')
            genre = station.get('tags', ['Unknown'])[0] if station.get('tags') else 'Unknown'
            
            # Only add if we don't have too many from this country/genre
            if countries.get(country, 0) < count/5 and genres.get(genre, 0) < count/5:
                selected.append(station)
                countries[country] = countries.get(country, 0) + 1
                genres[genre] = genres.get(genre, 0) + 1
                
                if len(selected) >= count:
                    break
        
        # If we couldn't get enough diverse stations, just take the first ones
        if len(selected) < count:
            remaining = count - len(selected)
            for station in valid_stations:
                if station not in selected:
                    selected.append(station)
                    remaining -= 1
                    if remaining <= 0:
                        break
                        
        return selected
    else:
        # If we don't have enough stations, return all valid ones
        return valid_stations

def record_station(station, duration=60, output_format='mp3'):
    """
    Record audio from a radio station for the specified duration.
    
    Args:
        station (dict): Radio station information
        duration (int): Recording duration in seconds
        output_format (str): Output file format ('mp3' or 'wav')
        
    Returns:
        tuple: (success, filename, metadata)
    """
    # Create a filename based on station name and timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_name = ''.join(c if c.isalnum() else '_' for c in station['name'])
    filename = f"{safe_name}_{timestamp}.{output_format}"
    filepath = os.path.join(AUDIO_DIR, filename)
    
    logging.info(f"Recording {station['name']} for {duration} seconds to {filename}")
    
    # Build the ffmpeg command
    try:
        # Use ffmpeg to record the stream
        command = [
            'ffmpeg',
            '-y',                        # Overwrite output file if it exists
            '-i', station['url'],        # Input URL
            '-t', str(duration),         # Duration
            '-c:a', 'copy' if output_format == 'mp3' else 'pcm_s16le',  # Codec (copy for mp3, PCM for wav)
            filepath                     # Output file
        ]
        
        # Run ffmpeg
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for the process to complete
        stdout, stderr = process.communicate()
        
        # Check if the file was created and has content
        if process.returncode != 0 or not os.path.exists(filepath) or os.path.getsize(filepath) < 1000:
            logging.error(f"Failed to record {station['name']}: {stderr.decode()}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return False, None, None
            
        # Create metadata
        metadata = {
            'station_name': station['name'],
            'stream_url': station['url'],
            'timestamp': datetime.datetime.now().isoformat(),
            'duration': duration,
            'format': output_format,
            'codec': station.get('codec', 'Unknown'),
            'bitrate': station.get('bitrate', 'Unknown'),
            'country': station.get('country', 'Unknown'),
            'tags': station.get('tags', []),
            'homepage': station.get('homepage', '')
        }
        
        # Save metadata
        metadata_filename = f"{os.path.splitext(filename)[0]}.json"
        metadata_filepath = os.path.join(METADATA_DIR, metadata_filename)
        
        with open(metadata_filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logging.info(f"Successfully recorded {station['name']} to {filename}")
        return True, filepath, metadata
        
    except Exception as e:
        logging.error(f"Error recording {station['name']}: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return False, None, None

def build_dataset(num_recordings=30, min_duration=30, max_duration=90, format='mp3'):
    """
    Build the complete RadioWave dataset.
    
    Args:
        num_recordings (int): Number of recordings to make
        min_duration (int): Minimum recording duration in seconds
        max_duration (int): Maximum recording duration in seconds
        format (str): Output file format ('mp3' or 'wav')
        
    Returns:
        list: List of successful recordings with their metadata
    """
    logging.info(f"Building RadioWave dataset with {num_recordings} recordings")
    
    # Select stations
    stations = select_stations(count=num_recordings*2)  # Get more than needed in case some fail
    if len(stations) < num_recordings:
        logging.warning(f"Only found {len(stations)} valid stations, less than the requested {num_recordings}")
    
    # Shuffle the stations
    random.shuffle(stations)
    
    # Record each station
    recordings = []
    for i, station in enumerate(stations):
        if len(recordings) >= num_recordings:
            break
            
        # Randomly select a duration within the specified range
        duration = random.randint(min_duration, max_duration)
        
        # Record the station
        success, filepath, metadata = record_station(station, duration, format)
        
        if success:
            recordings.append({'filepath': filepath, 'metadata': metadata})
            logging.info(f"Recording {len(recordings)}/{num_recordings} completed")
        else:
            logging.warning(f"Failed to record {station['name']}, trying next station")
        
        # Add a small delay between recordings to avoid hitting rate limits
        time.sleep(2)
    
    logging.info(f"Completed dataset with {len(recordings)} recordings")
    
    # Create a dataset summary
    summary = {
        'name': 'RadioWave Dataset',
        'description': 'A collection of AM/FM radio station recordings',
        'created_date': datetime.datetime.now().isoformat(),
        'num_recordings': len(recordings),
        'format': format,
        'recordings': [r['metadata'] for r in recordings]
    }
    
    # Save the dataset summary
    with open(os.path.join(BASE_DIR, 'dataset_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    return recordings

def check_ffmpeg():
    """
    Check if ffmpeg is installed and available.
    
    Returns:
        bool: True if ffmpeg is available, False otherwise
    """
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False

def main():
    """Main function to run the RadioWave dataset collection process."""
    parser = argparse.ArgumentParser(description='RadioWave Dataset Collector')
    parser.add_argument('--count', type=int, default=30, help='Number of recordings to collect')
    parser.add_argument('--min-duration', type=int, default=30, help='Minimum recording duration in seconds')
    parser.add_argument('--max-duration', type=int, default=90, help='Maximum recording duration in seconds')
    parser.add_argument('--format', choices=['mp3', 'wav'], default='mp3', help='Audio format')
    
    args = parser.parse_args()
    
    # Check if ffmpeg is installed
    if not check_ffmpeg():
        logging.error("ffmpeg is not installed or not in PATH. Please install ffmpeg and try again.")
        return
    
    # Build the dataset
    recordings = build_dataset(
        num_recordings=args.count,
        min_duration=args.min_duration, 
        max_duration=args.max_duration,
        format=args.format
    )
    
    if recordings:
        logging.info(f"Successfully created RadioWave dataset with {len(recordings)} recordings")
        logging.info(f"Dataset is stored in {os.path.abspath(BASE_DIR)}")
    else:
        logging.error("Failed to create RadioWave dataset")

if __name__ == '__main__':
    main()