import os
import requests
import time
from tqdm import tqdm
import pandas as pd

# Create directory for storing flags
os.makedirs('data/flags', exist_ok=True)

# List of commonly used country codes
COUNTRIES = [
    {'code': 'af', 'name': 'Afghanistan'},
    {'code': 'al', 'name': 'Albania'},
    {'code': 'dz', 'name': 'Algeria'},
    {'code': 'ad', 'name': 'Andorra'},
    {'code': 'ao', 'name': 'Angola'},
    {'code': 'ag', 'name': 'Antigua and Barbuda'},
    {'code': 'ar', 'name': 'Argentina'},
    {'code': 'am', 'name': 'Armenia'},
    {'code': 'au', 'name': 'Australia'},
    {'code': 'at', 'name': 'Austria'},
    {'code': 'az', 'name': 'Azerbaijan'},
    {'code': 'bs', 'name': 'Bahamas'},
    {'code': 'bh', 'name': 'Bahrain'},
    {'code': 'bd', 'name': 'Bangladesh'},
    {'code': 'bb', 'name': 'Barbados'},
    {'code': 'by', 'name': 'Belarus'},
    {'code': 'be', 'name': 'Belgium'},
    {'code': 'bz', 'name': 'Belize'},
    {'code': 'bj', 'name': 'Benin'},
    {'code': 'bt', 'name': 'Bhutan'},
    {'code': 'bo', 'name': 'Bolivia'},
    {'code': 'ba', 'name': 'Bosnia and Herzegovina'},
    {'code': 'bw', 'name': 'Botswana'},
    {'code': 'br', 'name': 'Brazil'},
    {'code': 'bn', 'name': 'Brunei'},
    {'code': 'bg', 'name': 'Bulgaria'},
    {'code': 'bf', 'name': 'Burkina Faso'},
    {'code': 'bi', 'name': 'Burundi'},
    {'code': 'kh', 'name': 'Cambodia'},
    {'code': 'cm', 'name': 'Cameroon'},
    {'code': 'ca', 'name': 'Canada'},
    {'code': 'cv', 'name': 'Cape Verde'},
    {'code': 'cf', 'name': 'Central African Republic'},
    {'code': 'td', 'name': 'Chad'},
    {'code': 'cl', 'name': 'Chile'},
    {'code': 'cn', 'name': 'China'},
    {'code': 'co', 'name': 'Colombia'},
    {'code': 'km', 'name': 'Comoros'},
    {'code': 'cg', 'name': 'Republic of the Congo'},
    {'code': 'cd', 'name': 'Democratic Republic of the Congo'},
    {'code': 'cr', 'name': 'Costa Rica'},
    {'code': 'ci', 'name': 'Ivory Coast'},
    {'code': 'hr', 'name': 'Croatia'},
    {'code': 'cu', 'name': 'Cuba'},
    {'code': 'cy', 'name': 'Cyprus'},
    {'code': 'cz', 'name': 'Czech Republic'},
    {'code': 'dk', 'name': 'Denmark'},
    {'code': 'dj', 'name': 'Djibouti'},
    {'code': 'dm', 'name': 'Dominica'},
    {'code': 'do', 'name': 'Dominican Republic'},
    {'code': 'ec', 'name': 'Ecuador'},
    {'code': 'eg', 'name': 'Egypt'},
    {'code': 'sv', 'name': 'El Salvador'},
    {'code': 'gq', 'name': 'Equatorial Guinea'},
    {'code': 'er', 'name': 'Eritrea'},
    {'code': 'ee', 'name': 'Estonia'},
    {'code': 'et', 'name': 'Ethiopia'},
    {'code': 'fj', 'name': 'Fiji'},
    {'code': 'fi', 'name': 'Finland'},
    {'code': 'fr', 'name': 'France'},
    {'code': 'ga', 'name': 'Gabon'},
    {'code': 'gm', 'name': 'Gambia'},
    {'code': 'ge', 'name': 'Georgia'},
    {'code': 'de', 'name': 'Germany'},
    {'code': 'gh', 'name': 'Ghana'},
    {'code': 'gr', 'name': 'Greece'},
    {'code': 'gd', 'name': 'Grenada'},
    {'code': 'gt', 'name': 'Guatemala'},
    {'code': 'gn', 'name': 'Guinea'},
    {'code': 'gw', 'name': 'Guinea-Bissau'},
    {'code': 'gy', 'name': 'Guyana'},
    {'code': 'ht', 'name': 'Haiti'},
    {'code': 'hn', 'name': 'Honduras'},
    {'code': 'hu', 'name': 'Hungary'},
    {'code': 'is', 'name': 'Iceland'},
    {'code': 'in', 'name': 'India'},
    {'code': 'id', 'name': 'Indonesia'},
    {'code': 'ir', 'name': 'Iran'},
    {'code': 'iq', 'name': 'Iraq'},
    {'code': 'ie', 'name': 'Ireland'},
    {'code': 'il', 'name': 'Israel'},
    {'code': 'it', 'name': 'Italy'},
    {'code': 'jm', 'name': 'Jamaica'},
    {'code': 'jp', 'name': 'Japan'},
    {'code': 'jo', 'name': 'Jordan'},
    {'code': 'kz', 'name': 'Kazakhstan'},
    {'code': 'ke', 'name': 'Kenya'},
    {'code': 'ki', 'name': 'Kiribati'},
    {'code': 'kp', 'name': 'North Korea'},
    {'code': 'kr', 'name': 'South Korea'},
    {'code': 'kw', 'name': 'Kuwait'},
    {'code': 'kg', 'name': 'Kyrgyzstan'},
    {'code': 'la', 'name': 'Laos'},
    {'code': 'lv', 'name': 'Latvia'},
    {'code': 'lb', 'name': 'Lebanon'},
    {'code': 'ls', 'name': 'Lesotho'},
    {'code': 'lr', 'name': 'Liberia'},
    {'code': 'ly', 'name': 'Libya'},
    {'code': 'li', 'name': 'Liechtenstein'},
    {'code': 'lt', 'name': 'Lithuania'},
    {'code': 'lu', 'name': 'Luxembourg'},
    {'code': 'mk', 'name': 'North Macedonia'},
    {'code': 'mg', 'name': 'Madagascar'},
    {'code': 'mw', 'name': 'Malawi'},
    {'code': 'my', 'name': 'Malaysia'},
    {'code': 'mv', 'name': 'Maldives'},
    {'code': 'ml', 'name': 'Mali'},
    {'code': 'mt', 'name': 'Malta'},
    {'code': 'mh', 'name': 'Marshall Islands'},
    {'code': 'mr', 'name': 'Mauritania'},
    {'code': 'mu', 'name': 'Mauritius'},
    {'code': 'mx', 'name': 'Mexico'},
    {'code': 'fm', 'name': 'Micronesia'},
    {'code': 'md', 'name': 'Moldova'},
    {'code': 'mc', 'name': 'Monaco'},
    {'code': 'mn', 'name': 'Mongolia'},
    {'code': 'me', 'name': 'Montenegro'},
    {'code': 'ma', 'name': 'Morocco'},
    {'code': 'mz', 'name': 'Mozambique'},
    {'code': 'mm', 'name': 'Myanmar'},
    {'code': 'na', 'name': 'Namibia'},
    {'code': 'nr', 'name': 'Nauru'},
    {'code': 'np', 'name': 'Nepal'},
    {'code': 'nl', 'name': 'Netherlands'},
    {'code': 'nz', 'name': 'New Zealand'},
    {'code': 'ni', 'name': 'Nicaragua'},
    {'code': 'ne', 'name': 'Niger'},
    {'code': 'ng', 'name': 'Nigeria'},
    {'code': 'no', 'name': 'Norway'},
    {'code': 'om', 'name': 'Oman'},
    {'code': 'pk', 'name': 'Pakistan'},
    {'code': 'pw', 'name': 'Palau'},
    {'code': 'pa', 'name': 'Panama'},
    {'code': 'pg', 'name': 'Papua New Guinea'},
    {'code': 'py', 'name': 'Paraguay'},
    {'code': 'pe', 'name': 'Peru'},
    {'code': 'ph', 'name': 'Philippines'},
    {'code': 'pl', 'name': 'Poland'},
    {'code': 'pt', 'name': 'Portugal'},
    {'code': 'qa', 'name': 'Qatar'},
    {'code': 'ro', 'name': 'Romania'},
    {'code': 'ru', 'name': 'Russia'},
    {'code': 'rw', 'name': 'Rwanda'},
    {'code': 'kn', 'name': 'Saint Kitts and Nevis'},
    {'code': 'lc', 'name': 'Saint Lucia'},
    {'code': 'vc', 'name': 'Saint Vincent and the Grenadines'},
    {'code': 'ws', 'name': 'Samoa'},
    {'code': 'sm', 'name': 'San Marino'},
    {'code': 'st', 'name': 'Sao Tome and Principe'},
    {'code': 'sa', 'name': 'Saudi Arabia'},
    {'code': 'sn', 'name': 'Senegal'},
    {'code': 'rs', 'name': 'Serbia'},
    {'code': 'sc', 'name': 'Seychelles'},
    {'code': 'sl', 'name': 'Sierra Leone'},
    {'code': 'sg', 'name': 'Singapore'},
    {'code': 'sk', 'name': 'Slovakia'},
    {'code': 'si', 'name': 'Slovenia'},
    {'code': 'sb', 'name': 'Solomon Islands'},
    {'code': 'so', 'name': 'Somalia'},
    {'code': 'za', 'name': 'South Africa'},
    {'code': 'ss', 'name': 'South Sudan'},
    {'code': 'es', 'name': 'Spain'},
    {'code': 'lk', 'name': 'Sri Lanka'},
    {'code': 'sd', 'name': 'Sudan'},
    {'code': 'sr', 'name': 'Suriname'},
    {'code': 'sz', 'name': 'Eswatini'},
    {'code': 'se', 'name': 'Sweden'},
    {'code': 'ch', 'name': 'Switzerland'},
    {'code': 'sy', 'name': 'Syria'},
    {'code': 'tw', 'name': 'Taiwan'},
    {'code': 'tj', 'name': 'Tajikistan'},
    {'code': 'tz', 'name': 'Tanzania'},
    {'code': 'th', 'name': 'Thailand'},
    {'code': 'tl', 'name': 'East Timor'},
    {'code': 'tg', 'name': 'Togo'},
    {'code': 'to', 'name': 'Tonga'},
    {'code': 'tt', 'name': 'Trinidad and Tobago'},
    {'code': 'tn', 'name': 'Tunisia'},
    {'code': 'tr', 'name': 'Turkey'},
    {'code': 'tm', 'name': 'Turkmenistan'},
    {'code': 'tv', 'name': 'Tuvalu'},
    {'code': 'ug', 'name': 'Uganda'},
    {'code': 'ua', 'name': 'Ukraine'},
    {'code': 'ae', 'name': 'United Arab Emirates'},
    {'code': 'gb', 'name': 'United Kingdom'},
    {'code': 'us', 'name': 'United States'},
    {'code': 'uy', 'name': 'Uruguay'},
    {'code': 'uz', 'name': 'Uzbekistan'},
    {'code': 'vu', 'name': 'Vanuatu'},
    {'code': 'va', 'name': 'Vatican City'},
    {'code': 've', 'name': 'Venezuela'},
    {'code': 'vn', 'name': 'Vietnam'},
    {'code': 'ye', 'name': 'Yemen'},
    {'code': 'zm', 'name': 'Zambia'},
    {'code': 'zw', 'name': 'Zimbabwe'}
]

def download_file(url, save_path, min_size_kb=5):
    """Download a file from URL and save it to the specified path"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, stream=True, timeout=20)
        
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Check file size to make sure it's not an error page
            file_size_kb = os.path.getsize(save_path) / 1024
            
            print(f"Downloaded flag ({file_size_kb:.1f}KB)")
            return True
        else:
            print(f"HTTP error {response.status_code}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        if os.path.exists(save_path):
            os.remove(save_path)
        return False

def download_flag_from_flagcdn(country_code, country_name):
    """Download flag from FlagCDN - a highly reliable source"""
    save_path = f"data/flags/{country_code}.png"
    
    # Check if we already have this flag
    if os.path.exists(save_path) and os.path.getsize(save_path) > 5000:
        print(f"Flag for {country_name} already exists, skipping")
        return True
    
    # FlagCDN has a consistent URL pattern and high-quality images
    url = f"https://flagcdn.com/w2560/{country_code.lower()}.png"
    print(f"Trying FlagCDN: {url}")
    
    return download_file(url, save_path, min_size_kb=5)

def download_flag_from_flagpedia(country_code, country_name):
    """Download flag from Flagpedia as backup"""
    save_path = f"data/flags/{country_code}.png"
    
    # Check if we already have this flag
    if os.path.exists(save_path) and os.path.getsize(save_path) > 5000:
        print(f"Flag for {country_name} already exists, skipping")
        return True
    
    # Flagpedia has larger resolution flags at w1160
    url = f"https://flagpedia.net/data/flags/w1160/{country_code.lower()}.png"
    print(f"Trying Flagpedia: {url}")
    
    return download_file(url, save_path, min_size_kb=5)

def download_flags():
    """Download flags for all countries using reliable sources"""
    print("Starting flag download process...")
    
    successful = 0
    failed = 0
    
    # Process countries
    for country in tqdm(COUNTRIES, desc="Downloading flags"):
        country_code = country['code']
        country_name = country['name']
        
        print(f"\nProcessing {country_name} ({country_code})...")
        
        # Try FlagCDN first (most reliable)
        if download_flag_from_flagcdn(country_code, country_name):
            successful += 1
            continue
        
        # Try Flagpedia as backup
        if download_flag_from_flagpedia(country_code, country_name):
            successful += 1
            continue
        
        # If both fail
        print(f"Could not download flag for {country_name}")
        failed += 1
        
        # Short pause between countries to be nice to servers
        time.sleep(0.5)
    
    # Count actual flag files
    flag_count = len([f for f in os.listdir('data/flags') if f.endswith('.png')])
    
    print("\nFlag download summary:")
    print(f"Total countries processed: {len(COUNTRIES)}")
    print(f"Successfully downloaded: {successful}")
    print(f"Failed downloads: {failed}")
    print(f"Total flags in folder: {flag_count}")
    
    # Save results to CSV
    results = []
    for country in COUNTRIES:
        code = country['code']
        name = country['name']
        flag_path = f"data/flags/{code}.png"
        
        has_flag = os.path.exists(flag_path)
        file_size = os.path.getsize(flag_path) if has_flag else 0
        
        results.append({
            'country_code': code,
            'country_name': name,
            'has_flag': has_flag,
            'file_size_bytes': file_size
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('data/flag_download_results.csv', index=False)
    print("Results saved to data/flag_download_results.csv")
    
    return flag_count

if __name__ == "__main__":
    download_flags()