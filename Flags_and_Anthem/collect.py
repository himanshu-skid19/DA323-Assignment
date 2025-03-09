import os
import requests
import time
import random
from tqdm import tqdm
import pandas as pd
import urllib.parse
import shutil
from bs4 import BeautifulSoup
import re

# Create directories for storing data
os.makedirs('data/flags', exist_ok=True)
os.makedirs('data/anthems/text', exist_ok=True)
os.makedirs('data/anthems/audio', exist_ok=True)

# Complete list of country codes and names
COUNTRIES = [
    {'code': 'af', 'name': 'Afghanistan', 'alt_codes': ['afghan']},
    {'code': 'al', 'name': 'Albania', 'alt_codes': ['albania']},
    {'code': 'dz', 'name': 'Algeria', 'alt_codes': ['algeria']},
    {'code': 'ad', 'name': 'Andorra', 'alt_codes': ['andorra']},
    {'code': 'ao', 'name': 'Angola', 'alt_codes': ['angola']},
    {'code': 'ag', 'name': 'Antigua and Barbuda', 'alt_codes': ['antigua']},
    {'code': 'ar', 'name': 'Argentina', 'alt_codes': ['argentina']},
    {'code': 'am', 'name': 'Armenia', 'alt_codes': ['armenia']},
    {'code': 'au', 'name': 'Australia', 'alt_codes': ['australia']},
    {'code': 'at', 'name': 'Austria', 'alt_codes': ['austria']},
    {'code': 'az', 'name': 'Azerbaijan', 'alt_codes': ['azerbaijan']},
    {'code': 'bs', 'name': 'Bahamas', 'alt_codes': ['bahamas']},
    {'code': 'bh', 'name': 'Bahrain', 'alt_codes': ['bahrain']},
    {'code': 'bd', 'name': 'Bangladesh', 'alt_codes': ['bangladesh']},
    {'code': 'bb', 'name': 'Barbados', 'alt_codes': ['barbados']},
    {'code': 'by', 'name': 'Belarus', 'alt_codes': ['belarus']},
    {'code': 'be', 'name': 'Belgium', 'alt_codes': ['belgium']},
    {'code': 'bz', 'name': 'Belize', 'alt_codes': ['belize']},
    {'code': 'bj', 'name': 'Benin', 'alt_codes': ['benin']},
    {'code': 'bt', 'name': 'Bhutan', 'alt_codes': ['bhutan']},
    {'code': 'bo', 'name': 'Bolivia', 'alt_codes': ['bolivia']},
    {'code': 'ba', 'name': 'Bosnia and Herzegovina', 'alt_codes': ['bosnia']},
    {'code': 'bw', 'name': 'Botswana', 'alt_codes': ['botswana']},
    {'code': 'br', 'name': 'Brazil', 'alt_codes': ['brazil']},
    {'code': 'bn', 'name': 'Brunei', 'alt_codes': ['brunei']},
    {'code': 'bg', 'name': 'Bulgaria', 'alt_codes': ['bulgaria']},
    {'code': 'bf', 'name': 'Burkina Faso', 'alt_codes': ['burkina']},
    {'code': 'bi', 'name': 'Burundi', 'alt_codes': ['burundi']},
    {'code': 'kh', 'name': 'Cambodia', 'alt_codes': ['cambodia']},
    {'code': 'cm', 'name': 'Cameroon', 'alt_codes': ['cameroon']},
    {'code': 'ca', 'name': 'Canada', 'alt_codes': ['canada']},
    {'code': 'cv', 'name': 'Cape Verde', 'alt_codes': ['capeverde']},
    {'code': 'cf', 'name': 'Central African Republic', 'alt_codes': ['centralafricanrepublic', 'car']},
    {'code': 'td', 'name': 'Chad', 'alt_codes': ['chad']},
    {'code': 'cl', 'name': 'Chile', 'alt_codes': ['chile']},
    {'code': 'cn', 'name': 'China', 'alt_codes': ['china']},
    {'code': 'co', 'name': 'Colombia', 'alt_codes': ['colombia']},
    {'code': 'km', 'name': 'Comoros', 'alt_codes': ['comoros']},
    {'code': 'cg', 'name': 'Republic of the Congo', 'alt_codes': ['congo']},
    {'code': 'cd', 'name': 'Democratic Republic of the Congo', 'alt_codes': ['drcongo', 'congodr']},
    {'code': 'cr', 'name': 'Costa Rica', 'alt_codes': ['costarica']},
    {'code': 'ci', 'name': 'Ivory Coast', 'alt_codes': ['ivorycoast', 'cotedivoire']},
    {'code': 'hr', 'name': 'Croatia', 'alt_codes': ['croatia']},
    {'code': 'cu', 'name': 'Cuba', 'alt_codes': ['cuba']},
    {'code': 'cy', 'name': 'Cyprus', 'alt_codes': ['cyprus']},
    {'code': 'cz', 'name': 'Czech Republic', 'alt_codes': ['czech', 'czechia']},
    {'code': 'dk', 'name': 'Denmark', 'alt_codes': ['denmark']},
    {'code': 'dj', 'name': 'Djibouti', 'alt_codes': ['djibouti']},
    {'code': 'dm', 'name': 'Dominica', 'alt_codes': ['dominica']},
    {'code': 'do', 'name': 'Dominican Republic', 'alt_codes': ['dominicanrepublic', 'dominican']},
    {'code': 'ec', 'name': 'Ecuador', 'alt_codes': ['ecuador']},
    {'code': 'eg', 'name': 'Egypt', 'alt_codes': ['egypt']},
    {'code': 'sv', 'name': 'El Salvador', 'alt_codes': ['elsalvador']},
    {'code': 'gq', 'name': 'Equatorial Guinea', 'alt_codes': ['equatorialguinea']},
    {'code': 'er', 'name': 'Eritrea', 'alt_codes': ['eritrea']},
    {'code': 'ee', 'name': 'Estonia', 'alt_codes': ['estonia']},
    {'code': 'et', 'name': 'Ethiopia', 'alt_codes': ['ethiopia']},
    {'code': 'fj', 'name': 'Fiji', 'alt_codes': ['fiji']},
    {'code': 'fi', 'name': 'Finland', 'alt_codes': ['finland']},
    {'code': 'fr', 'name': 'France', 'alt_codes': ['france']},
    {'code': 'ga', 'name': 'Gabon', 'alt_codes': ['gabon']},
    {'code': 'gm', 'name': 'Gambia', 'alt_codes': ['gambia']},
    {'code': 'ge', 'name': 'Georgia', 'alt_codes': ['georgia']},
    {'code': 'de', 'name': 'Germany', 'alt_codes': ['germany']},
    {'code': 'gh', 'name': 'Ghana', 'alt_codes': ['ghana']},
    {'code': 'gr', 'name': 'Greece', 'alt_codes': ['greece']},
    {'code': 'gd', 'name': 'Grenada', 'alt_codes': ['grenada']},
    {'code': 'gt', 'name': 'Guatemala', 'alt_codes': ['guatemala']},
    {'code': 'gn', 'name': 'Guinea', 'alt_codes': ['guinea']},
    {'code': 'gw', 'name': 'Guinea-Bissau', 'alt_codes': ['guineabissau']},
    {'code': 'gy', 'name': 'Guyana', 'alt_codes': ['guyana']},
    {'code': 'ht', 'name': 'Haiti', 'alt_codes': ['haiti']},
    {'code': 'hn', 'name': 'Honduras', 'alt_codes': ['honduras']},
    {'code': 'hu', 'name': 'Hungary', 'alt_codes': ['hungary']},
    {'code': 'is', 'name': 'Iceland', 'alt_codes': ['iceland']},
    {'code': 'in', 'name': 'India', 'alt_codes': ['india']},
    {'code': 'id', 'name': 'Indonesia', 'alt_codes': ['indonesia']},
    {'code': 'ir', 'name': 'Iran', 'alt_codes': ['iran']},
    {'code': 'iq', 'name': 'Iraq', 'alt_codes': ['iraq']},
    {'code': 'ie', 'name': 'Ireland', 'alt_codes': ['ireland']},
    {'code': 'il', 'name': 'Israel', 'alt_codes': ['israel']},
    {'code': 'it', 'name': 'Italy', 'alt_codes': ['italy']},
    {'code': 'jm', 'name': 'Jamaica', 'alt_codes': ['jamaica']},
    {'code': 'jp', 'name': 'Japan', 'alt_codes': ['japan']},
    {'code': 'jo', 'name': 'Jordan', 'alt_codes': ['jordan']},
    {'code': 'kz', 'name': 'Kazakhstan', 'alt_codes': ['kazakhstan']},
    {'code': 'ke', 'name': 'Kenya', 'alt_codes': ['kenya']},
    {'code': 'ki', 'name': 'Kiribati', 'alt_codes': ['kiribati']},
    {'code': 'kp', 'name': 'North Korea', 'alt_codes': ['northkorea']},
    {'code': 'kr', 'name': 'South Korea', 'alt_codes': ['southkorea', 'korea']},
    {'code': 'kw', 'name': 'Kuwait', 'alt_codes': ['kuwait']},
    {'code': 'kg', 'name': 'Kyrgyzstan', 'alt_codes': ['kyrgyzstan']},
    {'code': 'la', 'name': 'Laos', 'alt_codes': ['laos']},
    {'code': 'lv', 'name': 'Latvia', 'alt_codes': ['latvia']},
    {'code': 'lb', 'name': 'Lebanon', 'alt_codes': ['lebanon']},
    {'code': 'ls', 'name': 'Lesotho', 'alt_codes': ['lesotho']},
    {'code': 'lr', 'name': 'Liberia', 'alt_codes': ['liberia']},
    {'code': 'ly', 'name': 'Libya', 'alt_codes': ['libya']},
    {'code': 'li', 'name': 'Liechtenstein', 'alt_codes': ['liechtenstein']},
    {'code': 'lt', 'name': 'Lithuania', 'alt_codes': ['lithuania']},
    {'code': 'lu', 'name': 'Luxembourg', 'alt_codes': ['luxembourg']},
    {'code': 'mk', 'name': 'North Macedonia', 'alt_codes': ['macedonia']},
    {'code': 'mg', 'name': 'Madagascar', 'alt_codes': ['madagascar']},
    {'code': 'mw', 'name': 'Malawi', 'alt_codes': ['malawi']},
    {'code': 'my', 'name': 'Malaysia', 'alt_codes': ['malaysia']},
    {'code': 'mv', 'name': 'Maldives', 'alt_codes': ['maldives']},
    {'code': 'ml', 'name': 'Mali', 'alt_codes': ['mali']},
    {'code': 'mt', 'name': 'Malta', 'alt_codes': ['malta']},
    {'code': 'mh', 'name': 'Marshall Islands', 'alt_codes': ['marshallislands']},
    {'code': 'mr', 'name': 'Mauritania', 'alt_codes': ['mauritania']},
    {'code': 'mu', 'name': 'Mauritius', 'alt_codes': ['mauritius']},
    {'code': 'mx', 'name': 'Mexico', 'alt_codes': ['mexico']},
    {'code': 'fm', 'name': 'Micronesia', 'alt_codes': ['micronesia']},
    {'code': 'md', 'name': 'Moldova', 'alt_codes': ['moldova']},
    {'code': 'mc', 'name': 'Monaco', 'alt_codes': ['monaco']},
    {'code': 'mn', 'name': 'Mongolia', 'alt_codes': ['mongolia']},
    {'code': 'me', 'name': 'Montenegro', 'alt_codes': ['montenegro']},
    {'code': 'ma', 'name': 'Morocco', 'alt_codes': ['morocco']},
    {'code': 'mz', 'name': 'Mozambique', 'alt_codes': ['mozambique']},
    {'code': 'mm', 'name': 'Myanmar', 'alt_codes': ['myanmar', 'burma']},
    {'code': 'na', 'name': 'Namibia', 'alt_codes': ['namibia']},
    {'code': 'nr', 'name': 'Nauru', 'alt_codes': ['nauru']},
    {'code': 'np', 'name': 'Nepal', 'alt_codes': ['nepal']},
    {'code': 'nl', 'name': 'Netherlands', 'alt_codes': ['netherlands']},
    {'code': 'nz', 'name': 'New Zealand', 'alt_codes': ['newzealand']},
    {'code': 'ni', 'name': 'Nicaragua', 'alt_codes': ['nicaragua']},
    {'code': 'ne', 'name': 'Niger', 'alt_codes': ['niger']},
    {'code': 'ng', 'name': 'Nigeria', 'alt_codes': ['nigeria']},
    {'code': 'no', 'name': 'Norway', 'alt_codes': ['norway']},
    {'code': 'om', 'name': 'Oman', 'alt_codes': ['oman']},
    {'code': 'pk', 'name': 'Pakistan', 'alt_codes': ['pakistan']},
    {'code': 'pw', 'name': 'Palau', 'alt_codes': ['palau']},
    {'code': 'pa', 'name': 'Panama', 'alt_codes': ['panama']},
    {'code': 'pg', 'name': 'Papua New Guinea', 'alt_codes': ['papuanewguinea']},
    {'code': 'py', 'name': 'Paraguay', 'alt_codes': ['paraguay']},
    {'code': 'pe', 'name': 'Peru', 'alt_codes': ['peru']},
    {'code': 'ph', 'name': 'Philippines', 'alt_codes': ['philippines']},
    {'code': 'pl', 'name': 'Poland', 'alt_codes': ['poland']},
    {'code': 'pt', 'name': 'Portugal', 'alt_codes': ['portugal']},
    {'code': 'qa', 'name': 'Qatar', 'alt_codes': ['qatar']},
    {'code': 'ro', 'name': 'Romania', 'alt_codes': ['romania']},
    {'code': 'ru', 'name': 'Russia', 'alt_codes': ['russia']},
    {'code': 'rw', 'name': 'Rwanda', 'alt_codes': ['rwanda']},
    {'code': 'kn', 'name': 'Saint Kitts and Nevis', 'alt_codes': ['stkitts']},
    {'code': 'lc', 'name': 'Saint Lucia', 'alt_codes': ['stlucia']},
    {'code': 'vc', 'name': 'Saint Vincent and the Grenadines', 'alt_codes': ['stvincent']},
    {'code': 'ws', 'name': 'Samoa', 'alt_codes': ['samoa']},
    {'code': 'sm', 'name': 'San Marino', 'alt_codes': ['sanmarino']},
    {'code': 'st', 'name': 'Sao Tome and Principe', 'alt_codes': ['saotome']},
    {'code': 'sa', 'name': 'Saudi Arabia', 'alt_codes': ['saudiarabia']},
    {'code': 'sn', 'name': 'Senegal', 'alt_codes': ['senegal']},
    {'code': 'rs', 'name': 'Serbia', 'alt_codes': ['serbia']},
    {'code': 'sc', 'name': 'Seychelles', 'alt_codes': ['seychelles']},
    {'code': 'sl', 'name': 'Sierra Leone', 'alt_codes': ['sierraleone']},
    {'code': 'sg', 'name': 'Singapore', 'alt_codes': ['singapore']},
    {'code': 'sk', 'name': 'Slovakia', 'alt_codes': ['slovakia']},
    {'code': 'si', 'name': 'Slovenia', 'alt_codes': ['slovenia']},
    {'code': 'sb', 'name': 'Solomon Islands', 'alt_codes': ['solomonislands']},
    {'code': 'so', 'name': 'Somalia', 'alt_codes': ['somalia']},
    {'code': 'za', 'name': 'South Africa', 'alt_codes': ['southafrica']},
    {'code': 'ss', 'name': 'South Sudan', 'alt_codes': ['southsudan']},
    {'code': 'es', 'name': 'Spain', 'alt_codes': ['spain']},
    {'code': 'lk', 'name': 'Sri Lanka', 'alt_codes': ['srilanka']},
    {'code': 'sd', 'name': 'Sudan', 'alt_codes': ['sudan']},
    {'code': 'sr', 'name': 'Suriname', 'alt_codes': ['suriname']},
    {'code': 'sz', 'name': 'Eswatini', 'alt_codes': ['swaziland', 'eswatini']},
    {'code': 'se', 'name': 'Sweden', 'alt_codes': ['sweden']},
    {'code': 'ch', 'name': 'Switzerland', 'alt_codes': ['switzerland']},
    {'code': 'sy', 'name': 'Syria', 'alt_codes': ['syria']},
    {'code': 'tw', 'name': 'Taiwan', 'alt_codes': ['taiwan']},
    {'code': 'tj', 'name': 'Tajikistan', 'alt_codes': ['tajikistan']},
    {'code': 'tz', 'name': 'Tanzania', 'alt_codes': ['tanzania']},
    {'code': 'th', 'name': 'Thailand', 'alt_codes': ['thailand']},
    {'code': 'tl', 'name': 'East Timor', 'alt_codes': ['easttimor', 'timor']},
    {'code': 'tg', 'name': 'Togo', 'alt_codes': ['togo']},
    {'code': 'to', 'name': 'Tonga', 'alt_codes': ['tonga']},
    {'code': 'tt', 'name': 'Trinidad and Tobago', 'alt_codes': ['trinidad']},
    {'code': 'tn', 'name': 'Tunisia', 'alt_codes': ['tunisia']},
    {'code': 'tr', 'name': 'Turkey', 'alt_codes': ['turkey']},
    {'code': 'tm', 'name': 'Turkmenistan', 'alt_codes': ['turkmenistan']},
    {'code': 'tv', 'name': 'Tuvalu', 'alt_codes': ['tuvalu']},
    {'code': 'ug', 'name': 'Uganda', 'alt_codes': ['uganda']},
    {'code': 'ua', 'name': 'Ukraine', 'alt_codes': ['ukraine']},
    {'code': 'ae', 'name': 'United Arab Emirates', 'alt_codes': ['uae']},
    {'code': 'gb', 'name': 'United Kingdom', 'alt_codes': ['uk', 'britain']},
    {'code': 'us', 'name': 'United States', 'alt_codes': ['usa']},
    {'code': 'uy', 'name': 'Uruguay', 'alt_codes': ['uruguay']},
    {'code': 'uz', 'name': 'Uzbekistan', 'alt_codes': ['uzbekistan']},
    {'code': 'vu', 'name': 'Vanuatu', 'alt_codes': ['vanuatu']},
    {'code': 'va', 'name': 'Vatican City', 'alt_codes': ['vatican']},
    {'code': 've', 'name': 'Venezuela', 'alt_codes': ['venezuela']},
    {'code': 'vn', 'name': 'Vietnam', 'alt_codes': ['vietnam']},
    {'code': 'ye', 'name': 'Yemen', 'alt_codes': ['yemen']},
    {'code': 'zm', 'name': 'Zambia', 'alt_codes': ['zambia']},
    {'code': 'zw', 'name': 'Zimbabwe', 'alt_codes': ['zimbabwe']}
]

# Special additions and common country codes used on nationalanthems.info
SPECIAL_MAPPINGS = {
    'us': ['usa', 'unitedstates'],
    'gb': ['uk', 'britain', 'unitedkingdom'],
    'ru': ['russia', 'russianfederation'],
    'kr': ['korea', 'southkorea', 'republicofkorea'],
    'ae': ['uae', 'unitedarabemirates'],
    'ch': ['switzerland', 'swiss'],
    'cz': ['czech', 'czechia', 'czechrepublic'],
    'do': ['dominican', 'dominicanrepublic'],
    'uk': ['gb', 'britain', 'unitedkingdom'],
    'ba': ['bosnia', 'bosniaandherzegovina'],
    'by': ['belarus', 'belorussia'],
    'ph': ['philippines'],
    'za': ['southafrica'],
    'kr': ['southkorea'],
    'kp': ['northkorea'],
    'ca': ['can', 'canada'],
    'au': ['aus', 'australia'],
    'nz': ['newzealand'],
    'mx': ['mex', 'mexico'],
    'br': ['brazil', 'brasil'],
    'jp': ['japan'],
    'cn': ['china', 'prc'],
    'in': ['india'],
    'pt': ['portugal'],
    'es': ['spain'],
    'tr': ['turkey'],
    'eg': ['egypt'],
    'sa': ['saudiarabia'],
    'ng': ['nigeria'],
    'pk': ['pakistan'],
    'af': ['afghan', 'afghanistan'],
    'ua': ['ukraine'],
    'pl': ['poland'],
}

def download_file(url, save_path, min_size_kb=5):
    """Download a file from URL and save it to the specified path"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, stream=True, timeout=10)
        
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Check file size to make sure it's not an error page
            file_size_kb = os.path.getsize(save_path) / 1024
            
            return True
        else:
            print(f"HTTP error {response.status_code}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        if os.path.exists(save_path):
            os.remove(save_path)
        return False



def download_flag_from_github(country_code, country_name):
    """Download flag from GitHub repository"""
    save_path = f"data/flags/{country_code}.png"
    
    # Check if we already have this flag
    if os.path.exists(save_path) and os.path.getsize(save_path) > 10000:
        print(f"Flag for {country_name} already exists, skipping")
        return True
    
    # GitHub repository for country flags
    github_base = "https://raw.githubusercontent.com/hampusborgos/country-flags/main/"
    
    # Try PNG (1000px) first
    url = f"{github_base}png1000px/{country_code}.png"
    print(f"Trying {url}")
    
    if download_file(url, save_path, min_size_kb=10):
        print(f"Downloaded flag for {country_name} from GitHub (PNG)")
        return True
    
    # Try SVG if PNG fails
    svg_url = f"{github_base}svg/{country_code}.svg"
    svg_save_path = f"data/flags/{country_code}.svg"
    print(f"Trying {svg_url}")
    
    if download_file(svg_url, svg_save_path, min_size_kb=1):
        print(f"Downloaded flag for {country_name} from GitHub (SVG)")
        return True
    
    return False

def download_flag_from_flagpedia(country_code, country_name):
    """Download flag from Flagpedia.net"""
    save_path = f"data/flags/{country_code}.png"
    
    # Check if we already have this flag
    if os.path.exists(save_path) and os.path.getsize(save_path) > 10000:
        return True
    
    # Try Flagpedia.net
    url = f"https://flagpedia.net/data/flags/w580/{country_code}.png"
    print(f"Trying {url}")
    
    if download_file(url, save_path, min_size_kb=10):
        print(f"Downloaded flag for {country_name} from Flagpedia")
        return True
    
    return False

def download_flag_from_countryflags(country_code, country_name):
    """Download flag from countryflags.com"""
    save_path = f"data/flags/{country_code}.png"
    
    # Check if we already have this flag
    if os.path.exists(save_path) and os.path.getsize(save_path) > 10000:
        return True
    
    # Format country name for URL
    formatted_name = country_name.lower().replace(' ', '-')
    
    # Try countryflags.com
    url = f"https://www.countryflags.com/wp-content/uploads/{formatted_name}-flag.png"
    print(f"Trying {url}")
    
    if download_file(url, save_path, min_size_kb=10):
        print(f"Downloaded flag for {country_name} from countryflags.com")
        return True
    
    return False

def download_flags(countries):
    """Download flags for all countries using multiple sources"""
    print("Starting flag download process...")
    
    successful = 0
    failed = 0
    
    for country in tqdm(countries, desc="Downloading flags"):
        country_code = country['code']
        country_name = country['name']
        
        # Try GitHub first
        if download_flag_from_github(country_code, country_name):
            successful += 1
            continue
        
        # Try Flagpedia next
        if download_flag_from_flagpedia(country_code, country_name):
            successful += 1
            continue
        
        # Try countryflags.com as a last resort
        if download_flag_from_countryflags(country_code, country_name):
            successful += 1
            continue
        
        # If all methods failed
        print(f"Could not download flag for {country_name}")
        failed += 1
        
        # Short pause between countries
        time.sleep(random.uniform(0.3, 0.7))
    
    # Count actual flag files
    flag_count = len([f for f in os.listdir('data/flags') if f.endswith(('.png', '.svg'))])
    
    print("\nFlag download summary:")
    print(f"Total countries processed: {len(countries)}")
    print(f"Successfully downloaded: {successful}")
    print(f"Failed downloads: {failed}")
    print(f"Total flags in folder: {flag_count}")
    
    return flag_count


def get_anthem_text_from_wikipedia(country_name):
    """Try to get the national anthem text from Wikipedia"""
    # Format the country name for Wikipedia URL
    search_name = country_name.replace(" ", "_")
    
    # Try different URL patterns
    urls_to_try = [
        f"https://en.wikipedia.org/wiki/National_anthem_of_{search_name}",
        f"https://en.wikipedia.org/wiki/{search_name}_national_anthem",
        f"https://en.wikipedia.org/wiki/National_anthem_of_the_{search_name}"
    ]
    
    for url in urls_to_try:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Try to find the anthem name
                anthem_name = None
                anthem_heading = soup.find('h1', {'id': 'firstHeading'})
                if anthem_heading:
                    anthem_name = anthem_heading.text.strip()
                
                # Try to find the lyrics
                anthem_text = ""
                
                # Look for tables that might contain lyrics
                lyrics_tables = soup.select('table.wikitable, table.poem')
                if lyrics_tables:
                    for table in lyrics_tables:
                        rows = table.select('tr')
                        for row in rows:
                            cells = row.select('td')
                            if cells:
                                # Usually, the English translation is in the last column
                                anthem_text += cells[-1].get_text(strip=True) + "\n"
                
                # If we didn't find lyrics in tables, try looking for specific sections
                if not anthem_text:
                    lyrics_section = None
                    for heading in soup.select('h2, h3, h4'):
                        heading_text = heading.get_text().lower()
                        if 'lyrics' in heading_text or 'words' in heading_text or 'text' in heading_text:
                            lyrics_section = heading
                            break
                    
                    if lyrics_section:
                        # Get all content until the next heading
                        content = []
                        for sibling in lyrics_section.find_next_siblings():
                            if sibling.name in ('h2', 'h3', 'h4'):
                                break
                            if sibling.name == 'p':
                                content.append(sibling.get_text(strip=True))
                        
                        anthem_text = "\n".join(content)
                
                # If we still don't have lyrics, try looking for blockquotes
                if not anthem_text:
                    blockquotes = soup.select('blockquote')
                    if blockquotes:
                        anthem_text = blockquotes[0].get_text(strip=True)
                
                # Clean up the text
                anthem_text = re.sub(r'\[\d+\]', '', anthem_text)  # Remove reference numbers
                anthem_text = re.sub(r'\s+', ' ', anthem_text).strip()
                
                if anthem_text and len(anthem_text) > 50:  # Ensure it's substantial text
                    return (anthem_name, anthem_text, "Wikipedia")
        
        except Exception as e:
            print(f"Error accessing {url}: {e}")
        
        # Short delay between requests
        time.sleep(0.5)
    
    return (None, None, None)

def get_anthem_text_from_nationalanthems(country_code, alt_codes):
    """Try to get anthem text from nationalanthems.info"""
    
    # Base URL
    base_url = "https://nationalanthems.info/"
    
    # Try with each code
    for code in [country_code] + alt_codes:
        url = f"{base_url}{code}.html"
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Look for anthem text sections
                # This is based on typical structure of nationalanthems.info
                anthem_sections = soup.select('.lyrics, pre, .anthem-text, .content')
                
                if anthem_sections:
                    anthem_text = anthem_sections[0].get_text(strip=True, separator='\n')
                    
                    # Clean up the text
                    anthem_text = re.sub(r'\s+', ' ', anthem_text).strip()
                    
                    # Try to extract anthem name from title
                    title_elem = soup.find('title')
                    anthem_name = title_elem.get_text().split('-')[0].strip() if title_elem else "Unknown"
                    
                    if anthem_text and len(anthem_text) > 50:  # Ensure it's substantial text
                        return (anthem_name, anthem_text, "NationalAnthems.info")
        
        except Exception as e:
            print(f"Error accessing {url}: {e}")
        
        # Short delay between requests
        time.sleep(0.5)
    
    return (None, None, None)

def download_anthem_texts(countries):
    """Download anthem texts for all countries using multiple sources"""
    print("Starting anthem text download process...")
    
    successful = 0
    failed = 0
    
    for country in tqdm(countries, desc="Downloading anthem texts"):
        country_code = country['code']
        country_name = country['name']
        alt_codes = country['alt_codes'].copy()
        
        # Add special mappings if they exist
        if country_code in SPECIAL_MAPPINGS:
            alt_codes.extend(SPECIAL_MAPPINGS[country_code])
        
        # Define save path
        save_path = f"data/anthems/text/{country_code}.txt"
        
        # Check if we already have this anthem text
        if os.path.exists(save_path) and os.path.getsize(save_path) > 100:
            print(f"Anthem text for {country_name} already exists, skipping")
            successful += 1
            continue
        
        # Try Wikipedia first
        anthem_name, anthem_text, source = get_anthem_text_from_wikipedia(country_name)
        
        # If Wikipedia failed, try nationalanthems.info
        if not anthem_text:
            anthem_name, anthem_text, source = get_anthem_text_from_nationalanthems(country_code, alt_codes)
        
        # If we found anthem text, save it
        if anthem_text:
            # Add header with metadata
            header = f"Country: {country_name}\nAnthem: {anthem_name or 'Unknown'}\nSource: {source or 'Unknown'}\n\n"
            full_text = header + anthem_text
            
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(full_text)
            
            print(f"Downloaded anthem text for {country_name} from {source}")
            successful += 1
        else:
            print(f"Could not find anthem text for {country_name}")
            failed += 1
        
        # Short pause between countries
        time.sleep(random.uniform(0.5, 1.0))
    
    # Count actual anthem text files
    text_count = len([f for f in os.listdir('data/anthems/text') if f.endswith('.txt') and os.path.getsize(f"data/anthems/text/{f}") > 100])
    
    print("\nAnthem text download summary:")
    print(f"Total countries processed: {len(countries)}")
    print(f"Successfully downloaded: {successful}")
    print(f"Failed downloads: {failed}")
    print(f"Total anthem texts in folder: {text_count}")
    
    return text_count


def try_all_possible_urls(country):
    """Try all possible URL patterns for a country's anthem audio (your existing function)"""
    country_code = country['code']
    country_name = country['name']
    alt_codes = country['alt_codes'].copy()
    
    # Add special mappings if they exist
    if country_code in SPECIAL_MAPPINGS:
        alt_codes.extend(SPECIAL_MAPPINGS[country_code])
    
    # Clean country name for URL (lowercase, no spaces)
    clean_name = country_name.lower().replace(' ', '').replace('-', '')
    
    # Add the clean name to alt_codes if not already there
    if clean_name not in alt_codes:
        alt_codes.append(clean_name)
    
    # Remove duplicates
    alt_codes = list(set(alt_codes))
    
    # Define save path
    save_path = f"data/anthems/audio/{country_code}.mp3"
    
    # Check if the file already exists and is large enough
    if os.path.exists(save_path) and os.path.getsize(save_path) > 10000:
        print(f"Anthem audio for {country_name} already exists, skipping")
        return True
    
    print(f"Trying to download anthem audio for {country_name} ({country_code})...")
    
    # Base URL for nationalanthems.info
    base_url = "https://nationalanthems.info/"
    
    # Build URL patterns to try
    urls_to_try = []
    
    # Try direct MP3 download with different patterns
    for code in [country_code] + alt_codes:
        urls_to_try.extend([
            f"{base_url}{code}.mp3",
            f"{base_url}mp3/{code}.mp3",
            f"{base_url}audio/{code}.mp3"
        ])
    
    # Shuffle URLs for variety
    random.shuffle(urls_to_try)
    
    # Try each URL
    for url in urls_to_try:
        print(f"Trying {url}")
        if download_file(url, save_path, min_size_kb=10):
            print(f"Downloaded anthem audio for {country_name}")
            return True
        
        # Small pause between requests to be nice to the server
        time.sleep(0.5)
    
    print(f"Could not find anthem audio for {country_name} using direct URLs")
    return False

# ---------------------- MAIN FUNCTION ----------------------

def main():
    """Main function to run the complete data collection process"""
    print("Starting complete national data collection process (flags, anthem texts, and audio)...")
    
    # Randomize country order for anthem audio
    # But keep consistent order for flags and texts
    countries_copy = COUNTRIES.copy()
    
    # Step 1: Download flags
    flag_count = download_flags(countries_copy)
    
    # Step 2: Download anthem texts
    text_count = download_anthem_texts(countries_copy)
    
    # Step 3: Download anthem audio (using your existing code)
    # Randomize country order for audio to avoid patterns
    random.shuffle(countries_copy)
    
    # First try higher priority countries
    priority_codes = ['af', 'us', 'gb', 'ca', 'au', 'fr', 'de', 'jp', 'cn', 'in', 'br', 'ru', 'mx', 'eg', 'za']
    priority_countries = [c for c in countries_copy if c['code'] in priority_codes]
    other_countries = [c for c in countries_copy if c['code'] not in priority_codes]
    
    countries_to_process = priority_countries + other_countries
    
    audio_successful = 0
    audio_failed = 0
    
    # Process all countries for anthem audio
    for country in tqdm(countries_to_process, desc="Downloading anthem audio"):
        if try_all_possible_urls(country):
            audio_successful += 1
        else:
            audio_failed += 1
        
        # Be nice to the server with a random pause
        time.sleep(random.uniform(0.5, 1.5))
    
    # Count successful audio downloads
    audio_count = len([f for f in os.listdir('data/anthems/audio') 
                      if f.endswith('.mp3') and os.path.getsize(f"data/anthems/audio/{f}") > 10000])
    
    # Generate comprehensive report
    print("\n===== COLLECTION SUMMARY =====")
    print(f"Total countries processed: {len(countries_copy)}")
    print(f"Flags downloaded: {flag_count}")
    print(f"Anthem texts downloaded: {text_count}")
    print(f"Anthem audio files downloaded: {audio_count}")
    
    # Save comprehensive results to CSV
    results = []
    for country in COUNTRIES:
        code = country['code']
        name = country['name']
        
        flag_path = f"data/flags/{code}.png" if os.path.exists(f"data/flags/{code}.png") else f"data/flags/{code}.svg"
        text_path = f"data/anthems/text/{code}.txt"
        audio_path = f"data/anthems/audio/{code}.mp3"
        
        results.append({
            'country_code': code,
            'country_name': name,
            'has_flag': os.path.exists(flag_path),
            'flag_type': 'PNG' if os.path.exists(f"data/flags/{code}.png") else ('SVG' if os.path.exists(f"data/flags/{code}.svg") else 'None'),
            'has_anthem_text': os.path.exists(text_path),
            'text_size_kb': round(os.path.getsize(text_path)/1024, 1) if os.path.exists(text_path) else 0,
            'has_anthem_audio': os.path.exists(audio_path) and os.path.getsize(audio_path) > 10000,
            'audio_size_kb': round(os.path.getsize(audio_path)/1024, 1) if os.path.exists(audio_path) and os.path.getsize(audio_path) > 10000 else 0
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('data/national_data_collection_results.csv', index=False)
    print("Comprehensive results saved to data/national_data_collection_results.csv")

if __name__ == "__main__":
    main()