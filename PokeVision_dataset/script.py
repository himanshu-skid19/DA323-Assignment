import os
import csv
import time
import requests
import random
import pandas as pd
from PIL import Image
from io import BytesIO
import re
import json

# Selenium imports
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class PokemonPersonaImageCollector:
    def __init__(self, dataset_name, categories, images_per_category=50):
        self.dataset_name = dataset_name
        self.categories = categories
        self.images_per_category = images_per_category
        self.base_dir = f"{dataset_name}_dataset"
        self.metadata_path = os.path.join(self.base_dir, f"{dataset_name}_metadata.csv")
        
        # Set user agents for requests
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.4 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
        ]
        
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        
        with open(self.metadata_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['category', 'name', 'image_url', 'filename', 'width', 'height', 'format', 'source'])
        
        print(f"Initialized {self.dataset_name} dataset collector")
    
    def get_pokemon_by_type(self, pokemon_type):
        if pokemon_type.lower() == "mega-evolution":
            return self.get_mega_evolution_pokemon()
        elif pokemon_type.lower() == "fool-persona":
            return self.get_fool_personas()
        url = f"https://pokeapi.co/api/v2/type/{pokemon_type.lower()}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return [entry['pokemon']['name'] for entry in data['pokemon']]
        return []
    
    def get_mega_evolution_pokemon(self):
        mega_bases = [
            "venusaur", "charizard", "blastoise", "alakazam", "gengar",
            "kangaskhan", "pinsir", "gyarados", "aerodactyl", "mewtwo",
            "ampharos", "scizor", "heracross", "houndoom", "tyranitar",
            "blaziken", "gardevoir", "mawile", "aggron", "medicham",
            "manectric", "banette", "absol", "garchomp", "lucario",
            "abomasnow", "beedrill", "pidgeot", "slowbro", "steelix",
            "sceptile", "swampert", "sableye", "sharpedo", "camerupt",
            "altaria", "glalie", "salamence", "metagross", "latias",
            "latios", "rayquaza", "lopunny", "gallade", "audino", "diancie"
        ]
        special_megas = [
            "charizard-mega-x", "charizard-mega-y",
            "mewtwo-mega-x", "mewtwo-mega-y"
        ]
        mega_pokemon = []
        for base in mega_bases:
            if base in ["charizard", "mewtwo"]:
                continue
            mega_pokemon.append(f"{base}-mega")
        mega_pokemon.extend(special_megas)
        return mega_pokemon
    
    def get_fool_personas(self):
        fool_personas = ["Orpheus", "Thanatos", "Messiah", "Orpheus Telos", "Slime", "Legion", "Black Frost", "Izanagi", "Izanagi-no-Okami",
            "Magatsu-Izanagi", "Arsene", "Satanael", "Raoul", "Izanagi-no-Okami Picaro", "Orpheus Picaro", "Thanatos Picaro", "Messiah Picaro",
            "Obariyon", "High Pixie", "Decarabia", "Vishnu", "Yoshitsune", "Alice", "Futsunushi", "Sraosha", "Attis", "Asura", "Jack Frost",
            "Pixie", "Lucifer", "Mara", "Metatron", "Beelzebub", "Seth", "Cybele", "Kohryu", "Norn", "Trumpeter", "Michael", "Gabriel", "Raphael",
            "Uriel", "Sandalphon", "Ardha", "Bai Suzhen", "Shiki-Ouji", "Mothman", "Kaguya", "Kaguya Picaro", "Athena"
        ]
        return fool_personas
    
    def get_image_url(self, name, category):
        headers = {'User-Agent': random.choice(self.user_agents)}
        if category.lower() == "mega-evolution":
            return self.get_mega_evolution_image_url(name, headers)
        elif category.lower() == "fool-persona":
            return self.get_persona_image_url(name, headers)
        else:
            try:
                url = f"https://pokeapi.co/api/v2/pokemon/{name.lower()}"
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    return data['sprites']['other']['official-artwork']['front_default'], "pokeapi"
            except:
                pass
            return None, None
    
    def get_mega_evolution_image_url(self, pokemon_name, headers):
        base_name = pokemon_name.split("-")[0]
        urls_to_try = []
        form_type = None
        if "-mega" in pokemon_name:
            form_type = "mega"
            if "-mega-x" in pokemon_name:
                form_type = "mega-x"
            elif "-mega-y" in pokemon_name:
                form_type = "mega-y"
            if form_type == "mega":
                formatted_name = f"mega-{base_name}"
            elif form_type in ["mega-x", "mega-y"]:
                formatted_name = f"{form_type}-{base_name}"
            urls_to_try.extend([
                f"https://img.pokemondb.net/artwork/large/{formatted_name}.jpg",
                f"https://img.pokemondb.net/artwork/{formatted_name}.jpg",
                f"https://img.pokemondb.net/artwork/vector/large/{formatted_name}.png",
                f"https://archives.bulbagarden.net/media/upload/thumb/f/f7/{base_name.capitalize()}_M.png/600px-{base_name.capitalize()}_M.png",
                f"https://static.wikia.nocookie.net/espokemon/images/b/b4/Mega_{base_name.capitalize()}.png",
                f"https://www.serebii.net/art/th/{self.get_pokedex_number(base_name)}-m.png"
            ])
        for url in urls_to_try:
            try:
                response = requests.get(url, headers=headers, timeout=5)
                print(f"Trying URL: {url} -> Status: {response.status_code}, Bytes: {len(response.content)}")
                if response.status_code == 200 and len(response.content) > 5000:
                    print(f"Found {form_type} image for {pokemon_name} at {url}")
                    return url, "official_mega"
            except Exception as e:
                print(f"Exception for URL {url}: {e}")
                continue
        print(f"Could not find {form_type} artwork for {pokemon_name}, using Selenium Google image download")
        local_path = self.selenium_google_image_download(pokemon_name + " pokemon official artwork")
        if local_path:
            return local_path, "selenium_google_download"
        base_url = f"https://pokeapi.co/api/v2/pokemon/{base_name}"
        try:
            response = requests.get(base_url)
            if response.status_code == 200:
                data = response.json()
                base_artwork = data['sprites']['other']['official-artwork']['front_default']
                print(f"Falling back to base form for {pokemon_name}")
                return base_artwork, "base_fallback"
        except:
            pass
        return None, None
    
    def generate_persona_urls(self, persona_variant):
        urls = []
        urls.extend([
            f"https://static.wikia.nocookie.net/megamitensei/images/b/b0/{persona_variant}.png",
            f"https://static.wikia.nocookie.net/megamitensei/images/0/0e/{persona_variant}_P5R.png",
            f"https://static.wikia.nocookie.net/megamitensei/images/a/a4/{persona_variant}_P5.png",
            f"https://static.wikia.nocookie.net/megamitensei/images/5/5b/{persona_variant}_P4.png",
            f"https://static.wikia.nocookie.net/megamitensei/images/c/c1/{persona_variant}_P3.png"
        ])
        lower_variant = persona_variant.lower()
        urls.extend([
            f"https://static.wikia.nocookie.net/persona/images/{lower_variant[0]}/{lower_variant}/{persona_variant}.png",
            f"https://static.wikia.nocookie.net/persona/images/{lower_variant[0]}/{lower_variant}/{persona_variant}_P5.png",
            f"https://static.wikia.nocookie.net/persona/images/{lower_variant[0]}/{lower_variant}/{persona_variant}_P4.png",
            f"https://static.wikia.nocookie.net/persona/images/{lower_variant[0]}/{lower_variant}/{persona_variant}_P3.png"
        ])
        urls.append(f"https://static.wikia.nocookie.net/megamitensei/images/a/a0/{persona_variant}_Portrait.png")
        return urls
    
    def get_persona_image_url(self, persona_name, headers):
        def clean_variant(name, remove_hyphens=False):
            variant = name.replace(" ", "_")
            if remove_hyphens:
                variant = variant.replace("-", "")
            variant = re.sub(r"[^\w_]", "", variant)
            return variant
        variants = [
            clean_variant(persona_name, remove_hyphens=False),
            clean_variant(persona_name, remove_hyphens=True),
            clean_variant(persona_name.lower(), remove_hyphens=False),
            clean_variant(persona_name.lower(), remove_hyphens=True)
        ]
        variants = list(dict.fromkeys(variants))
        candidate_urls = []
        for variant in variants:
            urls = self.generate_persona_urls(variant)
            print(f"Generated URLs for variant '{variant}': {urls}")
            candidate_urls.extend(urls)
        for url in candidate_urls:
            try:
                response = requests.get(url, headers=headers, timeout=5)
                print(f"Trying persona URL: {url} -> Status: {response.status_code}, Bytes: {len(response.content)}")
                if response.status_code == 200 and len(response.content) > 3000:
                    try:
                        img = Image.open(BytesIO(response.content))
                        if img.width > 200 and img.height > 200:
                            print(f"Found image for {persona_name} at {url}")
                            return url, "official_persona"
                    except Exception as img_err:
                        print(f"Error opening image from {url}: {img_err}")
                        continue
            except Exception as e:
                print(f"Exception for persona URL {url}: {e}")
                continue
        print(f"Could not find official artwork for {persona_name}, using Selenium Google image download")
        local_path = self.selenium_google_image_download(persona_name + " persona game official artwork")
        if local_path:
            return local_path, "selenium_google_download"
        return None, None
    
    def selenium_google_image_download(self, query):
        """
        Opens Google Images, searches for the query, scrolls down, and takes a screenshot
        of the first image thumbnail. Returns the local file path.
        """
        print(f"Using Selenium to download image for query: {query}")
        options = Options()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.headless = True
        driver = webdriver.Chrome(options=options)
        driver.maximize_window()
        driver.get("https://images.google.com/")
        try:
            # Use a CSS selector to find the search box.
            wait = WebDriverWait(driver, 15)
            box = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input.gLFyf")))
            box.clear()
            box.send_keys(query)
            box.send_keys(Keys.ENTER)
            time.sleep(2)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            xpath = '//*[@id="islrg"]/div[1]/div[1]/a[1]/div[1]/img'
            img_element = driver.find_element(By.XPATH, xpath)
            temp_dir = os.path.join(self.base_dir, "temp")
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            safe_query = re.sub(r'\W+', '_', query)
            save_path = os.path.join(temp_dir, f"{safe_query}.png")
            img_element.screenshot(save_path)
            print(f"Downloaded image to {save_path}")
            driver.quit()
            return save_path
        except Exception as e:
            print("Error in selenium_google_image_download:", e)
            driver.quit()
            return None
    
    def get_pokedex_number(self, pokemon_name):
        try:
            url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_name.lower()}"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                return str(data['id']).zfill(3)
            return "000"
        except:
            return "000"
    
    def search_image_online(self, name, category_type):
        if category_type == "pokemon":
            search_term = f"{name} pokemon official artwork"
        else:
            search_term = f"{name} persona game official artwork"
        return self.selenium_google_image_download(search_term)
    
    def download_image(self, url, category, name, source):
        """
        If the URL is a local file path (does not start with "http"), open it directly.
        Otherwise, download via requests.
        """
        folder_name = category.strip()
        category_dir = os.path.join(self.base_dir, folder_name)
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)
        if not url.startswith("http"):
            try:
                img = Image.open(url)
                extension = img.format.lower() if img.format else 'png'
                clean_name = name.replace(" ", "_").replace("-", "_").replace("'", "").replace(".", "")
                filename = f"{clean_name}.{extension}"
                filepath = os.path.join(category_dir, filename)
                img.save(filepath)
                print(f"Saved local image for {name} ({category}) - Source: {source}")
                return {
                    'category': category,
                    'name': name,
                    'image_url': url,
                    'filename': filename,
                    'width': img.width,
                    'height': img.height,
                    'format': extension.upper(),
                    'source': source
                }
            except Exception as e:
                print(f"Error processing local image for {name}: {e}")
                return None
        else:
            headers = {
                'User-Agent': random.choice(self.user_agents),
                'Referer': 'https://www.google.com/'
            }
            try:
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code != 200:
                    print(f"Failed to download image from {url} (status: {response.status_code})")
                    return None
                img = Image.open(BytesIO(response.content))
                extension = img.format.lower() if img.format else 'jpg'
                clean_name = name.replace(" ", "_").replace("-", "_").replace("'", "").replace(".", "")
                filename = f"{clean_name}.{extension}"
                filepath = os.path.join(category_dir, filename)
                img.save(filepath)
                print(f"Saved: {name} ({category}) - Source: {source}")
                return {
                    'category': category,
                    'name': name,
                    'image_url': url,
                    'filename': filename,
                    'width': img.width,
                    'height': img.height,
                    'format': extension.upper(),
                    'source': source
                }
            except Exception as e:
                print(f"Error downloading {name}: {e}")
                return None
    
    def collect_dataset(self):
        all_metadata = []
        for category in self.categories:
            print(f"\n==== Processing category: {category} ====")
            if category.lower() == "fool-persona":
                item_list = self.get_fool_personas()
            else:
                item_list = self.get_pokemon_by_type(category)
            if not item_list:
                print(f"No items found for category: {category}")
                continue
            print(f"Found {len(item_list)} items for {category}")
            random.shuffle(item_list)
            category_count = 0
            special_category = category.lower() in ["mega-evolution", "fool-persona"]
            for name in item_list:
                if category_count >= self.images_per_category:
                    break
                image_url, source = self.get_image_url(name, category)
                if special_category and source == "base_fallback" and category_count < self.images_per_category / 2:
                    print(f"Skipping base fallback for {name} since we want special forms")
                    continue
                if image_url:
                    metadata = self.download_image(image_url, category, name, source)
                    if metadata:
                        all_metadata.append(metadata)
                        category_count += 1
                time.sleep(0.5)
            print(f"Downloaded {category_count} images for '{category}'")
        df = pd.DataFrame(all_metadata)
        if not df.empty:
            df.to_csv(self.metadata_path, index=False)
            category_stats = df['category'].value_counts().reset_index()
            category_stats.columns = ['Category', 'Image Count']
            category_stats.to_csv(os.path.join(self.base_dir, 'category_statistics.csv'), index=False)
            source_stats = df['source'].value_counts().reset_index()
            source_stats.columns = ['Source', 'Image Count']
            source_stats.to_csv(os.path.join(self.base_dir, 'source_statistics.csv'), index=False)
            print(f"\nDataset collection complete. Total images: {len(all_metadata)}")
            print(f"Metadata saved to: {self.metadata_path}")
        else:
            print("No images were collected. Check the error messages above.")

# Categories: Regular Pokémon types, Mega Evolution, and Fool Personas
categories = ["fool-persona"]

if __name__ == "__main__":
    collector = PokemonPersonaImageCollector("PokemonPersona", categories, images_per_category=50)
    collector.collect_dataset()
