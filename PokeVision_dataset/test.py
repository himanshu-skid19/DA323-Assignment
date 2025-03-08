import os
import csv
import time
import requests
import pandas as pd
from PIL import Image
from io import BytesIO

class PokemonImageCollector:
    def __init__(self, dataset_name, categories, images_per_category=50):
        self.dataset_name = dataset_name
        self.categories = categories
        self.images_per_category = images_per_category
        self.base_dir = f"{dataset_name}_dataset"
        self.metadata_path = os.path.join(self.base_dir, f"{dataset_name}_metadata.csv")
        
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        
        with open(self.metadata_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['category', 'pokemon_name', 'image_url', 'filename', 'width', 'height', 'format'])
        
        print(f"Initialized {self.dataset_name} dataset collector")
    
    def get_pokemon_by_type(self, pokemon_type):
        url = f"https://pokeapi.co/api/v2/type/{pokemon_type.lower()}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return [entry['pokemon']['name'] for entry in data['pokemon']]
        return []
    
    def get_image_url(self, pokemon_name):
        url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_name.lower()}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data['sprites']['other']['official-artwork']['front_default']
        return None
    
    def download_image(self, url, category, pokemon_name):
        try:
            folder_name = category.strip()
            category_dir = os.path.join(self.base_dir, folder_name)
            if not os.path.exists(category_dir):
                os.makedirs(category_dir)
            
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return None
            
            img = Image.open(BytesIO(response.content))
            extension = img.format.lower() if img.format else 'jpg'
            filename = f"{pokemon_name}.{extension}"
            filepath = os.path.join(category_dir, filename)
            img.save(filepath)
            
            return {
                'category': category,
                'pokemon_name': pokemon_name,
                'image_url': url,
                'filename': filename,
                'width': img.width,
                'height': img.height,
                'format': extension.upper()
            }
        except Exception as e:
            print(f"Error downloading {pokemon_name}: {e}")
            return None
    
    def collect_dataset(self):
        all_metadata = []
        for category in self.categories:
            print(f"\n==== Processing category: {category} ====")
            pokemon_list = self.get_pokemon_by_type(category)
            if not pokemon_list:
                print(f"No Pokémon found for category: {category}")
                continue
            
            category_count = 0
            for pokemon_name in pokemon_list:
                if category_count >= self.images_per_category:
                    break
                image_url = self.get_image_url(pokemon_name)
                if image_url:
                    metadata = self.download_image(image_url, category, pokemon_name)
                    if metadata:
                        all_metadata.append(metadata)
                        category_count += 1
                time.sleep(0.2)
            print(f"Downloaded {category_count} images for '{category}'")
        
        df = pd.DataFrame(all_metadata)
        if not df.empty:
            df.to_csv(self.metadata_path, index=False)
            print(f"\nDataset collection complete. Total images: {len(all_metadata)}")
        else:
            print("No images were collected. Check the error messages above.")

pokemon_categories = [
    "normal", "fire", "water", "grass", "electric", "ice", "fighting", "poison",
    "ground", "flying", "psychic", "bug", "rock", "ghost", "dragon", "dark", "steel", "fairy"
]

if __name__ == "__main__":
    collector = PokemonImageCollector("PokeVision", pokemon_categories)
    collector.collect_dataset()
