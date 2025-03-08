import os
import re
import json
import string
import logging
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt_tab')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("space_processor.log"),
        logging.StreamHandler()
    ]
)

# Define directories
DATA_DIR = "AstroCorpus\space_text_corpus"
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Download NLTK resources
def download_nltk_resources():
    """Download required NLTK resources."""
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        logging.info("NLTK resources downloaded successfully")
    except Exception as e:
        logging.error(f"Error downloading NLTK resources: {e}")
        raise

class SpaceTextProcessor:
    def __init__(self):
        """Initialize the text processor."""
        # Ensure NLTK resources are available
        download_nltk_resources()
        
        # Get English stopwords but remove ones important for space context
        self.stop_words = set(stopwords.words('english'))
        # Remove words that are important in astronomy context
        astronomy_terms = {
            'above', 'below', 'up', 'down', 'under', 'over',  # directional terms
            'during', 'before', 'after',  # temporal terms
            'through', 'between',  # spatial terms
            'where', 'when', 'why', 'how',  # question words
            'very', 'most', 'more', 'few', 'no', 'not',  # quantifiers/negation
            'near', 'far', 'beyond', 'around', 'within'  # distance terms
        }
        self.stop_words -= astronomy_terms
        
        # Initialize lemmatizer
        self.lemmatizer = WordNetLemmatizer()
        
        # Compile regex patterns for cleaning
        self.html_pattern = re.compile(r'<.*?>')
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.multiple_spaces = re.compile(r'\s+')
        self.non_ascii = re.compile(r'[^\x00-\x7F]+')
        
        # Define space and astronomy terms to preserve
        self.space_terms = self.load_astronomy_terms()
        
    def load_astronomy_terms(self):
        """Load astronomy and space terms to preserve during processing."""
        # Common space and astronomy terms that should be preserved
        terms = {
            # Celestial bodies
            'sun', 'moon', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune', 'pluto',
            'mercury', 'venus', 'planet', 'star', 'comet', 'asteroid', 'meteor', 'galaxy',
            'nebula', 'quasar', 'pulsar', 'supernova', 'nova', 'exoplanet', 'satellite',
            
            # Astronomical phenomena
            'eclipse', 'transit', 'occultation', 'conjunction', 'opposition', 'aphelion', 'perihelion',
            'equinox', 'solstice', 'retrograde', 'redshift', 'blueshift', 'parallax',
            
            # Space missions and organizations
            'nasa', 'esa', 'spacex', 'jaxa', 'isro', 'roscosmos', 'hubble', 'jwst', 'webb',
            'voyager', 'perseverance', 'curiosity', 'cassini', 'juno', 'apollo', 'artemis',
            
            # Astronomical concepts
            'orbit', 'gravity', 'astronomy', 'astrophysics', 'cosmology', 'astrometry',
            'parsec', 'lightyear', 'au', 'astronomical unit', 'magnitude', 'luminosity',
            
            # Black holes and exotic objects
            'black hole', 'event horizon', 'singularity', 'neutron star', 'white dwarf',
            'brown dwarf', 'red dwarf', 'magnetar', 'wormhole',
            
            # Cosmological concepts
            'big bang', 'cosmic microwave background', 'dark matter', 'dark energy',
            'inflation', 'multiverse', 'universe', 'expansion', 'cosmic', 'spacetime',
            
            # Observational astronomy
            'telescope', 'observatory', 'spectrometer', 'spectrograph', 'interferometer',
            'radio telescope', 'infrared', 'ultraviolet', 'x-ray', 'gamma ray',
            
            # Units and measurements
            'kilometer', 'meter', 'centimeter', 'millimeter', 'micrometer', 'nanometer',
            'light-year', 'astronomical unit', 'parsec', 'kiloparsec', 'megaparsec',
            'kelvin', 'celsius', 'fahrenheit', 'tesla', 'gauss', 'weber',
            
            # Spacecraft components
            'spacecraft', 'module', 'lander', 'rover', 'orbiter', 'probe', 'station',
            'antenna', 'solar panel', 'radioisotope', 'thruster', 'propellant',
            
            # Astronomy events
            'meteor shower', 'conjunction', 'opposition', 'transit', 'occultation',
            'eclipse', 'solar eclipse', 'lunar eclipse', 'transit', 'phase',
            
            # Space exploration
            'astronaut', 'cosmonaut', 'taikonaut', 'spacesuit', 'spacewalk', 'eva',
            'rocket', 'launch', 'landing', 'reentry', 'docking', 'rendezvous'
        }
        
        # Add multi-word terms
        multi_word_terms = {
            'james webb space telescope', 'hubble space telescope', 'solar system',
            'milky way', 'andromeda galaxy', 'international space station',
            'artemis program', 'apollo program', 'voyager mission',
            'curiosity rover', 'perseverance rover', 'spirit rover', 'opportunity rover',
            'dark matter', 'dark energy', 'event horizon', 'red giant',
            'white dwarf', 'black hole', 'neutron star', 'brown dwarf',
            'kuiper belt', 'oort cloud', 'asteroid belt', 'dwarf planet'
        }
        
        return terms.union(multi_word_terms)
    
    def remove_html_tags(self, text):
        """Remove HTML tags from text."""
        # Using BeautifulSoup for better HTML handling
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()
    
    def remove_urls(self, text):
        """Remove URLs from text."""
        return self.url_pattern.sub(' ', text)
    
    def remove_emails(self, text):
        """Remove email addresses from text."""
        return self.email_pattern.sub(' ', text)
    
    def remove_punctuation(self, text, preserve_units=True):
        """Remove punctuation from text, optionally preserving units."""
        if preserve_units:
            # Preserve units like km/h, m/s, etc.
            text = re.sub(r'([^\w/.-])([^\w/.-])', r'\1 \2', text)
            return text
        else:
            # Create a translator that replaces punctuation with spaces
            translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
            return text.translate(translator)
    
    def preserve_special_terms(self, text):
        """Preserve multi-word space and astronomy terms by joining them with underscores."""
        preserved_text = text.lower()
        
        # Sort terms by length (descending) to prioritize longer multi-word terms
        sorted_terms = sorted(self.space_terms, key=len, reverse=True)
        
        for term in sorted_terms:
            if ' ' in term:  # Only process multi-word terms
                # Replace the term with an underscore version
                preserved_text = re.sub(r'\b' + re.escape(term) + r'\b', term.replace(' ', '_'), preserved_text)
        
        return preserved_text
    
    def restore_special_terms(self, text):
        """Restore underscored terms back to space-separated form."""
        restored_text = text
        
        for term in self.space_terms:
            if ' ' in term:
                # Replace the underscore version with the original space version
                restored_text = restored_text.replace(term.replace(' ', '_'), term)
        
        return restored_text
    
    def remove_numbers(self, text, preserve_measurements=True):
        """Remove numbers, optionally preserving measurements."""
        if preserve_measurements:
            # Keep numbers that are part of measurements (e.g., 10km, 5.2AU)
            text = re.sub(r'\b(\d+)\b(?!\s*[a-zA-Z%°])', ' ', text)
            return text
        else:
            # Remove all standalone numbers
            return re.sub(r'\b\d+\b', ' ', text)
    
    def normalize_whitespace(self, text):
        """Normalize whitespace in text."""
        # Replace newlines and tabs with spaces
        text = re.sub(r'[\n\t\r]', ' ', text)
        # Replace multiple spaces with a single space
        return self.multiple_spaces.sub(' ', text).strip()
    
    def remove_non_ascii(self, text):
        """Remove non-ASCII characters."""
        return self.non_ascii.sub(' ', text)
    
    def remove_stopwords(self, text):
        """Remove stopwords while preserving space and astronomy terms."""
        words = word_tokenize(text.lower())
        processed_words = []
        
        for word in words:
            # Keep words that are not stopwords or are space terms
            if word not in self.stop_words or word in self.space_terms or '_' in word:
                processed_words.append(word)
        
        return ' '.join(processed_words)
    
    def lemmatize_text(self, text):
        """Lemmatize text while preserving special terms."""
        words = word_tokenize(text)
        lemmatized_words = []
        
        for word in words:
            # Don't lemmatize space terms or terms with underscores
            if word.lower() in self.space_terms or '_' in word:
                lemmatized_words.append(word)
            else:
                lemmatized_words.append(self.lemmatizer.lemmatize(word))
        
        return ' '.join(lemmatized_words)
    
    def process_text(self, text):
        """Apply all text processing steps."""
        if not text or not isinstance(text, str):
            return ""
        
        # Apply cleaning steps sequentially
        text = self.remove_html_tags(text)
        text = self.remove_urls(text)
        text = self.remove_emails(text)
        text = self.remove_non_ascii(text)
        
        # Preserve special multi-word terms before tokenization
        text = self.preserve_special_terms(text)
        
        text = self.remove_punctuation(text, preserve_units=True)
        text = self.remove_numbers(text, preserve_measurements=True)
        text = self.normalize_whitespace(text)
        text = self.remove_stopwords(text)
        
        # Lemmatize the text (skipping preserved terms)
        text = self.lemmatize_text(text)
        
        # Restore special terms with spaces
        text = self.restore_special_terms(text)
        
        return text.strip()
    
    def process_article(self, article):
        """Process a single article."""
        processed_article = {
            'url': article['url'],
            'title': self.process_text(article['title']),
            'date': article['date'],  # Keep date as is for metadata
            'content': self.process_text(article['content'])
        }
        return processed_article
    
    def process_category(self, category):
        """Process all articles in a category."""
        # Load raw data
        raw_file = os.path.join(RAW_DIR, f"{category}.json")
        if not os.path.exists(raw_file):
            logging.warning(f"Raw file not found for category: {category}")
            return
            
        try:
            with open(raw_file, 'r', encoding='utf-8') as f:
                articles = json.load(f)
        except Exception as e:
            logging.error(f"Error loading raw data for {category}: {e}")
            return
            
        # Process each article
        processed_articles = []
        processed_content = ""
        
        for article in articles:
            processed_article = self.process_article(article)
            processed_articles.append(processed_article)
            
            # Append to full text content
            if processed_article['title']:
                processed_content += processed_article['title'] + "\n\n"
            if processed_article['content']:
                processed_content += processed_article['content'] + "\n\n"
        
        # Save processed articles as JSON for reference
        processed_file_json = os.path.join(PROCESSED_DIR, f"{category}_processed.json")
        with open(processed_file_json, 'w', encoding='utf-8') as f:
            json.dump(processed_articles, f, ensure_ascii=False, indent=2)
            
        # Save processed content as plain text
        processed_file_txt = os.path.join(PROCESSED_DIR, f"{category}.txt")
        with open(processed_file_txt, 'w', encoding='utf-8') as f:
            f.write(processed_content)
            
        logging.info(f"Processed {len(processed_articles)} articles for category: {category}")
        
    def run(self):
        """Run the processor for all categories."""
        # Ensure processed directory exists
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        
        # Get all category files
        category_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.json')]
        
        for file in category_files:
            category = file.split('.')[0]
            logging.info(f"Processing category: {category}")
            self.process_category(category)

if __name__ == "__main__":
    logging.info("Starting SpaceTextCorpus text processing")
    
    processor = SpaceTextProcessor()
    processor.run()
    
    logging.info("SpaceTextCorpus text processing completed")