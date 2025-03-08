import os
import re
import time
import json
import random
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("space_collector.log"),
        logging.StreamHandler()
    ]
)

# Configure User-Agent rotation
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0'
]

# Define data directory
DATA_DIR = "space_text_corpus"
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Create directories if they don't exist
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Define categories and NASA-focused endpoint URLs that we know work from the logs
# Define categories and their respective websites
CATEGORIES = {
    "planetary_science": [
        "https://solarsystem.nasa.gov/",
        "https://www.planetary.org/",
        "https://www.universetoday.com/category/planets/",
    ],
    "space_exploration": [
        "https://www.nasa.gov/",
        "https://www.space.com/space-exploration",
        "https://www.esa.int/",
    ],
    "astrophysics": [
        "https://science.nasa.gov/astrophysics/",
        "https://astronomy.com/tags/astrophysics",
        "https://physicsworld.com/c/astronomy-space/astrophysics/",
    ],
    "cosmology": [
        "https://science.nasa.gov/cosmic-origins/",
        "https://astronomy.com/tags/cosmology",
        "https://www.space.com/cosmology",
    ],
    "black_holes": [
        "https://science.nasa.gov/astrophysics/focus-areas/black-holes/",
        "https://www.space.com/black-holes",
        "https://astronomy.com/tags/black-holes",
    ],
    "exoplanets": [
        "https://exoplanets.nasa.gov/",
        "http://exoplanet.eu/",
        "https://www.space.com/exoplanets",
    ],
    "stars_stellar_evolution": [
        "https://science.nasa.gov/astrophysics/focus-areas/how-do-stars-form-and-evolve/",
        "https://www.space.com/stars",
        "https://astronomy.com/tags/stars",
    ],
    "galaxies": [
        "https://science.nasa.gov/astrophysics/focus-areas/beyond-our-galaxy/",
        "https://www.space.com/galaxies",
        "https://astronomy.com/tags/galaxies",
    ],
    "space_weather": [
        "https://www.swpc.noaa.gov/",
        "https://spaceweather.nasa.gov/",
        "https://www.spaceweatherlive.com/",
    ],
    "telescopes_observatories": [
        "https://webb.nasa.gov/",
        "https://www.nasa.gov/mission_pages/hubble/",
        "https://www.skao.int/",
    ],
    "astrobiology_seti": [
        "https://astrobiology.nasa.gov/",
        "https://www.seti.org/",
        "https://astrobio.net/",
    ],
    "space_technology": [
        "https://technology.nasa.gov/",
        "https://www.space.com/space-tech",
        "https://www.esa.int/Enabling_Support/Space_Engineering_Technology",
    ],
    "space_agencies": [
        "https://www.nasa.gov/missions",
        "https://www.esa.int/Science_Exploration/Space_Science/ESA_s_Space_Science_Programme",
        "https://global.jaxa.jp/projects/",
    ],
    "gravitational_waves": [
        "https://www.ligo.caltech.edu/",
        "https://science.nasa.gov/astrophysics/focus-areas/gravitational-waves/",
        "https://www.space.com/gravitational-waves",
    ],
    "asteroids_comets": [
        "https://solarsystem.nasa.gov/small-bodies/",
        "https://www.minorplanetcenter.net/",
        "https://www.space.com/asteroids",
    ],
    "space_history": [
        "https://history.nasa.gov/",
        "https://www.space.com/space-history",
        "https://airandspace.si.edu/",
    ],
    "amateur_astronomy": [
        "https://skyandtelescope.org/",
        "https://astronomy.com/",
        "https://stargazerslounge.com/",
    ],
    "space_physics": [
        "https://science.nasa.gov/heliophysics/",
        "https://physicsworld.com/c/astronomy-space/space-physics/",
        "https://agu.org/space-physics",
    ],
    "dark_matter_energy": [
        "https://science.nasa.gov/astrophysics/focus-areas/dark-energy-dark-matter/",
        "https://www.space.com/dark-matter",
        "https://astronomy.com/tags/dark-matter",
    ],
    "space_news": [
        "https://spacenews.com/",
        "https://spaceflightnow.com/",
        "https://arstechnica.com/science/space/",
    ],
}

class SpaceDataCollector:
    def __init__(self):
        self.visited_urls = set()
        self.max_articles_per_site = 20
        self.delay = 2  # Delay between requests in seconds
        
    def get_random_user_agent(self):
        """Return a random user agent from the predefined list."""
        return random.choice(USER_AGENTS)
    
    def make_request(self, url):
        """Make HTTP request with error handling and retries."""
        headers = {'User-Agent': self.get_random_user_agent()}
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            logging.error(f"Request error for {url}: {e}")
            
            # Retry once after a pause for transient errors
            try:
                time.sleep(5)  # Wait before retry
                logging.info(f"Retrying request for {url}")
                response = requests.get(url, headers=headers, timeout=15)
                response.raise_for_status()
                return response
            except:
                return None
    
    def extract_article_links(self, base_url, soup, category):
        """Extract article links from the webpage with category-specific filtering."""
        article_links = []
        domain = urlparse(base_url).netloc
        
        # Define category-specific keywords to filter articles
        category_keywords = {
            "planetary_science": ["planet", "solar system", "mercury", "venus", "earth", "mars", 
                                 "jupiter", "saturn", "uranus", "neptune", "pluto", "kuiper belt", 
                                 "asteroid", "comet", "dwarf planet", "moon", "lunar"],
            
            "exoplanets": ["exoplanet", "extrasolar", "alien world", "habitable zone", "super earth", 
                          "hot jupiter", "earth-like", "kepler", "tess", "trappist", "planetary system"],
            
            "space_history": ["history", "apollo", "mercury program", "gemini", "shuttle", 
                             "first", "pioneer", "voyager", "historic", "anniversary", "heritage", "archive"],
            
            "black_holes": ["black hole", "event horizon", "singularity", "supermassive", 
                           "quasar", "gravitational wave", "ligo", "hawking", "accretion disk"],
            
            "stars_stellar_evolution": ["star", "stellar", "supernova", "red giant", "white dwarf", 
                                      "neutron star", "pulsar", "binary", "nebula", "fusion", "sunspot"],
            
            "space_telescopes": ["telescope", "hubble", "james webb", "webb", "jwst", "spitzer", 
                               "chandra", "fermi", "observatory", "kepler", "tess"],
            
            "space_missions": ["mission", "launch", "spacecraft", "rover", "lander", "orbiter", 
                             "probe", "satellite", "iss", "artemis", "perseverance", "opportunity"],
            
            "asteroids_comets": ["asteroid", "comet", "meteor", "meteorite", "impact", "bennu", 
                               "ryugu", "osiris-rex", "dimorphos", "dart", "didymos"],
            
            "cosmic_origins": ["big bang", "origin", "universe", "cosmic", "inflation", 
                             "early universe", "primordial", "formation", "cosmic background"],
            
            "mars_exploration": ["mars", "perseverance", "curiosity", "opportunity", "ingenuity",
                               "insight", "rover", "red planet", "martian", "olympus mons", "valles marineris"]
        }
        
        # NASA-specific selectors that we know work
        selectors = [
            'a', '.list_view_item a', '.content_page a',
            '.article-card a', '.image-feature a', '.feature-article a',
            '.mission-update a', '.feature-item a', '.grid-feature a',
            '.card a', '.card-body a', '.list-item a',
            '.media-blog a', '.media-link a', '.cards-container a', '.collection-item a',
            '.card-group a', '.grid a', '.grid-cell a', '.index-card a'
        ]
        
        # Extract all links
        for selector in selectors:
            links = soup.select(selector)
            for link in links:
                href = link.get('href')
                if href and not href.startswith('#') and not href.startswith('javascript:'):
                    # Convert relative URLs to absolute
                    if not href.startswith('http'):
                        href = urljoin(base_url, href)
                    
                    # Only include NASA URLs and filter for detailed pages
                    parsed_url = urlparse(href)
                    if parsed_url.netloc.endswith('nasa.gov') and len(parsed_url.path.split('/')) > 2:
                        # Get link text for keyword matching
                        link_text = link.get_text().strip().lower()
                        
                        # Check if link or URL contains category-specific keywords
                        keywords = category_keywords.get(category, [])
                        if any(keyword.lower() in link_text.lower() for keyword in keywords) or \
                           any(keyword.lower() in href.lower() for keyword in keywords):
                            article_links.append(href)
                        # For general sections like news, include anything that looks like an article
                        elif '/news/' in base_url:
                            if any(['/feature/' in href, '/article/' in href, '/news/' in href, 
                                  '/image-feature/' in href, '/missions/' in href]):
                                article_links.append(href)
        
        # Remove duplicates
        article_links = list(set(article_links))
        
        # If not enough category-specific articles, add some general ones from NASA
        if len(article_links) < self.max_articles_per_site / 2:
            general_links = []
            for link in soup.find_all('a', href=True):
                href = link.get('href')
                if href and not href.startswith('#') and not href.startswith('javascript:'):
                    if not href.startswith('http'):
                        href = urljoin(base_url, href)
                    
                    parsed_url = urlparse(href)
                    if parsed_url.netloc.endswith('nasa.gov') and len(parsed_url.path.split('/')) > 2:
                        if any(['/feature/' in href, '/article/' in href, '/news/' in href, 
                              '/image-feature/' in href, '/missions/' in href, '/science/' in href]):
                            general_links.append(href)
            
            # Add general links until we reach target or run out
            for link in general_links:
                if link not in article_links:
                    article_links.append(link)
                    if len(article_links) >= self.max_articles_per_site:
                        break
        
        return article_links[:self.max_articles_per_site]
    
    def extract_article_content(self, url, soup):
        """Extract title, date, and content from an article page."""
        article_data = {
            'url': url,
            'title': '',
            'date': '',
            'content': ''
        }
        
        # Extract title - NASA specific
        title_selectors = [
            'h1', 'h1.title', 'h1.article-title', 'h1.feature-title', 
            'h1.mission-title', 'h1.page-title', '.article-header h1',
            '.page-header h1', '.feature-header h1', '.content-title',
            '.wysiwyg_content h1', '.article-body h1'
        ]
        
        for selector in title_selectors:
            title_elem = soup.select_one(selector)
            if title_elem:
                article_data['title'] = title_elem.get_text().strip()
                break
        
        # Fallback to meta title
        if not article_data['title']:
            meta_title = soup.find('meta', property='og:title') or soup.find('meta', attrs={'name': 'title'})
            if meta_title:
                article_data['title'] = meta_title.get('content', '')
        
        # Extract date
        date_selectors = [
            'time', '.date', '.published', '.article-date', '.pub-date',
            '.release-date', '.meta-date', '.article-meta time',
            'meta[name="date"]', 'meta[property="article:published_time"]'
        ]
        
        for selector in date_selectors:
            date_elem = soup.select_one(selector)
            if date_elem:
                if date_elem.name == 'meta':
                    article_data['date'] = date_elem.get('content', '')
                else:
                    article_data['date'] = date_elem.get_text().strip()
                break
        
        # Extract content
        content_selectors = [
            '.wysiwyg_content', '.article-body', '.feature-content',
            '.mission-content', '.page-content', '.article-text',
            '#article-body', '.site-main', '.page-body', '.content-body',
            '.article-content', '.main-content', '.rich-text', '.content',
            '.nasa-article-content', '.nasa-feature-content'
        ]
        
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                # Remove unwanted elements
                for element in content_elem.select('script, style, nav, header, footer, aside, .share-tools, .social-media, .related-content'):
                    element.decompose()
                
                # Get all paragraphs
                paragraphs = content_elem.find_all('p')
                if paragraphs:
                    article_data['content'] = '\n\n'.join([p.get_text().strip() for p in paragraphs])
                else:
                    # If no paragraphs, get all text
                    article_data['content'] = content_elem.get_text().strip()
                break
        
        # Fallback content extraction - find the biggest text blob
        if not article_data['content']:
            # Find all text blocks
            text_blocks = []
            for tag in soup.find_all(['div', 'section', 'article']):
                # Skip navigation, headers, etc.
                if any(cls in tag.get('class', []) for cls in ['nav', 'header', 'footer', 'sidebar', 'menu']):
                    continue
                
                # Get text and clean it
                text = tag.get_text().strip()
                if len(text) > 200:  # Only substantial blocks
                    text_blocks.append(text)
            
            if text_blocks:
                # Use the largest text block
                text_blocks.sort(key=len, reverse=True)
                article_data['content'] = text_blocks[0]
        
        return article_data
    
    def crawl_site(self, url, category):
        """Crawl a website to extract articles relevant to the category."""
        logging.info(f"Crawling {url} for category: {category}")
        articles = []
        
        response = self.make_request(url)
        if not response:
            return articles
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract category-relevant article links
        article_links = self.extract_article_links(url, soup, category)
        logging.info(f"Found {len(article_links)} {category} article links for {url}")
        
        # Process each article
        for article_url in article_links:
            if article_url in self.visited_urls:
                continue
                
            self.visited_urls.add(article_url)
            logging.info(f"Processing article: {article_url}")
            
            # Add delay between requests
            time.sleep(self.delay)
            
            # Request the article page
            article_response = self.make_request(article_url)
            if not article_response:
                continue
                
            article_soup = BeautifulSoup(article_response.text, 'html.parser')
            article_data = self.extract_article_content(article_url, article_soup)
            
            # Check if we got meaningful content
            if article_data['title'] and article_data['content'] and len(article_data['content']) > 200:
                articles.append(article_data)
                logging.info(f"Successfully extracted article: {article_data['title']}")
            else:
                logging.warning(f"Failed to extract meaningful content from {article_url}")
            
            # Check if we've reached our quota
            if len(articles) >= self.max_articles_per_site:
                break
                
        return articles
    
    def collect_category_data(self, category, urls):
        """Collect data for a specific category from multiple URLs."""
        category_articles = []
        
        for url in urls:
            articles = self.crawl_site(url, category)
            category_articles.extend(articles)
            
            # If we have enough articles, stop
            if len(category_articles) >= self.max_articles_per_site * 2:
                break
        
        # Ensure we have enough articles for the category
        if category_articles:
            logging.info(f"Collected {len(category_articles)} articles for {category}")
            self.save_raw_data(category, category_articles)
        else:
            logging.warning(f"No articles found for category: {category}")
    
    def save_raw_data(self, category, articles):
        """Save raw articles data to JSON file."""
        if not articles:
            logging.warning(f"No articles found for category: {category}")
            return
            
        file_path = os.path.join(RAW_DIR, f"{category}.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
            
        logging.info(f"Saved {len(articles)} articles for category {category} to {file_path}")
    
    def run(self):
        """Run the data collector for all categories."""
        for category, urls in CATEGORIES.items():
            logging.info(f"Starting collection for category: {category}")
            self.collect_category_data(category, urls)
            logging.info(f"Completed collection for category: {category}")


if __name__ == "__main__":
    logging.info("Starting SpaceTextCorpus data collection")
    
    # Start the collector
    collector = SpaceDataCollector()
    collector.run()
    
    logging.info("SpaceTextCorpus data collection completed")