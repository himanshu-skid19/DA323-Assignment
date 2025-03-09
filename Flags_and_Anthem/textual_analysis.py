import os
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
from textblob import TextBlob
from gensim.models import Word2Vec
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Define paths - update this to your actual path
base_path = "data/anthem_translations"

# ISO 2-letter country code to country name mapping
country_map = {
    'ad': 'Andorra', 'ae': 'United Arab Emirates', 'af': 'Afghanistan', 'ag': 'Antigua and Barbuda',
    'al': 'Albania', 'am': 'Armenia', 'ao': 'Angola', 'ar': 'Argentina', 'at': 'Austria',
    'au': 'Australia', 'az': 'Azerbaijan', 'ba': 'Bosnia and Herzegovina', 'bb': 'Barbados',
    'bd': 'Bangladesh', 'be': 'Belgium', 'bf': 'Burkina Faso', 'bg': 'Bulgaria', 'bh': 'Bahrain',
    'bi': 'Burundi', 'bj': 'Benin', 'bn': 'Brunei', 'bo': 'Bolivia', 'br': 'Brazil',
    'bs': 'Bahamas', 'bt': 'Bhutan', 'bw': 'Botswana', 'by': 'Belarus', 'bz': 'Belize',
    'ca': 'Canada', 'cd': 'Democratic Republic of the Congo', 'cf': 'Central African Republic',
    'cg': 'Republic of the Congo', 'ch': 'Switzerland', 'ci': 'Ivory Coast', 'cl': 'Chile',
    'cm': 'Cameroon', 'cn': 'China', 'co': 'Colombia', 'cr': 'Costa Rica', 'cu': 'Cuba',
    'cv': 'Cape Verde', 'cy': 'Cyprus', 'cz': 'Czech Republic', 'de': 'Germany', 'dj': 'Djibouti',
    'dk': 'Denmark', 'dm': 'Dominica', 'do': 'Dominican Republic', 'dz': 'Algeria', 'ec': 'Ecuador',
    'ee': 'Estonia', 'eg': 'Egypt', 'er': 'Eritrea', 'es': 'Spain', 'et': 'Ethiopia',
    'fi': 'Finland', 'fj': 'Fiji', 'fm': 'Micronesia', 'fr': 'France', 'ga': 'Gabon',
    'gb': 'United Kingdom', 'gd': 'Grenada', 'ge': 'Georgia', 'gh': 'Ghana', 'gm': 'Gambia',
    'gn': 'Guinea', 'gq': 'Equatorial Guinea', 'gr': 'Greece', 'gt': 'Guatemala', 'gw': 'Guinea-Bissau',
    'gy': 'Guyana', 'hn': 'Honduras', 'hr': 'Croatia', 'ht': 'Haiti', 'hu': 'Hungary',
    'id': 'Indonesia', 'ie': 'Ireland', 'il': 'Israel', 'in': 'India', 'iq': 'Iraq',
    'ir': 'Iran', 'is': 'Iceland', 'it': 'Italy', 'jm': 'Jamaica', 'jo': 'Jordan',
    'jp': 'Japan', 'ke': 'Kenya', 'kg': 'Kyrgyzstan', 'kh': 'Cambodia', 'ki': 'Kiribati',
    'km': 'Comoros', 'kn': 'Saint Kitts and Nevis', 'kp': 'North Korea', 'kr': 'South Korea',
    'kw': 'Kuwait', 'kz': 'Kazakhstan', 'la': 'Laos', 'lb': 'Lebanon', 'lc': 'Saint Lucia',
    'li': 'Liechtenstein', 'lk': 'Sri Lanka', 'lr': 'Liberia', 'ls': 'Lesotho', 'lt': 'Lithuania',
    'lu': 'Luxembourg', 'lv': 'Latvia', 'ly': 'Libya', 'ma': 'Morocco', 'mc': 'Monaco',
    'md': 'Moldova', 'me': 'Montenegro', 'mg': 'Madagascar', 'mh': 'Marshall Islands',
    'mk': 'North Macedonia', 'ml': 'Mali', 'mm': 'Myanmar', 'mn': 'Mongolia', 'mr': 'Mauritania',
    'mt': 'Malta', 'mu': 'Mauritius', 'mv': 'Maldives', 'mw': 'Malawi', 'mx': 'Mexico',
    'my': 'Malaysia', 'mz': 'Mozambique', 'na': 'Namibia', 'ne': 'Niger', 'ng': 'Nigeria',
    'ni': 'Nicaragua', 'nl': 'Netherlands', 'no': 'Norway', 'np': 'Nepal', 'nr': 'Nauru',
    'nz': 'New Zealand', 'om': 'Oman', 'pa': 'Panama', 'pe': 'Peru', 'pg': 'Papua New Guinea',
    'ph': 'Philippines', 'pk': 'Pakistan', 'pl': 'Poland', 'pt': 'Portugal', 'pw': 'Palau',
    'py': 'Paraguay', 'qa': 'Qatar', 'ro': 'Romania', 'rs': 'Serbia', 'ru': 'Russia',
    'rw': 'Rwanda', 'sa': 'Saudi Arabia', 'sb': 'Solomon Islands', 'sc': 'Seychelles',
    'sd': 'Sudan', 'se': 'Sweden', 'sg': 'Singapore', 'si': 'Slovenia', 'sk': 'Slovakia',
    'sl': 'Sierra Leone', 'sm': 'San Marino', 'sn': 'Senegal', 'so': 'Somalia',
    'sr': 'Suriname', 'ss': 'South Sudan', 'st': 'São Tomé and Príncipe', 'sv': 'El Salvador',
    'sy': 'Syria', 'sz': 'Eswatini', 'td': 'Chad', 'tg': 'Togo', 'th': 'Thailand',
    'tj': 'Tajikistan', 'tl': 'East Timor', 'tm': 'Turkmenistan', 'tn': 'Tunisia', 'to': 'Tonga',
    'tr': 'Turkey', 'tt': 'Trinidad and Tobago', 'tv': 'Tuvalu', 'tw': 'Taiwan', 'tz': 'Tanzania',
    'ua': 'Ukraine', 'ug': 'Uganda', 'us': 'United States', 'uy': 'Uruguay', 'uz': 'Uzbekistan',
    'va': 'Vatican City', 'vc': 'Saint Vincent and the Grenadines', 've': 'Venezuela',
    'vn': 'Vietnam', 'vu': 'Vanuatu', 'ws': 'Samoa', 'ye': 'Yemen', 'za': 'South Africa',
    'zm': 'Zambia', 'zw': 'Zimbabwe'
}

# Function to load anthem translations
def load_anthem_translations(base_path):
    """
    Load all anthem translations from the specified directory
    """
    anthems = {}
    
    # Adjust patterns based on your actual file structure
    # This assumes text files with country code names like 'us.txt', 'fr.txt', etc.
    file_pattern = os.path.join(base_path, "*.txt")
    
    print(f"Looking for anthem translation files in: {base_path}")
    files = glob.glob(file_pattern)
    print(f"Found {len(files)} anthem translation files")
    
    for file_path in tqdm(files, desc="Loading anthem translations"):
        try:
            country_code = os.path.splitext(os.path.basename(file_path))[0].lower()
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            anthems[country_code] = text
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    return anthems

# Text preprocessing functions
def preprocess_text(text, remove_stopwords=True):
    """
    Preprocess text: lowercase, remove punctuation and numbers, tokenize,
    remove stopwords if specified, and lemmatize
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Remove short words (less than 3 characters)
    tokens = [token for token in tokens if len(token) >= 3]
    
    return tokens

def tokens_to_text(tokens):
    """Convert tokens back to text"""
    return ' '.join(tokens)

# Basic text statistics
def calculate_text_statistics(anthems):
    """Calculate basic statistics for each anthem"""
    stats = []
    
    for country_code, text in anthems.items():
        # Original text
        word_count_original = len(word_tokenize(text))
        
        # Preprocessed text with stopwords
        tokens_with_stopwords = preprocess_text(text, remove_stopwords=False)
        
        # Preprocessed text without stopwords
        tokens_without_stopwords = preprocess_text(text, remove_stopwords=True)
        
        # Calculate stats
        stat = {
            'country_code': country_code,
            'country_name': country_map.get(country_code, country_code),
            'original_word_count': word_count_original,
            'processed_word_count_with_stopwords': len(tokens_with_stopwords),
            'processed_word_count_without_stopwords': len(tokens_without_stopwords),
            'stopwords_removed': len(tokens_with_stopwords) - len(tokens_without_stopwords),
            'stopwords_percentage': ((len(tokens_with_stopwords) - len(tokens_without_stopwords)) / 
                                    len(tokens_with_stopwords) * 100 if len(tokens_with_stopwords) > 0 else 0),
            'unique_words': len(set(tokens_without_stopwords)),
            'lexical_diversity': (len(set(tokens_without_stopwords)) / 
                                len(tokens_without_stopwords) if len(tokens_without_stopwords) > 0 else 0)
        }
        stats.append(stat)
    
    return pd.DataFrame(stats)

# Word frequency analysis
def analyze_word_frequencies(anthems):
    """Analyze word frequencies across all anthems"""
    # Preprocess all anthems
    processed_anthems = {country: preprocess_text(text) for country, text in anthems.items()}
    
    # All words from all anthems
    all_words = []
    for tokens in processed_anthems.values():
        all_words.extend(tokens)
    
    # Count frequencies
    word_counts = Counter(all_words)
    
    # Top 50 most common words
    most_common = word_counts.most_common(50)
    
    return word_counts, most_common, processed_anthems

# Topic modeling with LDA
def perform_topic_modeling(processed_anthems, num_topics=5, num_words=10):
    """Perform topic modeling using Latent Dirichlet Allocation"""
    # Convert processed tokens to text
    texts = [tokens_to_text(tokens) for tokens in processed_anthems.values()]
    countries = list(processed_anthems.keys())
    
    # Create document-term matrix
    vectorizer = CountVectorizer(max_df=0.95, min_df=2)
    doc_term_matrix = vectorizer.fit_transform(texts)
    
    # LDA model
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(doc_term_matrix)
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Get topics
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[:-num_words-1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topics.append((topic_idx, top_words))
    
    # Get topic distribution for each anthem
    topic_distributions = lda.transform(doc_term_matrix)
    
    # Create a dataframe with country and dominant topic
    topic_df = pd.DataFrame({
        'country_code': countries,
        'country_name': [country_map.get(code, code) for code in countries],
        'dominant_topic': np.argmax(topic_distributions, axis=1)
    })
    
    # Add topic distribution columns
    for i in range(num_topics):
        topic_df[f'topic_{i}_weight'] = topic_distributions[:, i]
    
    return topics, topic_df

# Sentiment analysis
def analyze_sentiment(anthems):
    """Analyze sentiment for each anthem"""
    sentiment_data = []
    
    for country_code, text in anthems.items():
        blob = TextBlob(text)
        sentiment = blob.sentiment
        
        sentiment_data.append({
            'country_code': country_code,
            'country_name': country_map.get(country_code, country_code),
            'polarity': sentiment.polarity,
            'subjectivity': sentiment.subjectivity
        })
    
    return pd.DataFrame(sentiment_data)

# Word Cloud generation - FIXED VERSION WITH BOTH PARAMETERS
def generate_word_clouds(processed_anthems, anthems, output_dir="anthem_wordclouds"):
    """Generate word clouds for each anthem and a combined one"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Combined word cloud
    all_words = []
    for tokens in processed_anthems.values():
        all_words.extend(tokens)
    
    combined_text = " ".join(all_words)
    
    wordcloud = WordCloud(width=800, height=400, background_color='white', 
                         max_words=100, contour_width=3, contour_color='steelblue')
    wordcloud.generate(combined_text)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Combined Word Cloud from All National Anthems')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/combined_wordcloud.png")
    plt.close()
    
    # Generate word clouds for top 10 longest anthems
    stats_df = calculate_text_statistics(anthems)
    longest_anthems = stats_df.sort_values('original_word_count', ascending=False).head(10)
    
    for _, row in longest_anthems.iterrows():
        country_code = row['country_code']
        country_name = row['country_name']
        
        if country_code in processed_anthems:
            text = " ".join(processed_anthems[country_code])
            
            wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                max_words=50, contour_width=3, contour_color='steelblue')
            wordcloud.generate(text)
            
            plt.figure(figsize=(8, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Word Cloud: {country_name} National Anthem')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{country_code}_wordcloud.png")
            plt.close()

# Similarity analysis
def calculate_anthem_similarities(anthems, processed_anthems):
    """Calculate similarity between anthems using TF-IDF and cosine similarity"""
    # Convert processed tokens to text
    texts = [tokens_to_text(tokens) for tokens in processed_anthems.values()]
    countries = list(processed_anthems.keys())
    
    # TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix)
    
    # Create a DataFrame for the similarity matrix
    similarity_df = pd.DataFrame(cosine_sim, index=countries, columns=countries)
    
    # Find the most similar pairs
    similar_pairs = []
    
    for i in range(len(countries)):
        for j in range(i+1, len(countries)):
            country1 = countries[i]
            country2 = countries[j]
            similarity = similarity_df.loc[country1, country2]
            
            similar_pairs.append({
                'country1_code': country1,
                'country1_name': country_map.get(country1, country1),
                'country2_code': country2,
                'country2_name': country_map.get(country2, country2),
                'similarity': similarity
            })
    
    similar_pairs_df = pd.DataFrame(similar_pairs)
    
    return similarity_df, similar_pairs_df

# Common themes analysis using N-grams
def analyze_common_phrases(processed_anthems, n=2, top_n=20):
    """Analyze common phrases (n-grams) across anthems"""
    all_ngrams = []
    
    for tokens in processed_anthems.values():
        text_ngrams = list(ngrams(tokens, n))
        all_ngrams.extend(text_ngrams)
    
    # Count frequencies
    ngram_counts = Counter(all_ngrams)
    
    # Convert to readable format
    readable_ngrams = []
    for ngram, count in ngram_counts.most_common(top_n):
        phrase = " ".join(ngram)
        readable_ngrams.append((phrase, count))
    
    return readable_ngrams

# Visualization functions
def visualize_statistics(stats_df, output_dir="anthem_text_analysis"):
    """Visualize basic statistics"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Word count distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(stats_df['original_word_count'], kde=True)
    plt.title('Distribution of National Anthem Word Counts')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.savefig(f"{output_dir}/word_count_distribution.png")
    plt.close()
    
    # 2. Top 10 longest and shortest anthems
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    longest = stats_df.sort_values('original_word_count', ascending=False).head(10)
    sns.barplot(x='original_word_count', y='country_name', data=longest)
    plt.title('Top 10 Longest National Anthems')
    plt.xlabel('Word Count')
    
    plt.subplot(2, 1, 2)
    shortest = stats_df.sort_values('original_word_count').head(10)
    sns.barplot(x='original_word_count', y='country_name', data=shortest)
    plt.title('Top 10 Shortest National Anthems')
    plt.xlabel('Word Count')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/anthem_lengths.png")
    plt.close()
    
    # 3. Lexical diversity
    plt.figure(figsize=(10, 6))
    sns.histplot(stats_df['lexical_diversity'], kde=True)
    plt.title('Distribution of Lexical Diversity in National Anthems')
    plt.xlabel('Lexical Diversity (Unique Words / Total Words)')
    plt.ylabel('Frequency')
    plt.savefig(f"{output_dir}/lexical_diversity.png")
    plt.close()
    
    # 4. Top 10 anthems by lexical diversity
    plt.figure(figsize=(12, 6))
    diverse = stats_df.sort_values('lexical_diversity', ascending=False).head(10)
    sns.barplot(x='lexical_diversity', y='country_name', data=diverse)
    plt.title('Top 10 National Anthems by Lexical Diversity')
    plt.xlabel('Lexical Diversity')
    plt.savefig(f"{output_dir}/top_lexical_diversity.png")
    plt.close()

def visualize_word_frequencies(most_common, output_dir="anthem_text_analysis"):
    """Visualize word frequencies"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Top 20 most common words
    words, counts = zip(*most_common[:20])
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(counts), y=list(words))
    plt.title('Top 20 Most Common Words Across All National Anthems')
    plt.xlabel('Frequency')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/top_words.png")
    plt.close()

def visualize_sentiment(sentiment_df, output_dir="anthem_text_analysis"):
    """Visualize sentiment analysis results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Polarity vs Subjectivity scatter plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='polarity', y='subjectivity', data=sentiment_df)
    
    # Add country labels for outliers
    outliers = sentiment_df[
        (sentiment_df['polarity'] > sentiment_df['polarity'].quantile(0.9)) | 
        (sentiment_df['polarity'] < sentiment_df['polarity'].quantile(0.1)) |
        (sentiment_df['subjectivity'] > sentiment_df['subjectivity'].quantile(0.9)) |
        (sentiment_df['subjectivity'] < sentiment_df['subjectivity'].quantile(0.1))
    ]
    
    for _, row in outliers.iterrows():
        plt.annotate(row['country_name'], 
                    (row['polarity'], row['subjectivity']),
                    xytext=(5, 5),
                    textcoords='offset points')
    
    plt.title('Sentiment Analysis of National Anthems')
    plt.xlabel('Polarity (Negative → Positive)')
    plt.ylabel('Subjectivity (Objective → Subjective)')
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/sentiment_scatter.png")
    plt.close()
    
    # 2. Top 10 most positive and negative anthems
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    positive = sentiment_df.sort_values('polarity', ascending=False).head(10)
    sns.barplot(x='polarity', y='country_name', data=positive)
    plt.title('Top 10 Most Positive National Anthems')
    plt.xlabel('Polarity Score')
    
    plt.subplot(2, 1, 2)
    negative = sentiment_df.sort_values('polarity').head(10)
    sns.barplot(x='polarity', y='country_name', data=negative)
    plt.title('Top 10 Most Negative National Anthems')
    plt.xlabel('Polarity Score')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sentiment_extremes.png")
    plt.close()

def visualize_similarity(similar_pairs_df, output_dir="anthem_text_analysis"):
    """Visualize anthem similarities"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Top 15 most similar anthem pairs
    plt.figure(figsize=(12, 8))
    most_similar = similar_pairs_df.sort_values('similarity', ascending=False).head(15)
    
    # Create better labels
    most_similar['pair_label'] = most_similar.apply(
        lambda x: f"{x['country1_name']} & {x['country2_name']}", axis=1)
    
    sns.barplot(x='similarity', y='pair_label', data=most_similar)
    plt.title('Top 15 Most Similar National Anthem Pairs')
    plt.xlabel('Similarity Score')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/most_similar_anthems.png")
    plt.close()

def visualize_common_phrases(common_phrases, output_dir="anthem_text_analysis"):
    """Visualize common phrases"""
    os.makedirs(output_dir, exist_ok=True)
    
    phrases, counts = zip(*common_phrases)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=list(counts), y=list(phrases))
    plt.title('Most Common Phrases in National Anthems')
    plt.xlabel('Frequency')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/common_phrases.png")
    plt.close()

def visualize_topics(topics, topic_df, output_dir="anthem_text_analysis"):
    """Visualize topic modeling results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Topic distribution
    topic_counts = topic_df['dominant_topic'].value_counts().reset_index()
    topic_counts.columns = ['topic', 'count']
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='topic', y='count', data=topic_counts)
    plt.title('Distribution of Dominant Topics Across National Anthems')
    plt.xlabel('Topic')
    plt.ylabel('Number of Anthems')
    plt.savefig(f"{output_dir}/topic_distribution.png")
    plt.close()
    
    # 2. Top words for each topic
    for topic_idx, top_words in topics:
        plt.figure(figsize=(10, 4))
        y_pos = np.arange(len(top_words))
        plt.barh(y_pos, range(len(top_words), 0, -1))
        plt.yticks(y_pos, top_words)
        plt.title(f'Top Words in Topic {topic_idx}')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/topic_{topic_idx}_words.png")
        plt.close()

# Word embeddings and semantic similarity
def create_word_embeddings(processed_anthems):
    """Create word embeddings using Word2Vec"""
    # Prepare sentences for training
    sentences = list(processed_anthems.values())
    
    # Train Word2Vec model
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4)
    
    return model

def explore_semantic_relationships(word2vec_model, output_dir="anthem_text_analysis"):
    """Explore semantic relationships in the word embeddings"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Find words similar to common patriotic themes
    theme_words = ['nation', 'freedom', 'land', 'people', 'country', 'love', 'glory']
    
    semantic_data = []
    
    for word in theme_words:
        try:
            # Find similar words
            similar_words = word2vec_model.wv.most_similar(word, topn=5)
            
            for similar_word, similarity in similar_words:
                semantic_data.append({
                    'theme_word': word,
                    'similar_word': similar_word,
                    'similarity': similarity
                })
        except KeyError:
            print(f"Word '{word}' not in vocabulary")
    
    # Create and save dataframe
    semantic_df = pd.DataFrame(semantic_data)
    semantic_df.to_csv(f"{output_dir}/semantic_relationships.csv", index=False)
    
    # Visualize
    if not semantic_df.empty:
        plt.figure(figsize=(12, 8))
        for theme in theme_words:
            theme_data = semantic_df[semantic_df['theme_word'] == theme]
            if not theme_data.empty:
                plt.plot(theme_data['similar_word'], theme_data['similarity'], 'o-', label=theme)
        
        plt.title('Semantic Similarity to Key Patriotic Themes')
        plt.xlabel('Similar Words')
        plt.ylabel('Similarity Score')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/semantic_similarity.png")
        plt.close()
    
    return semantic_df

# Main analysis function - FIXED VERSION
def analyze_anthem_texts(base_path):
    """Main function to analyze anthem texts"""
    output_dir = "anthem_text_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load anthem translations
    anthems = load_anthem_translations(base_path)
    
    if not anthems:
        print("No anthem translations found. Please check the path.")
        return
    
    print(f"Loaded {len(anthems)} anthem translations")
    
    # Calculate basic statistics
    print("Calculating basic statistics...")
    stats_df = calculate_text_statistics(anthems)
    stats_df.to_csv(f"{output_dir}/anthem_statistics.csv", index=False)
    
    # Analyze word frequencies
    print("Analyzing word frequencies...")
    word_counts, most_common, processed_anthems = analyze_word_frequencies(anthems)
    
    # Create preprocessed versions (for reference)
    preprocessed = {country: tokens_to_text(tokens) for country, tokens in processed_anthems.items()}
    with open(f"{output_dir}/preprocessed_anthems.txt", 'w', encoding='utf-8') as f:
        for country, text in preprocessed.items():
            f.write(f"=== {country} ===\n{text}\n\n")
    
    # Topic modeling
    print("Performing topic modeling...")
    topics, topic_df = perform_topic_modeling(processed_anthems)
    topic_df.to_csv(f"{output_dir}/anthem_topics.csv", index=False)
    
    # Sentiment analysis
    print("Analyzing sentiment...")
    sentiment_df = analyze_sentiment(anthems)
    sentiment_df.to_csv(f"{output_dir}/anthem_sentiment.csv", index=False)
    
    # Word clouds - FIXED: Pass both processed_anthems and anthems
    print("Generating word clouds...")
    generate_word_clouds(processed_anthems, anthems)
    
    # Similarity analysis
    print("Calculating anthem similarities...")
    similarity_df, similar_pairs_df = calculate_anthem_similarities(anthems, processed_anthems)
    similarity_df.to_csv(f"{output_dir}/similarity_matrix.csv")
    similar_pairs_df.to_csv(f"{output_dir}/similar_anthem_pairs.csv", index=False)
    
    # Common phrases
    print("Analyzing common phrases...")
    bigrams = analyze_common_phrases(processed_anthems, n=2)
    trigrams = analyze_common_phrases(processed_anthems, n=3)
    
    with open(f"{output_dir}/common_phrases.txt", 'w', encoding='utf-8') as f:
        f.write("=== COMMON BIGRAMS ===\n")
        for phrase, count in bigrams:
            f.write(f"{phrase}: {count}\n")
        
        f.write("\n=== COMMON TRIGRAMS ===\n")
        for phrase, count in trigrams:
            f.write(f"{phrase}: {count}\n")
    
    # Word embeddings
    print("Creating word embeddings...")
    word2vec_model = create_word_embeddings(processed_anthems)
    semantic_df = explore_semantic_relationships(word2vec_model)
    
    # Visualizations
    print("Creating visualizations...")
    visualize_statistics(stats_df)
    visualize_word_frequencies(most_common)
    visualize_sentiment(sentiment_df)
    visualize_similarity(similar_pairs_df)
    visualize_common_phrases(bigrams)
    visualize_topics(topics, topic_df)
    
    print("Analysis complete! Results saved to the 'anthem_text_analysis' directory.")
    
    # Return key dataframes for further analysis
    return {
        'stats': stats_df,
        'sentiment': sentiment_df,
        'topics': topic_df,
        'similarities': similar_pairs_df,
        'semantic': semantic_df
    }

# Run the analysis
if __name__ == "__main__":
    results = analyze_anthem_texts(base_path)