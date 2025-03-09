import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.stats import pearsonr, spearmanr
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Define paths to the saved data from previous analyses
audio_data_path = "anthem_features_summary.csv"
text_data_path = "anthem_text_analysis/anthem_statistics.csv"
text_sentiment_path = "anthem_text_analysis/anthem_sentiment.csv"
text_topics_path = "anthem_text_analysis/anthem_topics.csv"

# Create output directory for multimodal analysis
output_dir = "multimodal_analysis"
os.makedirs(output_dir, exist_ok=True)

# Function to load and prepare data
def load_data():
    # Dictionary to store all dataframes
    data = {}
    
    # Load audio features if available
    if os.path.exists(audio_data_path):
        try:
            data['audio'] = pd.read_csv(audio_data_path)
            print(f"Loaded audio features for {len(data['audio'])} countries")
        except Exception as e:
            print(f"Error loading audio data: {e}")
    else:
        print(f"Audio data file not found at {audio_data_path}")
    
    # Load text statistics if available
    if os.path.exists(text_data_path):
        try:
            data['text_stats'] = pd.read_csv(text_data_path)
            print(f"Loaded text statistics for {len(data['text_stats'])} countries")
        except Exception as e:
            print(f"Error loading text statistics: {e}")
    else:
        print(f"Text statistics file not found at {text_data_path}")
    
    # Load text sentiment if available
    if os.path.exists(text_sentiment_path):
        try:
            data['text_sentiment'] = pd.read_csv(text_sentiment_path)
            print(f"Loaded text sentiment for {len(data['text_sentiment'])} countries")
        except Exception as e:
            print(f"Error loading text sentiment: {e}")
    else:
        print(f"Text sentiment file not found at {text_sentiment_path}")
    
    # Load text topics if available
    if os.path.exists(text_topics_path):
        try:
            data['text_topics'] = pd.read_csv(text_topics_path)
            print(f"Loaded text topics for {len(data['text_topics'])} countries")
        except Exception as e:
            print(f"Error loading text topics: {e}")
    else:
        print(f"Text topics file not found at {text_topics_path}")
    
    return data

# Function to merge datasets by country code - FIXED VERSION
def merge_datasets(data):
    merged_data = None
    
    # Start with audio data if available
    if 'audio' in data:
        merged_data = data['audio'][['country_code', 'country_name']].copy()
        
        # Add audio features
        numeric_cols = data['audio'].select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            if col not in ['country_code', 'country_name']:
                merged_data[f'audio_{col}'] = data['audio'][col]
    
    # Add text statistics if available
    if 'text_stats' in data:
        if merged_data is None:
            merged_data = data['text_stats'][['country_code', 'country_name']].copy()
            
            # Add text features with prefix
            text_features = [col for col in data['text_stats'].columns 
                            if col not in ['country_code', 'country_name']]
            for col in text_features:
                merged_data[f'text_{col}'] = data['text_stats'][col]
        else:
            # Create a dataframe for text features
            text_df = data['text_stats'][['country_code']].copy()
            
            # Add text features with prefix
            text_features = [col for col in data['text_stats'].columns 
                            if col not in ['country_code', 'country_name']]
            for col in text_features:
                text_df[f'text_{col}'] = data['text_stats'][col]
            
            # Merge on country_code
            merged_data = pd.merge(merged_data, text_df, on='country_code', how='outer')
    
    # Add text sentiment if available
    if 'text_sentiment' in data:
        if merged_data is None:
            merged_data = data['text_sentiment'][['country_code', 'country_name']].copy()
            merged_data['text_polarity'] = data['text_sentiment']['polarity']
            merged_data['text_subjectivity'] = data['text_sentiment']['subjectivity']
        else:
            # Create sentiment dataframe
            sentiment_df = data['text_sentiment'][['country_code']].copy()
            sentiment_df['text_polarity'] = data['text_sentiment']['polarity']
            sentiment_df['text_subjectivity'] = data['text_sentiment']['subjectivity']
            
            # Merge
            merged_data = pd.merge(merged_data, sentiment_df, on='country_code', how='outer')
    
    # Add text topics if available
    if 'text_topics' in data:
        if merged_data is None:
            merged_data = data['text_topics'][['country_code', 'country_name']].copy()
            
            # Add topic features
            topic_cols = [col for col in data['text_topics'].columns 
                          if col.startswith('topic_') and col.endswith('_weight')]
            for col in topic_cols:
                merged_data[col] = data['text_topics'][col]
        else:
            # Create topics dataframe
            topics_df = data['text_topics'][['country_code']].copy()
            
            # Add topic features
            topic_cols = [col for col in data['text_topics'].columns 
                          if col.startswith('topic_') and col.endswith('_weight')]
            for col in topic_cols:
                topics_df[col] = data['text_topics'][col]
            
            # Merge
            merged_data = pd.merge(merged_data, topics_df, on='country_code', how='outer')
    
    # Remove duplicates if any (based on country_code)
    if merged_data is not None and 'country_code' in merged_data.columns:
        # Check for duplicates
        if merged_data['country_code'].duplicated().any():
            print(f"Warning: Found {merged_data['country_code'].duplicated().sum()} duplicate country codes. Keeping first occurrence.")
            merged_data = merged_data.drop_duplicates(subset=['country_code'], keep='first')
    
    # Print merge summary
    if merged_data is not None:
        print(f"Merged dataset contains {len(merged_data)} countries with {len(merged_data.columns)} features")
        
        # Count NaN values by modality
        audio_cols = [col for col in merged_data.columns if col.startswith('audio_')]
        text_cols = [col for col in merged_data.columns if col.startswith('text_')]
        
        print(f"Audio features: {len(audio_cols)} columns")
        print(f"Text features: {len(text_cols)} columns")
        
        # Count countries with data in each modality
        countries_with_audio = merged_data.dropna(subset=audio_cols, how='all').shape[0] if audio_cols else 0
        countries_with_text = merged_data.dropna(subset=text_cols, how='all').shape[0] if text_cols else 0
        
        print(f"Countries with audio data: {countries_with_audio}")
        print(f"Countries with text data: {countries_with_text}")
        
        # Count countries with data in all modalities
        if audio_cols and text_cols:
            countries_with_both = merged_data.dropna(subset=audio_cols+text_cols, how='all').shape[0]
            countries_with_complete = merged_data.dropna(subset=audio_cols+text_cols).shape[0]
            print(f"Countries with both audio and text data: {countries_with_both}")
            print(f"Countries with complete data across all features: {countries_with_complete}")
    
    return merged_data

# Function to calculate cross-modal correlations
def calculate_cross_modal_correlations(merged_data):
    print("\nCalculating cross-modal correlations...")
    
    # Identify features by modality
    audio_cols = [col for col in merged_data.columns if col.startswith('audio_') 
                  and col != 'audio_country_code' and col != 'audio_country_name']
    text_cols = [col for col in merged_data.columns if col.startswith('text_') 
                 and col != 'text_country_code' and col != 'text_country_name']
    
    correlations = []
    
    # Audio x Text correlations
    if audio_cols and text_cols:
        for audio_col in audio_cols:
            for text_col in text_cols:
                # Get data without NaN values
                valid_data = merged_data[[audio_col, text_col]].dropna()
                
                if len(valid_data) >= 10:  # Only calculate if we have enough data points
                    try:
                        corr, p = pearsonr(valid_data[audio_col], valid_data[text_col])
                        if p < 0.05:  # Only report statistically significant correlations
                            correlations.append({
                                'feature1': audio_col,
                                'feature2': text_col,
                                'modality1': 'audio',
                                'modality2': 'text',
                                'correlation': corr,
                                'p_value': p,
                                'sample_size': len(valid_data)
                            })
                    except Exception as e:
                        pass
    
    # Convert to DataFrame and sort by correlation strength
    if correlations:
        corr_df = pd.DataFrame(correlations)
        corr_df['abs_correlation'] = corr_df['correlation'].abs()
        corr_df = corr_df.sort_values('abs_correlation', ascending=False).drop('abs_correlation', axis=1)
        
        # Save to CSV
        corr_df.to_csv(f"{output_dir}/cross_modal_correlations.csv", index=False)
        
        # Print top correlations
        print(f"Found {len(corr_df)} significant cross-modal correlations")
        print("\nTop 10 strongest cross-modal correlations:")
        print(corr_df.head(10))
        
        return corr_df
    else:
        print("No significant cross-modal correlations found with available data")
        return None

# Function to visualize top correlations
def visualize_top_correlations(merged_data, corr_df, top_n=5):
    if corr_df is None or len(corr_df) == 0:
        print("No correlations to visualize")
        return
    
    print(f"\nVisualizing top {top_n} cross-modal correlations...")
    
    # Get top N correlations
    top_corrs = corr_df.head(top_n)
    
    # Create scatter plots for each correlation
    for i, row in top_corrs.iterrows():
        feature1 = row['feature1']
        feature2 = row['feature2']
        corr = row['correlation']
        p = row['p_value']
        
        # Get data without NaN values
        valid_data = merged_data[['country_name', feature1, feature2]].dropna()
        
        # Create scatter plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=feature1, y=feature2, data=valid_data)
        
        # Add regression line
        sns.regplot(x=feature1, y=feature2, data=valid_data, scatter=False, color='red')
        
        # Label outliers
        if len(valid_data) > 10:
            # Calculate z-scores for both features
            z1 = (valid_data[feature1] - valid_data[feature1].mean()) / valid_data[feature1].std()
            z2 = (valid_data[feature2] - valid_data[feature2].mean()) / valid_data[feature2].std()
            
            # Find outliers (|z| > 1.5 for either feature)
            outliers = valid_data[(z1.abs() > 1.5) | (z2.abs() > 1.5)]
            
            # Add labels for outliers
            for _, row in outliers.iterrows():
                plt.annotate(row['country_name'], 
                            (row[feature1], row[feature2]),
                            xytext=(5, 5),
                            textcoords='offset points')
        
        # Clean up feature names for display
        feature1_display = feature1.replace('_', ' ').title()
        feature2_display = feature2.replace('_', ' ').title()
        
        # Add title and labels
        plt.title(f'Correlation between {feature1_display} and {feature2_display}\n'
                  f'r = {corr:.2f}, p = {p:.4f}, n = {len(valid_data)}')
        plt.xlabel(feature1_display)
        plt.ylabel(feature2_display)
        plt.tight_layout()
        
        # Save figure
        clean_name = f"{feature1}_{feature2}".replace('/', '_').replace(' ', '_')
        plt.savefig(f"{output_dir}/correlation_{clean_name}.png")
        plt.close()

# Function to create multimodal typology
def create_multimodal_typology(merged_data):
    print("\nCreating multimodal typology...")
    
    # Get numeric columns for clustering
    numeric_cols = merged_data.select_dtypes(include=['float64', 'int64']).columns
    
    # Check if we have enough numeric features
    if len(numeric_cols) < 3:
        print("Not enough numeric features for dimensionality reduction")
        return
    
    # Check if we have enough countries with data
    complete_data = merged_data[numeric_cols].dropna()
    if len(complete_data) < 10:
        print(f"Only {len(complete_data)} countries have complete data across modalities, not enough for clustering")
        return
    
    print(f"Performing clustering with {len(complete_data)} countries and {len(numeric_cols)} features")
    
    # Get country names for these rows
    countries = merged_data.loc[complete_data.index, 'country_name'].values
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(complete_data)
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=min(5, len(numeric_cols), len(complete_data)-1))
    pca_result = pca.fit_transform(scaled_data)
    
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    
    # Apply K-means clustering
    # Determine optimal number of clusters (2-6)
    max_clusters = min(6, len(complete_data))
    inertias = []
    for k in range(2, max_clusters+1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(pca_result)
        inertias.append(kmeans.inertia_)
    
    # Find elbow point (simplified method)
    optimal_k = 3  # Default to 3 clusters
    if len(inertias) > 1:
        # Calculate slopes
        slopes = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
        # Find where slope change is smallest (approximate elbow)
        slope_changes = [abs(slopes[i] - slopes[i+1]) for i in range(len(slopes)-1)]
        if slope_changes:
            optimal_k = slope_changes.index(min(slope_changes)) + 3  # +3 because we started at k=2 and need an extra +1
    
    print(f"Optimal number of clusters: {optimal_k}")
    
    # Apply K-means with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(pca_result)
    
    # Create a DataFrame with results
    typology_df = pd.DataFrame({
        'country_name': countries,
        'cluster': clusters
    })
    
    # Add original data
    for col in numeric_cols:
        typology_df[col] = complete_data[col].values
    
    # Save typology to CSV
    typology_df.to_csv(f"{output_dir}/multimodal_typology.csv", index=False)
    
    # Visualize the clusters using PCA
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, cmap='viridis')
    
    # Add country labels
    for i, country in enumerate(countries):
        plt.annotate(country, (pca_result[i, 0], pca_result[i, 1]), fontsize=8)
    
    plt.colorbar(scatter, label='Cluster')
    plt.title('Multimodal Typology of Countries (PCA + K-means)')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/multimodal_typology_pca.png")
    plt.close()
    
    # Also try t-SNE for better visualization if we have enough data points
    if len(complete_data) >= 15:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(complete_data)//2))
        tsne_result = tsne.fit_transform(scaled_data)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=clusters, cmap='viridis')
        
        # Add country labels
        for i, country in enumerate(countries):
            plt.annotate(country, (tsne_result[i, 0], tsne_result[i, 1]), fontsize=8)
        
        plt.colorbar(scatter, label='Cluster')
        plt.title('Multimodal Typology of Countries (t-SNE + K-means)')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/multimodal_typology_tsne.png")
        plt.close()
    
    # Analyze cluster characteristics
    cluster_stats = []
    for cluster_id in range(optimal_k):
        cluster_countries = typology_df[typology_df['cluster'] == cluster_id]
        
        # Calculate mean values for each feature by cluster
        cluster_means = cluster_countries[numeric_cols].mean()
        
        # Calculate z-scores compared to overall means
        overall_means = typology_df[numeric_cols].mean()
        overall_stds = typology_df[numeric_cols].std()
        
        z_scores = (cluster_means - overall_means) / overall_stds
        
        # Get top distinctive features (highest absolute z-scores)
        distinctive_features = z_scores.abs().sort_values(ascending=False).head(5)
        
        cluster_info = {
            'cluster_id': cluster_id,
            'num_countries': len(cluster_countries),
            'countries': ', '.join(cluster_countries['country_name'].values),
            'distinctive_features': ', '.join([f"{feat} ({z_scores[feat]:.2f}σ)" 
                                              for feat in distinctive_features.index])
        }
        cluster_stats.append(cluster_info)
    
    # Create cluster report
    cluster_report = pd.DataFrame(cluster_stats)
    cluster_report.to_csv(f"{output_dir}/cluster_analysis.csv", index=False)
    print("\nCluster analysis complete. See cluster_analysis.csv for details.")
    
    return typology_df

# Function to analyze regional patterns
def analyze_regional_patterns(merged_data):
    print("\nAnalyzing regional patterns...")
    
    # Define regions (simplified)
    regions = {
        'Europe': ['al', 'ad', 'at', 'by', 'be', 'ba', 'bg', 'hr', 'cz', 'dk', 'ee', 'fi', 'fr', 'de', 
                  'gr', 'hu', 'is', 'ie', 'it', 'lv', 'li', 'lt', 'lu', 'mk', 'mt', 'md', 'mc', 'me', 
                  'nl', 'no', 'pl', 'pt', 'ro', 'ru', 'sm', 'rs', 'sk', 'si', 'es', 'se', 'ch', 'ua', 'gb', 'va'],
        'Asia': ['af', 'am', 'az', 'bh', 'bd', 'bt', 'bn', 'kh', 'cn', 'cy', 'ge', 'in', 'id', 'ir', 
                'iq', 'il', 'jp', 'jo', 'kz', 'kw', 'kg', 'la', 'lb', 'my', 'mv', 'mn', 'mm', 'np', 
                'kp', 'om', 'pk', 'ph', 'qa', 'sa', 'sg', 'kr', 'lk', 'sy', 'tw', 'tj', 'th', 'tl', 
                'tr', 'tm', 'ae', 'uz', 'vn', 'ye'],
        'Africa': ['dz', 'ao', 'bj', 'bw', 'bf', 'bi', 'cm', 'cv', 'cf', 'td', 'km', 'cd', 'cg', 'ci', 
                  'dj', 'eg', 'gq', 'er', 'et', 'ga', 'gm', 'gh', 'gn', 'gw', 'ke', 'ls', 'lr', 'ly', 
                  'mg', 'mw', 'ml', 'mr', 'mu', 'ma', 'mz', 'na', 'ne', 'ng', 'rw', 'st', 'sn', 'sc', 
                  'sl', 'so', 'za', 'ss', 'sd', 'sz', 'tz', 'tg', 'tn', 'ug', 'zm', 'zw'],
        'Americas': ['ag', 'ar', 'bs', 'bb', 'bz', 'bo', 'br', 'ca', 'cl', 'co', 'cr', 'cu', 'dm', 
                    'do', 'ec', 'sv', 'gd', 'gt', 'gy', 'ht', 'hn', 'jm', 'mx', 'ni', 'pa', 'py', 
                    'pe', 'kn', 'lc', 'vc', 'sr', 'tt', 'us', 'uy', 've'],
        'Oceania': ['au', 'fj', 'ki', 'mh', 'fm', 'nr', 'nz', 'pw', 'pg', 'ws', 'sb', 'to', 'tv', 'vu']
    }
    
    # Add region to merged data
    merged_data['region'] = np.nan
    for region, country_codes in regions.items():
        merged_data.loc[merged_data['country_code'].str.lower().isin(country_codes), 'region'] = region
    
    # Count countries by region
    region_counts = merged_data['region'].value_counts()
    print("Countries by region:")
    print(region_counts)
    
    # Get numeric columns
    numeric_cols = merged_data.select_dtypes(include=['float64', 'int64']).columns
    
    # Calculate regional averages for all numeric features
    regional_stats = []
    for region in merged_data['region'].dropna().unique():
        region_data = merged_data[merged_data['region'] == region]
        
        # Calculate means for each numeric column
        region_means = {}
        for col in numeric_cols:
            if col in region_data.columns:
                mean_val = region_data[col].mean()
                if not pd.isna(mean_val):
                    region_means[col] = mean_val
        
        regional_stats.append({
            'region': region,
            'num_countries': len(region_data),
            **region_means
        })
    
    # Convert to DataFrame
    if regional_stats:
        regional_df = pd.DataFrame(regional_stats)
        regional_df.to_csv(f"{output_dir}/regional_averages.csv", index=False)
        
        # Identify distinctive features for each region
        print("\nDistinctive features by region:")
        for region in regional_df['region'].unique():
            region_row = regional_df[regional_df['region'] == region].iloc[0]
            
            # Calculate z-scores compared to global average
            z_scores = {}
            for col in numeric_cols:
                if col in region_row and col in merged_data.columns:
                    global_mean = merged_data[col].mean()
                    global_std = merged_data[col].std()
                    
                    if not pd.isna(global_std) and global_std > 0:
                        z_score = (region_row[col] - global_mean) / global_std
                        if not pd.isna(z_score):
                            z_scores[col] = z_score
            
            # Get top 3 distinctive features (highest absolute z-scores)
            if z_scores:
                sorted_features = sorted(z_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                print(f"\n{region}:")
                for feat, z in sorted_features:
                    direction = "higher" if z > 0 else "lower"
                    print(f"  - {feat}: {abs(z):.2f}σ {direction} than average")
    
    # Visualize key regional differences
    if 'regional_df' in locals():
        # Select a few interesting features from each modality (if available)
        interesting_features = []
        
        # Audio features
        audio_features = [col for col in numeric_cols if col.startswith('audio_')]
        if audio_features:
            interesting_features.extend(audio_features[:min(2, len(audio_features))])
        
        # Text features
        text_features = [col for col in numeric_cols if col.startswith('text_')]
        if text_features:
            interesting_features.extend(text_features[:min(2, len(text_features))])
        
        # Limit to features present in regional_df
        interesting_features = [f for f in interesting_features if f in regional_df.columns]
        
        if interesting_features:
            # Create bar plots for each interesting feature
            for feature in interesting_features:
                plt.figure(figsize=(10, 6))
                
                # Sort by feature value
                sorted_df = regional_df.sort_values(feature)
                
                # Clean feature name for display
                feature_display = feature.replace('_', ' ').title()
                
                # Create bar plot
                sns.barplot(x='region', y=feature, data=sorted_df)
                plt.title(f'Regional Comparison: {feature_display}')
                plt.ylabel(feature_display)
                plt.xlabel('Region')
                plt.xticks(rotation=45)
                
                # Add values on top of bars
                for i, v in enumerate(sorted_df[feature]):
                    plt.text(i, v, f"{v:.2f}", ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(f"{output_dir}/regional_{feature}.png")
                plt.close()

# Function to analyze intra-modal correlations - FIXED VERSION
def analyze_intra_modal_correlations(merged_data):
    print("\nAnalyzing intra-modal correlations...")
    
    # Identify features by modality
    audio_cols = [col for col in merged_data.columns if col.startswith('audio_') 
                  and col != 'audio_country_code' and col != 'audio_country_name']
    text_cols = [col for col in merged_data.columns if col.startswith('text_') 
                 and col != 'text_country_code' and col != 'text_country_name']
    
    # Analyze audio feature correlations
    if len(audio_cols) > 1:
        print("\nAnalyzing correlations between audio features...")
        audio_data = merged_data[audio_cols].dropna()
        
        if len(audio_data) >= 10:
            audio_corr = audio_data.corr()
            
            # Plot correlation heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(audio_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
            plt.title('Correlations Between Audio Features')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/audio_correlation_heatmap.png")
            plt.close()
            
            # Save correlation matrix
            audio_corr.to_csv(f"{output_dir}/audio_correlations.csv")
            
            # Find strongest correlations
            audio_corr_flat = audio_corr.unstack()
            audio_corr_flat = audio_corr_flat[audio_corr_flat < 1.0]  # Remove self-correlations
            audio_corr_flat = audio_corr_flat.sort_values(ascending=False)
            
            print("\nTop 5 audio feature correlations:")
            # Convert to list before slicing
            top_correlations = list(audio_corr_flat.items())[:5]
            for i, ((feat1, feat2), corr) in enumerate(top_correlations):
                print(f"{feat1} ↔ {feat2}: r = {corr:.2f}")
    
    # Analyze text feature correlations
    if len(text_cols) > 1:
        print("\nAnalyzing correlations between text features...")
        text_data = merged_data[text_cols].dropna()
        
        if len(text_data) >= 10:
            text_corr = text_data.corr()
            
            # Plot correlation heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(text_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
            plt.title('Correlations Between Text Features')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/text_correlation_heatmap.png")
            plt.close()
            
            # Save correlation matrix
            text_corr.to_csv(f"{output_dir}/text_correlations.csv")
            
            # Find strongest correlations
            text_corr_flat = text_corr.unstack()
            text_corr_flat = text_corr_flat[text_corr_flat < 1.0]  # Remove self-correlations
            text_corr_flat = text_corr_flat.sort_values(ascending=False)
            
            print("\nTop 5 text feature correlations:")
            # Convert to list before slicing
            top_correlations = list(text_corr_flat.items())[:5]
            for i, ((feat1, feat2), corr) in enumerate(top_correlations):
                print(f"{feat1} ↔ {feat2}: r = {corr:.2f}")

# Function to identify interesting feature relationships - FIXED VERSION
def identify_interesting_relationships(merged_data, corr_df):
    print("\nIdentifying interesting multimodal relationships...")
    
    if corr_df is None or len(corr_df) == 0:
        print("No significant correlations found for relationship analysis")
        return
    
    # Extract key features of interest
    # Audio: duration, tempo, spectral features
    key_audio_features = [col for col in merged_data.columns if col.startswith('audio_') and 
                          any(term in col for term in ['duration', 'tempo', 'spectral', 'energy', 'harmonic'])]
    
    # Text: sentiment, lexical diversity, word count
    key_text_features = [col for col in merged_data.columns if col.startswith('text_') and 
                         any(term in col for term in ['polarity', 'subjectivity', 'diversity', 'count', 'unique'])]
    
    # Find correlations involving these key features
    key_correlations = corr_df[
        (corr_df['feature1'].isin(key_audio_features) & corr_df['feature2'].isin(key_text_features)) |
        (corr_df['feature2'].isin(key_audio_features) & corr_df['feature1'].isin(key_text_features))
    ]
    
    if len(key_correlations) > 0:
        print(f"Found {len(key_correlations)} correlations between key audio and text features")
        print("\nMost interesting relationships:")
        
        # Get top 5 correlations
        top_correlations = key_correlations.head(5)
        for i, row in top_correlations.iterrows():
            feature1 = row['feature1'].replace('audio_', '').replace('text_', '')
            feature2 = row['feature2'].replace('audio_', '').replace('text_', '')
            direction = "positively" if row['correlation'] > 0 else "negatively"
            strength = "strongly" if abs(row['correlation']) > 0.6 else "moderately"
            
            print(f"- {feature1} is {direction} and {strength} correlated with {feature2} (r = {row['correlation']:.2f}, p = {row['p_value']:.4f})")
    else:
        print("No significant correlations found between key feature pairs")
    
    # Look for specific interesting pairings
    interesting_pairs = [
        ('audio_tempo', 'text_polarity'),  # Fast tempo vs positive sentiment
        ('audio_spectral_centroid', 'text_polarity'),  # "Brightness" vs positive sentiment
        ('audio_duration', 'text_original_word_count'),  # Anthem length vs lyric length
        ('audio_harmonic_ratio', 'text_lexical_diversity')  # Musical complexity vs linguistic complexity
    ]
    
    print("\nSpecific feature relationships of interest:")
    for pair in interesting_pairs:
        # Check if both features exist in the dataset
        if pair[0] not in merged_data.columns or pair[1] not in merged_data.columns:
            print(f"- {pair[0].replace('audio_', '').replace('text_', '')} vs {pair[1].replace('audio_', '').replace('text_', '')}: Features not found in the dataset")
            continue
            
        # Find pair in correlations (in either order)
        if corr_df is not None:
            pair_corr = corr_df[
                ((corr_df['feature1'] == pair[0]) & (corr_df['feature2'] == pair[1])) |
                ((corr_df['feature1'] == pair[1]) & (corr_df['feature2'] == pair[0]))
            ]
            
            if len(pair_corr) > 0:
                row = pair_corr.iloc[0]
                print(f"- {pair[0].replace('audio_', '').replace('text_', '')} vs {pair[1].replace('audio_', '').replace('text_', '')}:")
                print(f"  Correlation: r = {row['correlation']:.2f}, p = {row['p_value']:.4f}, n = {row['sample_size']}")
                
                # Check if it's a strong relationship
                if abs(row['correlation']) > 0.4:
                    if row['correlation'] > 0:
                        print("  This suggests countries with higher/greater values in one feature tend to have higher/greater values in the other")
                    else:
                        print("  This suggests countries with higher/greater values in one feature tend to have lower values in the other")
            else:
                # If no significant correlation, calculate it anyway just to report
                valid_data = merged_data[[pair[0], pair[1]]].dropna()
                if len(valid_data) >= 10:
                    try:
                        corr, p = pearsonr(valid_data[pair[0]], valid_data[pair[1]])
                        print(f"- {pair[0].replace('audio_', '').replace('text_', '')} vs {pair[1].replace('audio_', '').replace('text_', '')}:")
                        print(f"  Correlation: r = {corr:.2f}, p = {p:.4f}, n = {len(valid_data)}")
                        print("  This relationship is not statistically significant")
                    except Exception as e:
                        print(f"- {pair[0].replace('audio_', '').replace('text_', '')} vs {pair[1].replace('audio_', '').replace('text_', '')}: Error calculating correlation: {e}")
                else:
                    print(f"- {pair[0].replace('audio_', '').replace('text_', '')} vs {pair[1].replace('audio_', '').replace('text_', '')}: Insufficient data for analysis")
        else:
            # If correlation dataframe is None, calculate directly
            valid_data = merged_data[[pair[0], pair[1]]].dropna()
            if len(valid_data) >= 10:
                try:
                    corr, p = pearsonr(valid_data[pair[0]], valid_data[pair[1]])
                    print(f"- {pair[0].replace('audio_', '').replace('text_', '')} vs {pair[1].replace('audio_', '').replace('text_', '')}:")
                    print(f"  Correlation: r = {corr:.2f}, p = {p:.4f}, n = {len(valid_data)}")
                    sig_status = "statistically significant" if p < 0.05 else "not statistically significant"
                    print(f"  This relationship is {sig_status}")
                except Exception as e:
                    print(f"- {pair[0].replace('audio_', '').replace('text_', '')} vs {pair[1].replace('audio_', '').replace('text_', '')}: Error calculating correlation: {e}")
            else:
                print(f"- {pair[0].replace('audio_', '').replace('text_', '')} vs {pair[1].replace('audio_', '').replace('text_', '')}: Insufficient data for analysis")

# Main function to run the analysis
def main():
    print("Starting multimodal correlation analysis...")
    
    # Load data from the two modalities
    data = load_data()
    
    # Merge datasets by country code
    merged_data = merge_datasets(data)
    
    if merged_data is not None:
        # Save the merged data
        merged_data.to_csv(f"{output_dir}/merged_multimodal_data.csv", index=False)
        
        # Calculate cross-modal correlations
        corr_df = calculate_cross_modal_correlations(merged_data)
        
        # Visualize top correlations
        if corr_df is not None and len(corr_df) > 0:
            visualize_top_correlations(merged_data, corr_df)
        
        # Analyze intra-modal correlations
        analyze_intra_modal_correlations(merged_data)
        
        # Identify interesting relationships
        identify_interesting_relationships(merged_data, corr_df)
        
        # Create multimodal typology
        typology_df = create_multimodal_typology(merged_data)
        
        # Analyze regional patterns
        analyze_regional_patterns(merged_data)
        
        print("\nMultimodal analysis complete! Results saved to the 'multimodal_analysis' directory.")
    else:
        print("Could not perform multimodal analysis due to insufficient data.")

if __name__ == "__main__":
    main()