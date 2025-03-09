import os
import json
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pydub import AudioSegment
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Base directory where dataset is stored
BASE_DIR = 'Audio_Dataset_Collection/radiowave_dataset'
AUDIO_DIR = os.path.join(BASE_DIR, 'audio')
METADATA_DIR = os.path.join(BASE_DIR, 'metadata')
ANALYSIS_DIR = os.path.join(BASE_DIR, 'analysis')

# Create analysis directory if it doesn't exist
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, 'logs', 'analysis.log')),
        logging.StreamHandler()
    ]
)

def load_dataset_metadata():
    """
    Load the dataset summary and all file metadata.
    
    Returns:
        tuple: (dataset_summary, list_of_file_metadata)
    """
    # Load dataset summary
    summary_path = os.path.join(BASE_DIR, 'dataset_summary.json')
    if not os.path.exists(summary_path):
        logging.error(f"Dataset summary file not found: {summary_path}")
        return None, []
    
    with open(summary_path, 'r') as f:
        dataset_summary = json.load(f)
    
    # Load individual file metadata
    file_metadata = []
    for filename in os.listdir(METADATA_DIR):
        if filename.endswith('.json'):
            with open(os.path.join(METADATA_DIR, filename), 'r') as f:
                metadata = json.load(f)
                # Add the corresponding audio filename
                audio_filename = os.path.splitext(filename)[0] + '.' + metadata['format']
                metadata['audio_filename'] = audio_filename
                file_metadata.append(metadata)
    
    logging.info(f"Loaded metadata for {len(file_metadata)} audio files")
    return dataset_summary, file_metadata

def extract_audio_features(audio_path):
    """
    Extract audio features using librosa.
    
    Args:
        audio_path (str): Path to the audio file
        
    Returns:
        dict: Dictionary of audio features
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=None)
        
        # Basic features
        duration = librosa.get_duration(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        
        # Rhythm features
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfccs, axis=1)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_means = np.mean(chroma, axis=1)
        
        # Energy features
        rms = np.mean(librosa.feature.rms(y=y))
        
        # Compute harmonic and percussive components
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        harmonic_ratio = np.sum(y_harmonic**2) / (np.sum(y_harmonic**2) + np.sum(y_percussive**2) + 1e-8)
        
        # Pack everything into a dictionary
        features = {
            'duration': duration,
            'tempo': tempo,
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth,
            'spectral_rolloff': spectral_rolloff,
            'zero_crossing_rate': zero_crossing_rate,
            'rms_energy': rms,
            'harmonic_ratio': harmonic_ratio
        }
        
        # Add MFCCs
        for i, mfcc in enumerate(mfcc_means):
            features[f'mfcc_{i+1}'] = mfcc
            
        # Add chroma features
        for i, chroma_val in enumerate(chroma_means):
            features[f'chroma_{i+1}'] = chroma_val
            
        return features
        
    except Exception as e:
        logging.error(f"Error extracting features from {audio_path}: {e}")
        return None

def classify_speech_vs_music(features):
    """
    Classify audio as speech or music based on features.
    
    Args:
        features (dict): Audio features
        
    Returns:
        str: 'speech', 'music', or 'mixed'
    """
    # Simple heuristic approach based on audio features
    # Speech typically has:
    # - Higher zero crossing rate
    # - Lower harmonic ratio
    # - Lower spectral centroid
    # - Higher spectral flatness
    
    # These thresholds are approximations and would need to be tuned
    # based on the specific dataset
    if features['zero_crossing_rate'] > 0.05 and features['harmonic_ratio'] < 0.4:
        return 'speech'
    elif features['spectral_centroid'] > 2000 and features['harmonic_ratio'] > 0.6:
        return 'music'
    else:
        return 'mixed'

def analyze_dataset():
    """
    Perform comprehensive analysis on the RadioWave dataset.
    
    Returns:
        pandas.DataFrame: Analysis results
    """
    # Load metadata
    dataset_summary, file_metadata = load_dataset_metadata()
    if not dataset_summary:
        return None
    
    # Create a dataframe to store analysis results
    analysis_results = []
    
    # Analyze each audio file
    for metadata in file_metadata:
        audio_path = os.path.join(AUDIO_DIR, metadata['audio_filename'])
        
        if not os.path.exists(audio_path):
            logging.warning(f"Audio file not found: {audio_path}")
            continue
        
        logging.info(f"Analyzing {metadata['audio_filename']}")
        
        # Extract features
        features = extract_audio_features(audio_path)
        if not features:
            continue
        
        # Classify speech vs. music
        content_type = classify_speech_vs_music(features)
        
        # Combine metadata and features
        result = {
            'filename': metadata['audio_filename'],
            'station_name': metadata['station_name'],
            'country': metadata['country'],
            'tags': ', '.join(metadata['tags']) if metadata['tags'] else '',
            'content_type': content_type,
            'recorded_duration': metadata['duration'],
            'actual_duration': features['duration']
        }
        
        # Add features to result
        result.update(features)
        
        analysis_results.append(result)
    
    # Convert to DataFrame
    if analysis_results:
        df = pd.DataFrame(analysis_results)
        logging.info(f"Analysis completed for {len(df)} audio files")
        
        # Save results to CSV
        csv_path = os.path.join(ANALYSIS_DIR, 'analysis_results.csv')
        df.to_csv(csv_path, index=False)
        logging.info(f"Analysis results saved to {csv_path}")
        
        return df
    else:
        logging.error("No analysis results produced")
        return None

def cluster_audio_content(df):
    """
    Cluster audio files based on their features.
    
    Args:
        df (pandas.DataFrame): Analysis results
        
    Returns:
        pandas.DataFrame: DataFrame with cluster assignments
    """
    if df is None or len(df) < 3:  # Need at least 3 samples for meaningful clustering
        logging.error("Not enough data for clustering")
        return None
    
    # Select features for clustering
    feature_cols = [
        'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
        'zero_crossing_rate', 'rms_energy', 'harmonic_ratio',
        'tempo'
    ]
    
    # Add MFCCs if available
    mfcc_cols = [col for col in df.columns if col.startswith('mfcc_')]
    feature_cols.extend(mfcc_cols[:5])  # Use first 5 MFCCs
    
    # Make sure all feature columns exist
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    if not feature_cols:
        logging.error("No valid feature columns found for clustering")
        return None
    
    # Extract features
    X = df[feature_cols].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine optimal number of clusters (2-5)
    inertia = []
    k_range = range(2, min(6, len(df) + 1))
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
    
    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.savefig(os.path.join(ANALYSIS_DIR, 'cluster_elbow.png'))
    plt.close()
    
    # Choose number of clusters based on the elbow method
    # Simple heuristic: use the "elbow" point, or default to 3
    k = 3  # Default
    if len(inertia) > 2:
        # Calculate differences in inertia
        diffs = [inertia[i] - inertia[i+1] for i in range(len(inertia)-1)]
        
        # Find the elbow point where the rate of improvement slows
        if len(diffs) > 1 and diffs[0] > 2 * diffs[1]:
            k = k_range[1]  # Pick the second point
        else:
            k = k_range[0]  # Default to first point
    
    # Perform clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add cluster assignments to DataFrame
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = clusters
    
    # Create PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'PCA1': X_pca[:, 0],
        'PCA2': X_pca[:, 1],
        'Cluster': clusters,
        'Content': df['content_type']
    })
    
    # Plot clusters
    plt.figure(figsize=(12, 8))
    
    # Plot clusters
    sns.scatterplot(
        data=plot_df,
        x='PCA1',
        y='PCA2',
        hue='Cluster',
        style='Content',
        palette='viridis',
        s=100,
        alpha=0.7
    )
    
    # Add station names as annotations
    for i, row in df.iterrows():
        plt.annotate(
            row['station_name'][:15] + '...' if len(row['station_name']) > 15 else row['station_name'],
            (X_pca[i, 0], X_pca[i, 1]),
            fontsize=8,
            alpha=0.7,
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    plt.title('Audio Content Clusters')
    plt.xlabel(f'PCA1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PCA2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, 'content_clusters.png'))
    plt.close()
    
    # Save to CSV
    df_with_clusters.to_csv(os.path.join(ANALYSIS_DIR, 'clustered_results.csv'), index=False)
    
    # Analyze cluster characteristics
    cluster_analysis = df_with_clusters.groupby('cluster').agg({
        'content_type': pd.Series.mode,
        'spectral_centroid': 'mean',
        'harmonic_ratio': 'mean',
        'zero_crossing_rate': 'mean',
        'rms_energy': 'mean',
        'tempo': 'mean'
    }).reset_index()
    
    # Save cluster analysis
    cluster_analysis.to_csv(os.path.join(ANALYSIS_DIR, 'cluster_analysis.csv'), index=False)
    
    return df_with_clusters

def visualize_audio_features(df):
    """
    Create visualizations of audio features.
    
    Args:
        df (pandas.DataFrame): Analysis results
    """
    if df is None or len(df) == 0:
        logging.error("No data available for visualization")
        return
    
    # 1. Distribution of content types
    plt.figure(figsize=(10, 6))
    content_counts = df['content_type'].value_counts()
    content_counts.plot(kind='bar', color=['skyblue', 'salmon', 'lightgreen'])
    plt.title('Distribution of Audio Content Types')
    plt.xlabel('Content Type')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, 'content_type_distribution.png'))
    plt.close()
    
    # 2. Spectral features by content type
    plt.figure(figsize=(14, 8))
    
    feature_cols = ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'harmonic_ratio']
    
    # Normalize features for better comparison
    df_norm = df.copy()
    for col in feature_cols:
        if col in df.columns:
            df_norm[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-8)
    
    # Create boxplot
    feature_cols = [col for col in feature_cols if col in df.columns]
    if feature_cols:
        sns.boxplot(
            data=pd.melt(
                df_norm,
                id_vars=['content_type'],
                value_vars=feature_cols,
                var_name='Feature',
                value_name='Normalized Value'
            ),
            x='Feature',
            y='Normalized Value',
            hue='content_type'
        )
        plt.title('Spectral Features by Content Type (Normalized)')
        plt.ylabel('Normalized Value')
        plt.tight_layout()
        plt.savefig(os.path.join(ANALYSIS_DIR, 'spectral_features_by_content.png'))
        plt.close()
    
    # 3. Country distribution
    plt.figure(figsize=(12, 6))
    country_counts = df['country'].value_counts().head(10)  # Top 10 countries
    country_counts.plot(kind='bar', color='lightblue')
    plt.title('Top 10 Countries of Origin')
    plt.xlabel('Country')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, 'country_distribution.png'))
    plt.close()
    
    # 4. Correlation heatmap of audio features
    plt.figure(figsize=(14, 12))
    
    # Select numerical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude specific columns
    exclude_cols = ['recorded_duration', 'actual_duration', 'cluster']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols and not col.startswith('mfcc_') and not col.startswith('chroma_')]
    
    # Create correlation matrix
    if len(feature_cols) > 1:
        corr = df[feature_cols].corr()
        
        # Generate heatmap
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation of Audio Features')
        plt.tight_layout()
        plt.savefig(os.path.join(ANALYSIS_DIR, 'feature_correlation.png'))
        plt.close()
    
    # 5. Tag word cloud
    try:
        from wordcloud import WordCloud
        
        # Combine all tags
        all_tags = ' '.join(df['tags'].fillna(''))
        
        if all_tags.strip():
            # Generate word cloud
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                max_words=100,
                contour_width=3,
                contour_color='steelblue'
            ).generate(all_tags)
            
            # Plot
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Radio Station Tag Cloud')
            plt.tight_layout()
            plt.savefig(os.path.join(ANALYSIS_DIR, 'tag_wordcloud.png'))
            plt.close()
    except ImportError:
        logging.warning("WordCloud package not installed, skipping tag cloud visualization")

def check_dependencies():
    """
    Check if all required dependencies are installed.
    
    Returns:
        bool: True if all dependencies are available, False otherwise
    """
    missing_deps = []
    
    try:
        import librosa
    except ImportError:
        missing_deps.append('librosa')
    
    try:
        import pandas
    except ImportError:
        missing_deps.append('pandas')
    
    try:
        import matplotlib
    except ImportError:
        missing_deps.append('matplotlib')
    
    try:
        import seaborn
    except ImportError:
        missing_deps.append('seaborn')
    
    try:
        import numpy
    except ImportError:
        missing_deps.append('numpy')
    
    try:
        import sklearn
    except ImportError:
        missing_deps.append('scikit-learn')
    
    try:
        from pydub import AudioSegment
    except ImportError:
        missing_deps.append('pydub')
    
    if missing_deps:
        logging.error(f"Missing dependencies: {', '.join(missing_deps)}")
        logging.error("Please install the missing dependencies with: pip install " + " ".join(missing_deps))
        return False
    
    return True

def generate_insights_report(df):
    """
    Generate a text report with insights from the analysis.
    
    Args:
        df (pandas.DataFrame): Analysis results
    """
    if df is None or len(df) == 0:
        logging.error("No data available for insights report")
        return
    
    # Open report file
    report_path = os.path.join(ANALYSIS_DIR, 'insights_report.md')
    with open(report_path, 'w') as f:
        f.write("# RadioWave Dataset Analysis Insights\n\n")
        
        # Dataset overview
        f.write("## Dataset Overview\n\n")
        f.write(f"- Total audio files analyzed: {len(df)}\n")
        f.write(f"- Average duration: {df['actual_duration'].mean():.2f} seconds\n")
        f.write(f"- Content types distribution:\n")
        
        content_counts = df['content_type'].value_counts()
        for content_type, count in content_counts.items():
            percentage = 100 * count / len(df)
            f.write(f"  - {content_type}: {count} ({percentage:.1f}%)\n")
        
        # Country distribution
        f.write("\n## Geographic Distribution\n\n")
        country_counts = df['country'].value_counts()
        f.write("Top 5 countries:\n\n")
        
        for country, count in country_counts.head(5).items():
            percentage = 100 * count / len(df)
            f.write(f"- {country}: {count} ({percentage:.1f}%)\n")
        
        # Audio characteristics
        f.write("\n## Audio Characteristics\n\n")
        
        # Speech vs. music comparison
        f.write("### Speech vs. Music Comparison\n\n")
        
        features_to_compare = [
            'spectral_centroid', 'spectral_bandwidth', 
            'zero_crossing_rate', 'harmonic_ratio', 'tempo'
        ]
        
        features_to_compare = [f for f in features_to_compare if f in df.columns]
        
        if len(features_to_compare) > 0 and 'content_type' in df.columns:
            f.write("| Feature | Speech | Music | Mixed |\n")
            f.write("|---------|--------|-------|-------|\n")
            
            speech_df = df[df['content_type'] == 'speech']
            music_df = df[df['content_type'] == 'music']
            mixed_df = df[df['content_type'] == 'mixed']
            
            for feature in features_to_compare:
                speech_val = speech_df[feature].mean() if len(speech_df) > 0 else float('nan')
                music_val = music_df[feature].mean() if len(music_df) > 0 else float('nan')
                mixed_val = mixed_df[feature].mean() if len(mixed_df) > 0 else float('nan')
                
                f.write(f"| {feature} | {speech_val:.2f} | {music_val:.2f} | {mixed_val:.2f} |\n")
        
        # Cluster analysis
        if 'cluster' in df.columns:
            f.write("\n## Cluster Analysis\n\n")
            
            cluster_counts = df['cluster'].value_counts().sort_index()
            for cluster, count in cluster_counts.items():
                percentage = 100 * count / len(df)
                cluster_df = df[df['cluster'] == cluster]
                
                # Most common content type in this cluster
                most_common_content = cluster_df['content_type'].mode()[0] if len(cluster_df) > 0 else "Unknown"
                
                # Dominant features for this cluster
                f.write(f"### Cluster {cluster} ({count} samples, {percentage:.1f}%)\n\n")
                f.write(f"- Dominant content type: {most_common_content}\n")
                f.write("- Characteristic features:\n")
                
                for feature in features_to_compare:
                    cluster_mean = cluster_df[feature].mean()
                    overall_mean = df[feature].mean()
                    difference = ((cluster_mean - overall_mean) / overall_mean) * 100
                    
                    if abs(difference) > 10:  # Only report significant differences
                        direction = "higher" if difference > 0 else "lower"
                        f.write(f"  - {feature}: {abs(difference):.1f}% {direction} than average\n")
                
                f.write("\n")
        
        # Interesting observations
        f.write("## Interesting Observations\n\n")
        
        # Find stations with highest/lowest spectral features
        if 'spectral_centroid' in df.columns:
            highest_centroid = df.loc[df['spectral_centroid'].idxmax()]
            f.write(f"- Station with brightest sound (highest spectral centroid): {highest_centroid['station_name']} ({highest_centroid['spectral_centroid']:.2f})\n")
        
        if 'harmonic_ratio' in df.columns:
            highest_harmonic = df.loc[df['harmonic_ratio'].idxmax()]
            f.write(f"- Station with most harmonic content: {highest_harmonic['station_name']} ({highest_harmonic['harmonic_ratio']:.2f})\n")
        
        if 'zero_crossing_rate' in df.columns:
            highest_zcr = df.loc[df['zero_crossing_rate'].idxmax()]
            f.write(f"- Station with highest speech-like qualities (zero crossing rate): {highest_zcr['station_name']} ({highest_zcr['zero_crossing_rate']:.4f})\n")
        
        if 'tempo' in df.columns:
            highest_tempo = df.loc[df['tempo'].idxmax()]
            lowest_tempo = df.loc[df['tempo'].idxmin()]
            f.write(f"- Fastest tempo: {highest_tempo['station_name']} ({highest_tempo['tempo']:.2f} BPM)\n")
            f.write(f"- Slowest tempo: {lowest_tempo['station_name']} ({lowest_tempo['tempo']:.2f} BPM)\n")
        
        # Conclusion
        f.write("\n## Conclusion\n\n")
        f.write("The RadioWave dataset provides a diverse collection of audio samples from radio stations worldwide. ")
        f.write("The analysis reveals distinct patterns between speech and music content, with clear acoustic signatures ")
        f.write("that can be used for automatic content type classification. The clustering analysis has identified ")
        f.write("natural groupings in the data, which could be useful for applications like content-based recommendation ")
        f.write("systems, automated radio monitoring, or audio fingerprinting.\n\n")
        
        f.write("This dataset demonstrates the potential for using machine learning techniques to analyze ")
        f.write("and categorize radio content, which could have applications in media monitoring, content ")
        f.write("discovery, and automated radio programming analysis.")
    
    logging.info(f"Insights report generated: {report_path}")

def main():
    """Main function to analyze the RadioWave dataset."""
    # Check dependencies
    if not check_dependencies():
        return
    
    logging.info("Starting RadioWave dataset analysis")
    
    # Analyze dataset
    df = analyze_dataset()
    
    if df is not None and len(df) > 0:
        # Perform clustering
        df_clustered = cluster_audio_content(df)
        
        # Create visualizations
        visualize_audio_features(df_clustered if df_clustered is not None else df)
        
        # Generate insights report
        generate_insights_report(df_clustered if df_clustered is not None else df)
        
        logging.info("Analysis completed successfully")
        logging.info(f"Results saved to {os.path.abspath(ANALYSIS_DIR)}")
    else:
        logging.error("Analysis failed - no data to analyze")

if __name__ == '__main__':
    main()