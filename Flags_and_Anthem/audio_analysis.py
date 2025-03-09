import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
from tqdm import tqdm
import seaborn as sns
from scipy.stats import pearsonr
import warnings

# Suppress librosa warnings for corrupted files
warnings.filterwarnings("ignore", category=UserWarning)

# Define the path to your anthem audio files
base_path = "data/anthems/audio"

# Function to extract features from audio files
def extract_audio_features(file_path):
    try:
        # Load the audio file with a timeout
        y, sr = librosa.load(file_path, sr=None, res_type='kaiser_fast')
        
        # Basic features
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Get tempo with higher tolerance for errors
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        except:
            tempo = np.nan
        
        # Spectral features - with error handling
        try:
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0])
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)[0])
        except:
            spectral_centroid = np.nan
            spectral_bandwidth = np.nan
            spectral_rolloff = np.nan
        
        # Chromagram
        try:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
        except:
            chroma_mean = np.array([np.nan] * 12)
        
        # MFCC
        try:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)
        except:
            mfcc_mean = np.array([np.nan] * 13)
        
        # Zero crossing rate
        try:
            zcr = np.mean(librosa.feature.zero_crossing_rate(y)[0])
        except:
            zcr = np.nan
        
        # RMS energy
        try:
            rms = np.mean(librosa.feature.rms(y=y)[0])
        except:
            rms = np.nan
        
        # Harmonic and percussive components
        try:
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            harmonic_ratio = np.sum(y_harmonic**2) / np.sum(y**2)
        except:
            harmonic_ratio = np.nan
        
        return {
            'duration': duration,
            'tempo': tempo,
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth,
            'spectral_rolloff': spectral_rolloff,
            'zero_crossing_rate': zcr,
            'rms_energy': rms,
            'harmonic_ratio': harmonic_ratio,
            'chroma_mean': chroma_mean,
            'mfcc_mean': mfcc_mean,
            'raw_audio': y,
            'sample_rate': sr
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Create a function to visualize audio features
def visualize_audio(country_code, features, output_dir="anthem_analysis"):
    try:
        y = features['raw_audio']
        sr = features['sample_rate']
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(14, 10))
        
        # Plot waveform
        plt.subplot(3, 1, 1)
        librosa.display.waveshow(y, sr=sr)
        plt.title(f'Waveform - {country_code}')
        
        # Plot spectrogram
        plt.subplot(3, 1, 2)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram - {country_code}')
        
        # Plot chromagram
        plt.subplot(3, 1, 3)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
        plt.colorbar()
        plt.title(f'Chromagram - {country_code}')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{country_code}_analysis.png")
        plt.close()
    except Exception as e:
        print(f"Error creating visualization for {country_code}: {e}")

# Main analysis function
def analyze_anthems(base_path):
    # Get list of all audio files
    audio_files = [f for f in os.listdir(base_path) if f.endswith('.mp3')]
    
    # Extract features for each anthem
    anthem_features = {}
    summary_data = []
    
    for audio_file in tqdm(audio_files, desc="Analyzing anthems"):
        country_code = os.path.splitext(audio_file)[0]
        file_path = os.path.join(base_path, audio_file)
        
        features = extract_audio_features(file_path)
        if features:
            anthem_features[country_code] = features
            
            # Create summary data for basic features
            summary_data.append({
                'country_code': country_code,
                'duration': features['duration'],
                'tempo': features['tempo'],
                'spectral_centroid': features['spectral_centroid'],
                'spectral_bandwidth': features['spectral_bandwidth'],
                'spectral_rolloff': features['spectral_rolloff'],
                'zero_crossing_rate': features['zero_crossing_rate'],
                'rms_energy': features['rms_energy'],
                'harmonic_ratio': features['harmonic_ratio']
            })
            
            # Generate visualizations
            visualize_audio(country_code, features)
    
    # Create a summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Add country names
    summary_df = add_country_names(summary_df)
    
    return anthem_features, summary_df

# Function to add country names (fixed to avoid the country_converter error)
def add_country_names(summary_df):
    # Manual mapping of common country codes to names (ISO 2-letter codes)
    # This replaces the country_converter library which was causing errors
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
    
    # Add the country names to the dataframe
    summary_df['country_name'] = summary_df['country_code'].map(country_map)
    
    # For any missing mappings, use the country code
    summary_df['country_name'] = summary_df['country_name'].fillna(summary_df['country_code'])
    
    return summary_df

# Function to create comparative visualizations - FIXED VERSION
def create_comparative_visualizations(summary_df):
    """Create comparative visualizations across all anthems"""
    # Create output directory
    output_dir = "anthem_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Duration distribution
    plt.figure(figsize=(12, 6))
    duration_data = summary_df['duration'].dropna()
    if len(duration_data) > 1:  # Check if we have enough data points
        sns.histplot(duration_data, kde=True)
        plt.title('Distribution of National Anthem Durations')
        plt.xlabel('Duration (seconds)')
        plt.ylabel('Count')
    else:
        plt.text(0.5, 0.5, "Insufficient data for duration histogram", 
                 ha='center', va='center', fontsize=14)
    plt.savefig(f"{output_dir}/duration_distribution.png")
    plt.close()
    
    # 2. Tempo distribution
    plt.figure(figsize=(12, 6))
    tempo_data = summary_df['tempo'].dropna()
    if len(tempo_data) > 1:  # Make sure we have multiple tempo values
        sns.histplot(tempo_data, kde=True)
        plt.title('Distribution of National Anthem Tempos')
        plt.xlabel('Tempo (BPM)')
        plt.ylabel('Count')
    else:
        plt.text(0.5, 0.5, "Insufficient tempo data for histogram", 
                 ha='center', va='center', fontsize=14)
    plt.savefig(f"{output_dir}/tempo_distribution.png")
    plt.close()
    
    # 3. Correlation matrix of features
    plt.figure(figsize=(10, 8))
    correlation_features = summary_df.drop(['country_code', 'country_name'], axis=1, errors='ignore')
    # Drop rows with NaN values for correlation calculation
    correlation_features = correlation_features.dropna()
    if len(correlation_features) > 1:  # Check if we have enough data
        correlation_matrix = correlation_features.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix of Audio Features')
    else:
        plt.text(0.5, 0.5, "Insufficient data for correlation matrix", 
                 ha='center', va='center', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_matrix.png")
    plt.close()
    
    # 4. Top 10 longest and shortest anthems
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    if len(summary_df) >= 10:  # Check if we have enough anthems
        longest = summary_df.sort_values('duration', ascending=False).head(10)
        sns.barplot(x='duration', y='country_name', data=longest)
        plt.title('Top 10 Longest National Anthems')
        plt.xlabel('Duration (seconds)')
    else:
        plt.text(0.5, 0.5, "Not enough anthems for top 10 comparison", 
                 ha='center', va='center', fontsize=14)
    
    plt.subplot(2, 1, 2)
    if len(summary_df) >= 10:  # Check if we have enough anthems
        shortest = summary_df.sort_values('duration').head(10)
        sns.barplot(x='duration', y='country_name', data=shortest)
        plt.title('Top 10 Shortest National Anthems')
        plt.xlabel('Duration (seconds)')
    else:
        plt.text(0.5, 0.5, "Not enough anthems for top 10 comparison", 
                 ha='center', va='center', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/anthem_durations_extremes.png")
    plt.close()
    
    # 5. Tempo vs. Duration scatter plot
    plt.figure(figsize=(10, 8))
    # Filter data to only include rows with valid tempo and duration
    scatter_data = summary_df.dropna(subset=['tempo', 'duration'])
    
    if len(scatter_data) > 1:  # Check if we have enough data points
        sns.scatterplot(x='tempo', y='duration', data=scatter_data)
        
        # Add country labels to outlier points (if we have enough data)
        if len(scatter_data) >= 20:  # Only identify outliers if we have sufficient data
            outliers = scatter_data[
                ((scatter_data['tempo'] > scatter_data['tempo'].quantile(0.95)) | 
                (scatter_data['duration'] > scatter_data['duration'].quantile(0.95)))
            ]
            
            for _, row in outliers.iterrows():
                plt.annotate(row['country_name'], 
                            (row['tempo'], row['duration']),
                            xytext=(5, 5),
                            textcoords='offset points')
        
        plt.title('Tempo vs. Duration of National Anthems')
        plt.xlabel('Tempo (BPM)')
        plt.ylabel('Duration (seconds)')
    else:
        plt.text(0.5, 0.5, "Insufficient data for tempo vs. duration scatter plot", 
                 ha='center', va='center', fontsize=14)
    
    plt.savefig(f"{output_dir}/tempo_vs_duration.png")
    plt.close()

# Additional analysis for audio characteristics - FIXED VERSION
def analyze_audio_characteristics(summary_df):
    output_dir = "anthem_characteristics"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Spectral centroid distribution (brightness)
    plt.figure(figsize=(12, 6))
    spectral_data = summary_df['spectral_centroid'].dropna()
    if len(spectral_data) > 1:  # Check if we have enough data points
        sns.histplot(spectral_data, kde=True)
        plt.title('Distribution of Spectral Centroid Values (Brightness)')
        plt.xlabel('Spectral Centroid')
        plt.ylabel('Count')
    else:
        plt.text(0.5, 0.5, "Insufficient data for spectral centroid histogram", 
                ha='center', va='center', fontsize=14)
    plt.savefig(f"{output_dir}/spectral_centroid_distribution.png")
    plt.close()
    
    # 2. Top 10 "brightest" and "darkest" anthems
    plt.figure(figsize=(12, 10))
    
    # Get non-NaN data
    valid_spectral_df = summary_df.dropna(subset=['spectral_centroid'])
    
    if len(valid_spectral_df) >= 10:  # Check if we have enough data
        plt.subplot(2, 1, 1)
        brightest = valid_spectral_df.sort_values('spectral_centroid', ascending=False).head(10)
        sns.barplot(x='spectral_centroid', y='country_name', data=brightest)
        plt.title('Top 10 "Brightest" National Anthems (High Spectral Centroid)')
        
        plt.subplot(2, 1, 2)
        darkest = valid_spectral_df.sort_values('spectral_centroid').head(10)
        sns.barplot(x='spectral_centroid', y='country_name', data=darkest)
        plt.title('Top 10 "Darkest" National Anthems (Low Spectral Centroid)')
    else:
        plt.text(0.5, 0.5, "Insufficient data for spectral centroid comparison", 
                ha='center', va='center', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/spectral_centroid_extremes.png")
    plt.close()
    
    # 3. Energy vs Harmonic Ratio
    plt.figure(figsize=(10, 8))
    valid_data = summary_df.dropna(subset=['harmonic_ratio', 'rms_energy'])
    
    if len(valid_data) > 1:  # Check if we have enough data points
        sns.scatterplot(x='harmonic_ratio', y='rms_energy', data=valid_data)
        
        if len(valid_data) >= 20:  # Only identify outliers if we have sufficient data
            outliers = valid_data[
                ((valid_data['harmonic_ratio'] > valid_data['harmonic_ratio'].quantile(0.95)) | 
                (valid_data['rms_energy'] > valid_data['rms_energy'].quantile(0.95)))
            ]
            
            for _, row in outliers.iterrows():
                plt.annotate(row['country_name'], 
                            (row['harmonic_ratio'], row['rms_energy']),
                            xytext=(5, 5),
                            textcoords='offset points')
        
        plt.title('Energy vs Harmonic Ratio of National Anthems')
        plt.xlabel('Harmonic Ratio (more melodic →)')
        plt.ylabel('RMS Energy (louder →)')
    else:
        plt.text(0.5, 0.5, "Insufficient data for energy vs. harmonic ratio scatter plot", 
                ha='center', va='center', fontsize=14)
    
    plt.savefig(f"{output_dir}/energy_vs_harmonic.png")
    plt.close()

# Run the complete analysis - FIXED VERSION
if __name__ == "__main__":
    anthem_features, summary_df = analyze_anthems(base_path)

    # Save summary data to CSV
    summary_df.to_csv("anthem_features_summary.csv", index=False)

    # Create visualizations
    create_comparative_visualizations(summary_df)
    analyze_audio_characteristics(summary_df)

    # Print some interesting statistics
    print("\n--- Anthem Analysis Summary ---")
    print(f"Total anthems analyzed: {len(summary_df)}")
    print(f"Average anthem duration: {summary_df['duration'].mean():.2f} seconds")

    # Handle tempo statistics safely
    if summary_df['tempo'].notna().any():
        print(f"Average anthem tempo: {summary_df['tempo'].mean():.2f} BPM")
        print(f"Number of anthems with detected tempo: {summary_df['tempo'].notna().sum()} out of {len(summary_df)}")
    else:
        print("No valid tempo data was extracted from the anthems")

    # Identify longest and shortest anthems
    if not summary_df.empty:
        longest_idx = summary_df['duration'].idxmax()
        shortest_idx = summary_df['duration'].idxmin()
        print(f"Longest anthem: {summary_df.loc[longest_idx, 'country_name']} ({summary_df.loc[longest_idx, 'duration']:.2f} seconds)")
        print(f"Shortest anthem: {summary_df.loc[shortest_idx, 'country_name']} ({summary_df.loc[shortest_idx, 'duration']:.2f} seconds)")

    # Handle tempo information safely
    if summary_df['tempo'].notna().sum() >= 2:
        tempo_max_idx = summary_df['tempo'].idxmax()
        tempo_min_idx = summary_df.loc[summary_df['tempo'].notna(), 'tempo'].idxmin()
        print(f"Fastest tempo: {summary_df.loc[tempo_max_idx, 'country_name']} ({summary_df.loc[tempo_max_idx, 'tempo']:.2f} BPM)")
        print(f"Slowest tempo: {summary_df.loc[tempo_min_idx, 'country_name']} ({summary_df.loc[tempo_min_idx, 'tempo']:.2f} BPM)")

    # Additional insights with error handling
    print("\n--- Additional Audio Characteristics ---")

    if summary_df['spectral_centroid'].notna().any():
        brightest_idx = summary_df['spectral_centroid'].idxmax()
        darkest_idx = summary_df.loc[summary_df['spectral_centroid'].notna(), 'spectral_centroid'].idxmin()
        print(f"Brightest anthem: {summary_df.loc[brightest_idx, 'country_name']} (highest spectral centroid)")
        print(f"Darkest anthem: {summary_df.loc[darkest_idx, 'country_name']} (lowest spectral centroid)")
    else:
        print("No valid spectral centroid data was extracted")

    if summary_df['rms_energy'].notna().any():
        loudest_idx = summary_df['rms_energy'].idxmax()
        quietest_idx = summary_df.loc[summary_df['rms_energy'].notna(), 'rms_energy'].idxmin()
        print(f"Loudest anthem: {summary_df.loc[loudest_idx, 'country_name']} (highest RMS energy)")
        print(f"Quietest anthem: {summary_df.loc[quietest_idx, 'country_name']} (lowest RMS energy)")
    else:
        print("No valid RMS energy data was extracted")

    # Run correlations between audio features if sufficient data is available
    print("\n--- Feature Correlations ---")
    features = ['duration', 'tempo', 'spectral_centroid', 'rms_energy', 'harmonic_ratio']
    has_correlations = False

    for i in range(len(features)):
        for j in range(i+1, len(features)):
            f1, f2 = features[i], features[j]
            valid_data = summary_df[[f1, f2]].dropna()
            if len(valid_data) > 10:  # Only calculate if we have enough data points
                try:
                    corr, p = pearsonr(valid_data[f1], valid_data[f2])
                    if p < 0.05:  # Only report statistically significant correlations
                        has_correlations = True
                        correlation_strength = "strong" if abs(corr) > 0.5 else "moderate" if abs(corr) > 0.3 else "weak"
                        direction = "positive" if corr > 0 else "negative"
                        print(f"{f1} and {f2}: {direction} {correlation_strength} correlation (r={corr:.2f}, p={p:.4f})")
                except Exception as e:
                    print(f"Could not calculate correlation between {f1} and {f2}: {e}")

    if not has_correlations:
        print("No statistically significant correlations found or insufficient data for correlation analysis")

    print("\nAnalysis complete! Check the output directories for visualizations and summary files.")