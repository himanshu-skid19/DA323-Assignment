import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pydub import AudioSegment
from tqdm import tqdm
import seaborn as sns
from scipy.stats import pearsonr
import country_converter as coco

# Define the path to your anthem audio files
base_path = "data/anthems/audio"

# Function to extract features from audio files
def extract_audio_features(file_path):
    try:
        # Load the audio file
        y, sr = librosa.load(file_path, sr=None)
        
        # Basic features
        duration = librosa.get_duration(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0])
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)[0])
        
        # Chromagram
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        
        # Zero crossing rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(y)[0])
        
        # RMS energy
        rms = np.mean(librosa.feature.rms(y=y)[0])
        
        # Harmonic and percussive components
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        harmonic_ratio = np.sum(y_harmonic**2) / np.sum(y**2)
        
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
    country_names = coco.convert(codes=summary_df['country_code'].tolist(), 
                                to='name_short', 
                                not_found=None)
    summary_df['country_name'] = country_names
    
    return anthem_features, summary_df

# Function to create comparative visualizations
def create_comparative_visualizations(summary_df):
    # Create output directory
    output_dir = "anthem_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Duration distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(summary_df['duration'], kde=True)
    plt.title('Distribution of National Anthem Durations')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Count')
    plt.savefig(f"{output_dir}/duration_distribution.png")
    plt.close()
    
    # 2. Tempo distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(summary_df['tempo'], kde=True)
    plt.title('Distribution of National Anthem Tempos')
    plt.xlabel('Tempo (BPM)')
    plt.ylabel('Count')
    plt.savefig(f"{output_dir}/tempo_distribution.png")
    plt.close()
    
    # 3. Correlation matrix of features
    plt.figure(figsize=(10, 8))
    correlation_features = summary_df.drop(['country_code', 'country_name'], axis=1, errors='ignore')
    correlation_matrix = correlation_features.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Audio Features')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_matrix.png")
    plt.close()
    
    # 4. Top 10 longest and shortest anthems
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    longest = summary_df.sort_values('duration', ascending=False).head(10)
    sns.barplot(x='duration', y='country_name', data=longest)
    plt.title('Top 10 Longest National Anthems')
    plt.xlabel('Duration (seconds)')
    
    plt.subplot(2, 1, 2)
    shortest = summary_df.sort_values('duration').head(10)
    sns.barplot(x='duration', y='country_name', data=shortest)
    plt.title('Top 10 Shortest National Anthems')
    plt.xlabel('Duration (seconds)')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/anthem_durations_extremes.png")
    plt.close()
    
    # 5. Tempo vs. Duration scatter plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='tempo', y='duration', data=summary_df)
    
    # Add country labels to outlier points
    outliers = summary_df[
        (summary_df['tempo'] > summary_df['tempo'].quantile(0.95)) | 
        (summary_df['duration'] > summary_df['duration'].quantile(0.95))
    ]
    
    for _, row in outliers.iterrows():
        plt.annotate(row['country_name'], 
                    (row['tempo'], row['duration']),
                    xytext=(5, 5),
                    textcoords='offset points')
    
    plt.title('Tempo vs. Duration of National Anthems')
    plt.xlabel('Tempo (BPM)')
    plt.ylabel('Duration (seconds)')
    plt.savefig(f"{output_dir}/tempo_vs_duration.png")
    plt.close()

# Run the complete analysis
anthem_features, summary_df = analyze_anthems(base_path)

# Save summary data to CSV
summary_df.to_csv("anthem_features_summary.csv", index=False)

# Create comparative visualizations
create_comparative_visualizations(summary_df)

# Print some interesting statistics
print("\n--- Anthem Analysis Summary ---")
print(f"Total anthems analyzed: {len(summary_df)}")
print(f"Average anthem duration: {summary_df['duration'].mean():.2f} seconds")
print(f"Average anthem tempo: {summary_df['tempo'].mean():.2f} BPM")
print(f"Longest anthem: {summary_df.loc[summary_df['duration'].idxmax(), 'country_name']} ({summary_df['duration'].max():.2f} seconds)")
print(f"Shortest anthem: {summary_df.loc[summary_df['duration'].idxmin(), 'country_name']} ({summary_df['duration'].min():.2f} seconds)")
print(f"Fastest tempo: {summary_df.loc[summary_df['tempo'].idxmax(), 'country_name']} ({summary_df['tempo'].max():.2f} BPM)")
print(f"Slowest tempo: {summary_df.loc[summary_df['tempo'].idxmin(), 'country_name']} ({summary_df['tempo'].min():.2f} BPM)")