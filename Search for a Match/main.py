import os
import cv2
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import resample
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from tqdm import tqdm

def extract_ball_position(video_path):
    """Extract the ball's position over time from the video."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    positions = []
    velocities = []
    prev_pos = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to isolate the ball (assuming it's bright)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (the ball)
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            
            # Calculate centroid
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                positions.append((cx, cy))
                
                # Calculate velocity
                if prev_pos:
                    vx = cx - prev_pos[0]
                    vy = cy - prev_pos[1]
                    velocities.append((vx, vy))
                else:
                    velocities.append((0, 0))
                    
                prev_pos = (cx, cy)
            else:
                # If no ball detected, use previous position
                if prev_pos:
                    positions.append(prev_pos)
                    velocities.append((0, 0))
                else:
                    positions.append((0, 0))
                    velocities.append((0, 0))
        else:
            # If no contours detected, use previous position
            if prev_pos:
                positions.append(prev_pos)
                velocities.append((0, 0))
            else:
                positions.append((0, 0))
                velocities.append((0, 0))
    
    cap.release()
    
    # Convert to arrays for easier processing
    positions = np.array(positions)
    velocities = np.array(velocities)
    
    return positions, velocities, fps

def extract_audio_features(audio_path):
    """Extract relevant features from the audio file."""
    sample_rate, audio_data = wavfile.read(audio_path)
    
    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Normalize audio
    audio_data = audio_data / np.max(np.abs(audio_data))
    
    # Extract envelope (amplitude)
    window_size = int(0.01 * sample_rate)  # 10ms window
    envelope = []
    
    for i in range(0, len(audio_data), window_size):
        chunk = audio_data[i:i+window_size]
        if len(chunk) > 0:
            envelope.append(np.max(np.abs(chunk)))
    
    # Compute spectral centroid over time
    stft_size = 1024
    hop_size = 512
    spectral_centroids = []
    
    for i in range(0, len(audio_data) - stft_size, hop_size):
        chunk = audio_data[i:i+stft_size]
        if len(chunk) == stft_size:
            magnitudes = np.abs(np.fft.rfft(chunk * np.hanning(stft_size)))
            frequencies = np.fft.rfftfreq(stft_size, 1/sample_rate)
            if np.sum(magnitudes) > 0:
                centroid = np.sum(frequencies * magnitudes) / np.sum(magnitudes)
                spectral_centroids.append(centroid)
            else:
                spectral_centroids.append(0)
    
    return np.array(envelope), np.array(spectral_centroids), sample_rate

def compute_motion_audio_correlation(positions, velocities, fps, audio_envelope, spectral_centroids, audio_sr):
    """
    Compute correlation between motion features and audio features.
    Resamples one signal to match the other's timing.
    """
    # Calculate motion energy (magnitude of velocity)
    motion_energy = np.sqrt(velocities[:, 0]**2 + velocities[:, 1]**2)
    
    # Calculate height (y-coordinate) - invert so higher position is larger value
    height = positions[:, 1].max() - positions[:, 1]
    
    # Calculate horizontal position (normalized to 0-1)
    x_position = positions[:, 0] / positions[:, 0].max() if positions[:, 0].max() > 0 else positions[:, 0]
    
    # Determine target length for resampling
    video_length = len(motion_energy) / fps
    audio_envelope_length = len(audio_envelope) * 0.01  # 10ms windows
    spectral_length = len(spectral_centroids) * (1024/audio_sr)  # stft size / sample rate
    
    # Choose the longer one as reference and resample the shorter one
    if video_length >= audio_envelope_length and video_length >= spectral_length:
        # Resample audio features to match video
        target_length = len(motion_energy)
        resampled_envelope = resample(audio_envelope, target_length)
        resampled_centroids = resample(spectral_centroids, target_length)
        resampled_motion = motion_energy
        resampled_height = height
        resampled_x_pos = x_position
    elif audio_envelope_length >= video_length and audio_envelope_length >= spectral_length:
        # Resample video and spectral features to match envelope
        target_length = len(audio_envelope)
        resampled_envelope = audio_envelope
        resampled_motion = resample(motion_energy, target_length)
        resampled_height = resample(height, target_length)
        resampled_x_pos = resample(x_position, target_length)
        resampled_centroids = resample(spectral_centroids, target_length)
    else:
        # Resample video and envelope to match spectral
        target_length = len(spectral_centroids)
        resampled_centroids = spectral_centroids
        resampled_motion = resample(motion_energy, target_length)
        resampled_height = resample(height, target_length)
        resampled_x_pos = resample(x_position, target_length)
        resampled_envelope = resample(audio_envelope, target_length)
    
    # Normalize all features
    resampled_envelope = (resampled_envelope - np.mean(resampled_envelope)) / (np.std(resampled_envelope) if np.std(resampled_envelope) > 0 else 1)
    resampled_centroids = (resampled_centroids - np.mean(resampled_centroids)) / (np.std(resampled_centroids) if np.std(resampled_centroids) > 0 else 1)
    resampled_motion = (resampled_motion - np.mean(resampled_motion)) / (np.std(resampled_motion) if np.std(resampled_motion) > 0 else 1)
    resampled_height = (resampled_height - np.mean(resampled_height)) / (np.std(resampled_height) if np.std(resampled_height) > 0 else 1)
    resampled_x_pos = (resampled_x_pos - np.mean(resampled_x_pos)) / (np.std(resampled_x_pos) if np.std(resampled_x_pos) > 0 else 1)
    
    # Compute correlations
    corr_motion_envelope = np.corrcoef(resampled_motion, resampled_envelope)[0, 1]
    corr_height_envelope = np.corrcoef(resampled_height, resampled_envelope)[0, 1]
    corr_x_envelope = np.corrcoef(resampled_x_pos, resampled_envelope)[0, 1]
    corr_motion_centroid = np.corrcoef(resampled_motion, resampled_centroids)[0, 1]
    corr_height_centroid = np.corrcoef(resampled_height, resampled_centroids)[0, 1]
    
    # Replace NaN with 0
    correlations = [
        corr_motion_envelope, corr_height_envelope, corr_x_envelope,
        corr_motion_centroid, corr_height_centroid
    ]
    correlations = [0 if np.isnan(c) else c for c in correlations]
    
    # Compute DTW distance for additional measure
    dtw_distance = dtw_simplified(resampled_motion, resampled_envelope)
    
    # Create feature vector
    feature_vector = np.array(correlations + [dtw_distance])
    
    return feature_vector

def dtw_simplified(s1, s2, window=10):
    """Simplified Dynamic Time Warping to measure similarity between two signals."""
    n, m = len(s1), len(s2)
    dtw_matrix = np.ones((n+1, m+1)) * np.inf
    dtw_matrix[0, 0] = 0
    
    # Use window constraint for efficiency
    w = max(window, abs(n-m))
    
    for i in range(1, n+1):
        for j in range(max(1, i-w), min(m+1, i+w+1)):
            cost = (s1[i-1] - s2[j-1])**2
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],    # insertion
                dtw_matrix[i, j-1],    # deletion
                dtw_matrix[i-1, j-1]   # match
            )
    
    return dtw_matrix[n, m]

def process_all_files(audio_dir, video_dir):
    """Process all files and compute similarity matrix."""
    audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
    video_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4')])
    
    # Extract features from all files
    audio_features = {}
    video_features = {}
    
    print("Extracting audio features...")
    for audio_file in tqdm(audio_files):
        file_path = os.path.join(audio_dir, audio_file)
        envelope, centroids, sr = extract_audio_features(file_path)
        audio_features[audio_file] = (envelope, centroids, sr)
    
    print("Extracting video features...")
    for video_file in tqdm(video_files):
        file_path = os.path.join(video_dir, video_file)
        positions, velocities, fps = extract_ball_position(file_path)
        video_features[video_file] = (positions, velocities, fps)
    
    # Compute correlation matrix
    similarity_matrix = np.zeros((len(audio_files), len(video_files)))
    
    print("Computing correlations...")
    for i, audio_file in enumerate(tqdm(audio_files)):
        audio_envelope, audio_centroids, audio_sr = audio_features[audio_file]
        
        for j, video_file in enumerate(video_files):
            positions, velocities, fps = video_features[video_file]
            
            # Compute correlation
            feature_vector = compute_motion_audio_correlation(
                positions, velocities, fps,
                audio_envelope, audio_centroids, audio_sr
            )
            
            # Use a weighted sum of correlations as similarity
            similarity = np.mean([f for f in feature_vector[:-1]]) - 0.2 * feature_vector[-1]
            similarity_matrix[i, j] = similarity
    
    return audio_files, video_files, similarity_matrix

def find_best_matches(audio_files, video_files, similarity_matrix):
    """
    Find the best matches using the Hungarian algorithm.
    If not available, use a greedy approach.
    """
    try:
        from scipy.optimize import linear_sum_assignment
        
        # Hungarian algorithm for optimal assignment
        # We want to maximize similarity, so we negate the similarity matrix
        row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
        
        matches = [(audio_files[i], video_files[j]) for i, j in zip(row_ind, col_ind)]
        return matches
    
    except ImportError:
        # Fallback to greedy approach
        matches = []
        used_audio = set()
        used_video = set()
        
        # Create a flattened list of (similarity, audio_idx, video_idx)
        flat_similarities = []
        for i in range(similarity_matrix.shape[0]):
            for j in range(similarity_matrix.shape[1]):
                flat_similarities.append((similarity_matrix[i, j], i, j))
        
        # Sort by similarity (highest first)
        flat_similarities.sort(reverse=True)
        
        # Assign greedily
        for sim, i, j in flat_similarities:
            if i not in used_audio and j not in used_video:
                matches.append((audio_files[i], video_files[j]))
                used_audio.add(i)
                used_video.add(j)
                
                if len(matches) == min(len(audio_files), len(video_files)):
                    break
        
        return matches

def save_results(matches, output_file):
    """Save the matches to a CSV file."""
    df = pd.DataFrame(matches, columns=['audio_filename', 'video_filename'])
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

def visualize_matches(audio_files, video_files, similarity_matrix, matches, output_file=None):
    """Visualize the similarity matrix and the chosen matches."""
    plt.figure(figsize=(12, 10))
    plt.imshow(similarity_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='Similarity')
    plt.xlabel('Video Files')
    plt.ylabel('Audio Files')
    
    # Highlight the chosen matches
    match_indices = []
    for audio_file, video_file in matches:
        i = audio_files.index(audio_file)
        j = video_files.index(video_file)
        match_indices.append((i, j))
        plt.plot(j, i, 'rx', markersize=10)
    
    plt.title('Audio-Video Similarity Matrix with Matches')
    
    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()

def main():
    # Directories
    audio_dir = 'Search for a Match/dataset/audio_only'
    video_dir = 'Search for a Match/dataset/video_only'
    output_file = 'Search for a Match/dataset/submit_solution_mapping.csv'
    
    # Process all files
    audio_files, video_files, similarity_matrix = process_all_files(audio_dir, video_dir)
    
    # Find the best matches
    matches = find_best_matches(audio_files, video_files, similarity_matrix)
    
    # Visualize the results
    visualize_matches(audio_files, video_files, similarity_matrix, matches, 'similarity_matrix.png')
    
    # Save results
    save_results(matches, output_file)

if __name__ == "__main__":
    main()