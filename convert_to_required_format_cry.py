import os
import soundfile as sf
import librosa
import numpy as np
from typing import List, Tuple, Set

def create_directory(path: str) -> None:
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def get_audio_files(base_path: str, prefix: str, output_path: str, 
                    audio_extensions: Set[str] = {'.wav', '.mp3', '.flac'}) -> Tuple[List[str], List[str]]:
    """
    Recursively get paths of all audio files in the directory structure and generate save paths.
    
    Args:
        base_path: Root directory to search for audio files
        prefix: Prefix to use for output filenames
        output_path: Directory where processed files will be saved
        audio_extensions: Set of audio file extensions to look for
    
    Returns:
        Tuple of (input file paths, output file paths)
    """
    file_paths = []
    save_file_paths = []
    counter = 0
    
    # Handle depth 0 case (files in the base directory)
    if os.path.isfile(base_path):
        # If base_path is a file, process only that file
        if any(base_path.lower().endswith(ext) for ext in audio_extensions):
            file_paths.append(base_path)
            save_file_paths.append(os.path.join(output_path, f"{prefix}_{counter}.wav"))
            counter += 1
    else:
        # Recursively walk through all subdirectories
        for root, _, files in os.walk(base_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in audio_extensions):
                    file_paths.append(os.path.join(root, file))
                    save_file_paths.append(os.path.join(output_path, f"{prefix}_{counter}.wav"))
                    counter += 1
    
    print(f"Found {counter} audio files for {prefix}")
    return file_paths, save_file_paths

def process_audio_file(file_path: str, output_path: str, target_sr: int = 16000) -> None:
    """
    Process a single audio file: load, normalize, and save with target parameters.
    
    Args:
        file_path: Path to input audio file
        output_path: Path to save processed audio
        target_sr: Target sample rate (default: 16000)
    """
    try:
        # Load and resample audio
        data, _ = librosa.load(file_path, sr=target_sr, mono=True)
        
        # Normalize data between -1 and 1
        data = data / np.max(np.abs(data))
        data = data.astype(np.float32)
        
        # Save processed audio
        sf.write(output_path, data, target_sr, subtype='PCM_16')
        print(f"Processed and saved: {output_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

def process_dataset(input_dir: str, prefix: str, output_dir: str, target_sr: int = 16000) -> None:
    """
    Process all audio files in a dataset directory.
    
    Args:
        input_dir: Input directory containing the dataset
        prefix: Prefix to use for output filenames
        output_dir: Directory where processed files will be saved
        target_sr: Target sample rate (default: 16000)
    """
    file_paths, save_file_paths = get_audio_files(input_dir, prefix, output_dir)
    
    for input_path, output_path in zip(file_paths, save_file_paths):
        process_audio_file(input_path, output_path, target_sr)

def main():
    # Define paths for different categories
    consistent_data_path_cry = "./dataClassified/cry_data/"
    consistent_data_path_scream = "./dataClassified/scream_data/"
    consistent_data_path_control = "./dataClassified/control_data/"

    # Input paths
    donateacry_path = "./tempData/InfantCryData/donateacry_corpus/"
    mendelyan_path = "./tempData/MendelyCryData/Infants Cry Sound/Dataset"
    kaggle_scream_path = "./tempData/KaggleHumanScreamData/Screaming/"
    kaggle_control_path = "./tempData/KaggleHumanScreamData/NotScreaming/"
    libri_control_path = "./tempData/LibriControlData/LibriSpeech/test-clean/"
    mozilla_control_path = "./tempData/MozillaControlData/cv-corpus-20.0-delta-2024-12-06/clips"

    # Create output directories
    create_directory(consistent_data_path_cry)
    create_directory(consistent_data_path_scream)
    create_directory(consistent_data_path_control)

    # Process cry datasets
    process_dataset(donateacry_path, "donateacry", consistent_data_path_cry)
    process_dataset(mendelyan_path, "mendely_cry", consistent_data_path_cry)

    # Process scream dataset
    process_dataset(kaggle_scream_path, "kaggle_scream", consistent_data_path_scream)

    # Process control datasets
    process_dataset(kaggle_control_path, "kaggle_control", consistent_data_path_control)
    process_dataset(libri_control_path, "libri_control", consistent_data_path_control)
    process_dataset(mozilla_control_path, "mozilla_control", consistent_data_path_control)

if __name__ == "__main__":
    main()