import os
import soundfile as sf
import librosa
import numpy as np

# Consistent data path
consistent_data_path = "./dataClassified/cry_data/"
if not os.path.exists(consistent_data_path):
    os.makedirs(consistent_data_path)

# ============================
# Process DonateACRY Datasets
# ============================

infant_cry = "./tempData/1/donateacry_corpus/"

# Find folders inside the infant_cry folder
folders = os.listdir(infant_cry)
print("DonateACRY folders:", folders)

# Create a list of file paths to all the .wav files inside the folders
file_paths = []
save_file_paths = []
counter = 0

for folder in folders:
    folder_path = os.path.join(infant_cry, folder)
    files = os.listdir(folder_path)
    for file in files:
        if file.endswith(".wav"):
            file_paths.append(os.path.join(folder_path, file))
            save_file_paths.append(os.path.join(consistent_data_path, f"donateacry_{counter}.wav"))
            counter += 1

# Desired settings
target_sr = 16000  # 16kHz

# Process and save DonateACRY files
for i, file_path in enumerate(file_paths):
    data, _ = librosa.load(file_path, sr=target_sr, mono=True)
    # Normalize data between -1 and 1
    data = data / np.max(np.abs(data))
    data = data.astype(np.float32)
    output_path = save_file_paths[i]
    sf.write(output_path, data, target_sr, subtype='PCM_16')
    print(f"Processed and saved: {output_path}")

# ============================
# Process Mendelyan Cry Datasets
# ============================

mendelyan_cry = "./tempData/MendelyCryData/Infants Cry Sound/Dataset"

# Find folders inside the mendelyan_cry folder
folders = os.listdir(mendelyan_cry)
print("Mendelyan folders:", folders)

# Create a list of file paths to all the .wav files inside the folders
file_paths = []
save_file_paths = []
counter = 0

for folder in folders:
    folder_path = os.path.join(mendelyan_cry, folder)
    files = os.listdir(folder_path)
    for file in files:
        if file.endswith(".wav"):
            file_paths.append(os.path.join(folder_path, file))
            save_file_paths.append(os.path.join(consistent_data_path, f"mendely_cry_{counter}.wav"))
            counter += 1

# Process and save Mendelyan Cry files
for i, file_path in enumerate(file_paths):
    data, _ = librosa.load(file_path, sr=target_sr, mono=True)
    # Normalize data between -1 and 1
    data = data / np.max(np.abs(data))
    data = data.astype(np.float32)
    output_path = save_file_paths[i]
    sf.write(output_path, data, target_sr, subtype='PCM_16')
    print(f"Processed and saved: {output_path}")


# ============================
