import os
train_normal_file_names = []
train_abnormal_file_names = []
test_file_names = []
test_abnormal_file_names = []
test_normal_file_names = []

for dirname, _, filenames in os.walk('./isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_abnormal/v2.0.0/edf/train/normal'):
    for filename in filenames:
        train_normal_file_names.append(os.path.join(dirname, filename))
for dirname, _, filenames in os.walk('./isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_abnormal/v2.0.0/edf/train/abnormal'):
    for filename in filenames:
        train_abnormal_file_names.append(os.path.join(dirname, filename))
for dirname, _, filenames in os.walk('./isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_abnormal/v2.0.0/edf/eval/normal'):
    for filename in filenames:
        test_normal_file_names.append(os.path.join(dirname, filename))
for dirname, _, filenames in os.walk('./isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_abnormal/v2.0.0/edf/eval/abnormal'):
    for filename in filenames:
        test_abnormal_file_names.append(os.path.join(dirname, filename))
        
!mkdir drive/MyDrive/Data
!mkdir drive/MyDrive/Data/abnormal/
!mkdir drive/MyDrive/Data/normal/
!mkdir drive/MyDrive/Data/abnormal/train/
!mkdir drive/MyDrive/Data/normal/train/
!mkdir drive/MyDrive/Data/abnormal/test/
!mkdir drive/MyDrive/Data/normal/test/

from tqdm import tqdm

for file_path in tqdm(train_normal_file_names):
    !mv $file_path drive/MyDrive/Data/normal/train/
for file_path in tqdm(train_abnormal_file_names):
    !mv $file_path drive/MyDrive/Data/abnormal/train/
for file_path in tqdm(test_normal_file_names):
    !mv $file_path drive/MyDrive/Data/normal/test/
for file_path in tqdm(test_abnormal_file_names):
    !mv $file_path drive/MyDrive/Data/abnormal/test/
