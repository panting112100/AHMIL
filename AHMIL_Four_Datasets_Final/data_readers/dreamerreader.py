import os
import glob
import scipy.io as sio
import numpy as np
from scipy import signal

fs = 128
tasks = {'valence': 0, 'arousal': 1, 'dominance': 2}

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Apply bandpass filter to EEG data
    Args:
        data: [num_samples, num_channels]
        lowcut: low cutoff frequency (Hz)
        highcut: high cutoff frequency (Hz)
        fs: sampling rate (Hz)
        order: filter order
    Returns:
        filtered_data: same shape as input
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')

    filtered_data = np.zeros_like(data)
    for ch in range(data.shape[1]):
        filtered_data[:, ch] = signal.filtfilt(b, a, data[:, ch])
    return filtered_data

def load_dreamer_mat(mat_path):
    print(f"Loading original DREAMER.mat from {mat_path}...")
    try:
        mat = sio.loadmat(mat_path)
    except Exception as e:
        print(f"Error loading .mat file: {e}")
        return None

    dreamer = mat['DREAMER'][0, 0]
    data_struct = dreamer['Data']                   
    
    subjects_data = []
    n_subjects = data_struct.shape[1]
    
    for sub_id in range(n_subjects):
        sub_data = data_struct[0, sub_id]

        eeg_stimuli = sub_data['EEG'][0, 0]['stimuli'][0, 0] 
        eeg_baseline = sub_data['EEG'][0, 0]['baseline'][0, 0]

        score_val = sub_data['ScoreValence'][0, 0]
        score_aro = sub_data['ScoreArousal'][0, 0]
        score_dom = sub_data['ScoreDominance'][0, 0]
        
        trials_data = []
        trials_labels = []
        
        for trial_id in range(eeg_stimuli.shape[0]):
            raw_stim = eeg_stimuli[trial_id, 0]
            raw_base = eeg_baseline[trial_id, 0]

            raw_stim = bandpass_filter(raw_stim, lowcut=4.0, highcut=30.0, fs=fs)
            raw_base = bandpass_filter(raw_base, lowcut=4.0, highcut=30.0, fs=fs)

            base_mean = np.mean(raw_base, axis=0, keepdims=True)
            clean_stim = raw_stim - base_mean
            
            mean = np.mean(clean_stim, axis=0, keepdims=True)
            std = np.std(clean_stim, axis=0, keepdims=True) + 1e-8
            clean_stim = (clean_stim - mean) / std

            l_v = 1 if score_val[trial_id, 0] >= 3 else 0
            l_a = 1 if score_aro[trial_id, 0] >= 3 else 0
            l_d = 1 if score_dom[trial_id, 0] >= 3 else 0
            
            trials_data.append(clean_stim)
            trials_labels.append([l_v, l_a, l_d])
            
        subjects_data.append({'data': trials_data, 'labels': np.array(trials_labels)})
        
    return subjects_data

def split_trial(data, label, window_size, stride):
    segments = []
    labels = []
    n_samples = data.shape[0]
    start = 0
    while start + window_size <= n_samples:
        seg = data[start : start + window_size, :]
        segments.append(seg)
        labels.append(label)
        start += stride
        
    if len(segments) == 0:
        return np.array([]), np.array([])
        
    return np.array(segments), np.array(labels)

def preprocess_and_save(mat_path, save_dir, twin=2.0, stride_overlap=0.0):
    subjects_data = load_dreamer_mat(mat_path)
    if not subjects_data:
        return

    window_size = int(twin * fs)
    stride = int(window_size * (1 - stride_overlap))
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print(f"Preprocessing: Window={twin}s, Stride={stride} samples...")

    for i, sub in enumerate(subjects_data):
        sub_final_data = []
        sub_final_labels = []
        sub_lengths = []
        
        for trial_idx, (trial_data, trial_label) in enumerate(zip(sub['data'], sub['labels'])):
            segs, labs = split_trial(trial_data, trial_label, window_size, stride)
            
            if len(segs) > 0:
                sub_final_data.append(segs)
                sub_final_labels.append(labs)
                sub_lengths.append(len(segs))
        
        if len(sub_final_data) > 0:
            data_concat = np.concatenate(sub_final_data, axis=0)
            labels_concat = np.concatenate(sub_final_labels, axis=0)
            lengths_arr = np.array(sub_lengths, dtype=np.int64)

            save_name = os.path.join(save_dir, f"sub{i+1:02d}.npz")
            np.savez(save_name, 
                     data=data_concat.astype(np.float32), 
                     labels=labels_concat.astype(np.int32), 
                     lengths=lengths_arr)
                                       
            print(f"Saved {save_name}: Data {data_concat.shape}")

def load_dataset_preprocessed(datapath):
    npzfiles = glob.glob(os.path.join(datapath, "*.npz"))
    npzfiles.sort()
    
    if len(npzfiles) == 0:
        raise ValueError(f"No .npz files found in {datapath}. Please run preprocess first.")

    subjects = [os.path.basename(npz_f)[:-4] for npz_f in npzfiles]

    data = []
    labels = []
    lengths = []
    
    for npz_f in npzfiles:
        try:
            npz_data = np.load(npz_f, allow_pickle=True)
            data.append(npz_data['data'])    
            labels.append(npz_data['labels']) 
            lengths.append(npz_data['lengths'])
        except Exception as e:
            print(f"Error reading {npz_f}: {e}")

    return data, labels, lengths, subjects

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess the DREAMER.mat file into subject-level NPZ files.")
    parser.add_argument("--mat_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--window_sec", type=float, default=2.0)
    parser.add_argument("--stride_overlap", type=float, default=0.0)
    args = parser.parse_args()
    preprocess_and_save(args.mat_path, args.output_dir, twin=args.window_sec, stride_overlap=args.stride_overlap)
