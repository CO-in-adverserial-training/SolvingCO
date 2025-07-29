import numpy as np

# A Function For Automatically Detecting The Time Range Of CO Occurance
def detect_window_range(window_size, alignments, tail_portion=0.25):
    window_list = [(0 , -1)]
    tail_size = int(window_size * tail_portion)
    capture_next = True
    for start_index in range(len(alignments) - window_size):
        avg_alignment_start = np.mean(alignments[start_index: start_index + tail_size])
        avg_alignment_finish = np.mean(alignments[start_index + window_size - tail_size: start_index + window_size])
        if abs(avg_alignment_start - avg_alignment_finish) >= 0.5:
            if capture_next:
                window_list.append((start_index, start_index + window_size))
                capture_next = False
        else:
            capture_next = True
    return window_list

def plot_loss_and_accuracy(metrics_path: str):
    pass