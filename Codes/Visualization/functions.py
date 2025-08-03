import numpy as np
import json
import matplotlib.pyplot as plt

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

def plot_loss_and_accuracy(args, training_loss, training_accuracy, attack_evaluation_loss, attack_evaluation_accuracy,
                            fgsm_evaluation_loss, fgsm_evaluation_accuracy, pgd_evaluation_loss, pgd_evaluation_accuracy):
    figure, axis = plt.subplots(1,2, figsize=(15,5))
    axis[0].plot(np.arange(len(training_loss)) ,training_loss, '-o', label=f'Train Loss {args.attack}')
    axis[0].plot(np.arange(len(attack_evaluation_loss)) ,attack_evaluation_loss, '-o', label=f'Test Loss {args.attack}')
    axis[0].plot(np.arange(len(fgsm_evaluation_loss)) ,fgsm_evaluation_loss, '-o', label='Test Loss FGSM')
    axis[0].plot(np.arange(len(pgd_evaluation_loss)) ,pgd_evaluation_loss, '-o', label='Test Loss PGD')
    # axis[0].plot(np.arange(len(test_losses_benign)) ,test_losses_benign, '-o', label='Test Loss Benign')
    axis[0].set_title("Loss vs. Epochs")
    axis[0].set_xlabel("Epochs")
    axis[0].set_ylabel("Loss")
    axis[0].legend()
    axis[0].grid()
    axis[1].set_title("Accuracy vs. Epochs")
    axis[1].set_xlabel("Epochs")
    axis[1].set_ylabel("Accuracy")
    axis[1].plot(np.arange(len(training_accuracy)),training_accuracy, '-o', label=f'Train Accuracy {args.attack}')
    axis[1].plot(np.arange(len(attack_evaluation_accuracy)),attack_evaluation_accuracy, '-o', label=f'Test Accuracy {args.attack}')
    axis[1].plot(np.arange(len(fgsm_evaluation_accuracy)),fgsm_evaluation_accuracy, '-o', label='Test Accuracy FGSM')
    axis[1].plot(np.arange(len(pgd_evaluation_accuracy)),pgd_evaluation_accuracy, '-o', label='Test Accuracy PGD')
    # axis[1].plot(np.arange(len(accs_benign)),accs_benign, '-o', label='Test Accuracy Benign')
    axis[1].legend()
    axis[1].grid(visible=True, which= 'minor', color='k', linestyle='-', alpha=0.4)
    axis[1].grid(visible=True, which= 'major', color='b', linestyle='-', alpha=0.8)
    plt.minorticks_on()
    plt.savefig(f"{args.root_path}/plots/loss_accuracy_plot.pdf")
    plt.show()


def plot_alignment(args, alignments: list):
    plt.figure(figsize=(15,6))
    plt.plot(alignments)
    plt.grid()
    plt.savefig(f"{args.root_path}/plots/alignment_plot.pdf")
    plt.show()