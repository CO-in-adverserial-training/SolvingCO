import torch
import numpy as np
from sklearn.metrics import classification_report

def accuracy_evaluation(model, dataloader, classes, is_attacked=False, method=None, epsilon=8 / 255, alpha=16 / 255, detailed=True, device:str="cuda"):
    model_training = model.training
    model.eval()
    pred_list = []
    true_labels = []
    for i, data in enumerate(dataloader, 0):
        try:
            images, labels = data
        except:
            images, labels, index = data
        images, labels = images.to(device), labels.to(device)
        if is_attacked:
            adv_images = method(images, labels, model, epsilon, alpha)
            preds = model(adv_images)
        else:
            preds = model(images)
        indices = torch.argmax(preds, 1)
        indices = indices.cpu()
        labels = labels.cpu()
        for j in range(len(indices)):
            pred_list.append(classes[indices[j]])
            true_labels.append(classes[labels[j]])
    if model_training:
        model.train()
    if detailed:
        print(classification_report(true_labels, pred_list, digits=4))
    else:
        return (np.array(true_labels) == np.array(pred_list)).sum() / len(true_labels)