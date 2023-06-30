import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms, models
import pandas as pd
import torch
import torch.optim as optim
import os
from glob import glob
from skimage import io, transform
from matplotlib import pyplot as plt
import json
import optuna
from optuna.trial import TrialState
import joblib
from sklearn import metrics
from importlib import reload
from torchvision.utils import save_image

import modules.markush as md # custom module
import modules.patches as pt
reload(md) # Automatically reload changes made to custom module
reload(pt)
# ------------------------------------------ Data creation ----------------------------
DATA_DIR = "./data/training"

# create MD dataset 
MD = md.MarkushDataset.from_dir(DATA_DIR)

# Simply use all data as testing data
MD_split = {}
MD_split['test'] = md.MarkushDataset.from_parent(MD, range(len(MD))) 


# --------------------------

def investigate_batch(inputs, labels, preds, saliency_maps, parent_ids, image_count, output_dir):
    for i, input in enumerate(inputs):
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(input.cpu().detach().numpy().transpose(1, 2, 0))
        ax[0].axis('off')
        ax[1].imshow(saliency_maps[i].cpu(), cmap='hot')
        ax[1].axis('off')
        plt.tight_layout()

        
        if labels[i] == 0: # The image contains no Markush Structure
            if preds[i] == 0: # The patch is a True Negative
                plt.savefig(f"{output_dir}/TN/{parent_ids[i][:-4]}-{image_count}.png")
            elif preds[i] == 1: # The patch is a False Positive
                plt.savefig(f"{output_dir}/FP/{parent_ids[i][:-4]}-{image_count}.png")
        elif labels[i] == 1: # The image contains a markush structure
            if preds[i] == 0: # The patch is a False Negative
                plt.savefig(f"{output_dir}/FN/{parent_ids[i][:-4]}-{image_count}.png")
            elif preds[i] == 1: # The patch is a True Positive
                plt.savefig(f"{output_dir}/TP/{parent_ids[i][:-4]}-{image_count}.png")
        
        image_count += 1
        plt.close()
        

    return image_count
        

# Specify random transforms
def get_transforms():

    testing_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()])

    data_transforms = {
        'test': testing_transforms
    }

    return data_transforms

# Create train and test patches sets
def get_data(patchsize: tuple):
    patches_datasets = {x: pt.PatchesDataset(
        MD_split[x],
        crop_width = patchsize[0],
        crop_height = patchsize[1],
        ann_area_ratio = 0.5,
        transform=get_transforms()[x])

        for x in ['test']}

    patches_dataloaders = {x: torch.utils.data.DataLoader(
                            patches_datasets[x],
                            batch_size=BATCHSIZE, 
                            shuffle=True,
                            num_workers=NUM_WORKERS)
        
        for x in ['test']}

    return patches_dataloaders

# Define the CNN model
def define_model():

    return torch.load(f"./Results/{STUDY_NAME}.pt").to(DEVICE)


def evaluate_model():
    """
    This function evaluates performance on the test set, by training on both the training and validation sets.
    It returns the model, a classification report, the confusion matrix and the test set results.
    """

    model = define_model()

    dataloaders = get_data(INPUT_SIZE)


    # Evaluation on test set
    model.eval()
    test_running_preds = []
    test_running_parents = []
    test_running_parent_labels = []
    test_running_outputs0 = []
    test_running_outputs1 = []
    image_count = 0

    # We can only keep track of patch level results if we have labels for the patches
    if PATCH_LEVEL_RESULTS:
        test_running_labels = []

    # Iterate over data in batches
    for inputs, labels, parent_ids, parent_labels in dataloaders['test']:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        # If we are investigating errors, we want grad enabled to make saliency maps
        with torch.set_grad_enabled(False or ERROR_INVESTIGATION):
            inputs.requires_grad_()

            outputs = model(inputs)
            # Get predictions from whichever value is larger (output 1 or 2)
            max_values, preds = torch.max(outputs, 1)

            # Calculate Saliency maps and save them together with the images in their corresponding folders (FN, TN, FP, TP)
            if ERROR_INVESTIGATION:
                # Compute gradients
                scores = torch.gather(outputs, 1, preds.view(-1, 1))
                scores.backward(torch.ones_like(scores))

                # Compute saliency maps
                saliency_maps, _ = torch.max(inputs.grad.data.abs(), dim=1)

                # Reset gradients
                inputs.grad.zero_()

                # Reset the requires_grad flag
                inputs.requires_grad_(False)


                image_count = investigate_batch(inputs, labels,
                                                 preds, saliency_maps,
                                                 parent_ids, image_count,
                                                 output_dir='./Results/Error_Investigation_IV3_IN_FM/')


        # Save these values for exporting, analysis, reporting
        test_running_preds.extend(preds.cpu().numpy())
        test_running_parents.extend(parent_ids)
        test_running_parent_labels.extend(parent_labels.cpu().numpy())
        test_running_outputs0.extend(outputs.cpu().detach().numpy()[:,0])
        test_running_outputs1.extend(outputs.cpu().detach().numpy()[:,1])

        # Only if we can get patch level results, save the labels
        if PATCH_LEVEL_RESULTS:
            test_running_labels.extend(labels.cpu().numpy())

    # Create DF with results, including output logits
    results = pd.DataFrame({
        'output_0': test_running_outputs0,
        'output_1': test_running_outputs1,
        'prediction': test_running_preds,
        'parent_image': test_running_parents,
        'parent_label': test_running_parent_labels
    })

    if PATCH_LEVEL_RESULTS:
        return (model, metrics.classification_report(test_running_labels, test_running_preds, output_dict=True),
                metrics.confusion_matrix(test_running_labels, test_running_preds), results)

    else:
        return results


# -------------- Specify Constants

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCHSIZE = 8
NUM_WORKERS = 2
STUDY_NAME = "moretesting"
STUDY_COMMENT = ""
N_TRIALS = 25
ARCHITECTURE = "resnet18" # "inceptionv3_USPTO", ïnceptionv3_IN" or "resnet18"
INPUT_SIZE = (224, 224)
STOPPING_PATIENCE = 5
MAX_EPOCHS = 50
NR_REPETITIONS = 1
PATCH_LEVEL_RESULTS = True
ERROR_INVESTIGATION = True
        
if __name__ == '__main__':
    for i in range(NR_REPETITIONS):

        # Check if a study with this name already exists
        if os.path.isfile(f"Results/{STUDY_NAME}.pkl"):
            print("Study already exists, evaluating")

            # Run NR_REPETITIONS times, each with different name, manually combine results
            SAVE_NAME = STUDY_NAME + f"_Error_Investigation_{i}"

        else:
            raise Exception("Study does not exist")

        print("\n\nEvaluating on Testset...\n\n")

        # Run on test set using the best hyperparameters found
        final_model, classi_report, confusion_mat, results = evaluate_model()
        # Save results as csv
        results.to_csv(f"Results/{SAVE_NAME}.csv", index=False)


        with open(f"Results/{SAVE_NAME}.txt", 'x') as f:
            
            f.write(f"\n{STUDY_NAME}\n")
            f.write(f"{STUDY_COMMENT}\n\n")
            f.write(f"  STUDY_NAME: {STUDY_NAME}\n")
            f.write(f"  ARCHITECTURE: {ARCHITECTURE}\n")
            f.write(f"  INPUT_SIZE: {INPUT_SIZE}\n")
            f.write(f"  MAX_EPOCHS: {MAX_EPOCHS}\n")
            f.write(f"  DATA_DIR: {DATA_DIR}\n")
            f.write(f"  NR_REPETITIONS: {NR_REPETITIONS}\n")
            f.write(f"  PATCH_LEVEL_RESULTS: {PATCH_LEVEL_RESULTS}\n")
            

            if PATCH_LEVEL_RESULTS:
                f.write(f"\n\nTest Results - Patch Level\n")
                f.write(f"{json.dumps(classi_report, indent=2)}\n\n")

                tn, fp, fn, tp = confusion_mat.ravel()

                f.write(f"Label 0: [{tn} {fp}]\nLabel 1: [{fn} {tp}]\n")
                f.write(f"  Pred:   0   1\n")

                f.write(f"\ntn: {tn}\nfp: {fp}\nfn: {fn}\ntp: {tp}")



            f.write(f"\n\n-------------- Image Level Results, simple method -------------\n\n")
            grouped_by_image = results.groupby(['parent_image']).agg({'parent_label': max, 'prediction': max})
            f.write(f"{json.dumps(metrics.classification_report(grouped_by_image['parent_label'], grouped_by_image['prediction'], output_dict=True), indent=4)}\n")
            confusion_mat = metrics.confusion_matrix(grouped_by_image['parent_label'], grouped_by_image['prediction'])

            tn, fp, fn, tp = confusion_mat.ravel()
            f.write(f"Label 0: [{tn} {fp}]\nLabel 1: [{fn} {tp}]\n")
            f.write(f"  Pred:   0   1\n")

            f.write(f"\ntn: {tn}\nfp: {fp}\nfn: {fn}\ntp: {tp}")


        with open(f"Results/{SAVE_NAME}.txt", 'r') as f:
            print(f.read())

