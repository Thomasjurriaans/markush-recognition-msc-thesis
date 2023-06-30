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

import modules.markush as md # custom module
import modules.patches as pt
reload(md) # Automatically reload changes made to custom module
reload(pt)
# ------------------------------------------ Data creation ----------------------------

# create MD dataset 
MD = md.MarkushDataset.from_dir('./data/training')

# split it into train and val/test, we stratify to make sure the classes are balanced
train_indices, valtest_indices = train_test_split(
    range(len(MD)),
    stratify=MD.data.label,
    test_size=0.3
)

# split into val and test 50/50
val_indices, test_indices = train_test_split(
    valtest_indices,
    stratify=MD.data.label[valtest_indices],
    test_size=0.50
)

MD_split = {}
MD_split['train'] = md.MarkushDataset.from_parent(MD, train_indices)
MD_split['val'] = md.MarkushDataset.from_parent(MD, val_indices)
MD_split['train_val'] = md.MarkushDataset.from_parent(MD, train_indices+val_indices)
MD_split['test'] = md.MarkushDataset.from_parent(MD, test_indices)


# --------------------------

# Specify random transforms
def get_transforms(trial):

    trans_prob = trial.suggest_float("transform_probability", 0, 1)

    training_transforms = transforms.Compose([
        # Duplicate gray channel 3 times to work with RGB based models
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomPerspective(distortion_scale=0.25, fill=255, p=trans_prob),
        transforms.RandomPosterize(bits=4, p=trans_prob),
        transforms.RandomChoice([transforms.RandomAdjustSharpness(sharpness_factor=2, p=trans_prob),  # Either sharpen or blur
                                transforms.RandomAdjustSharpness(sharpness_factor=0, p=trans_prob)]),
        transforms.ToTensor()
    ])

    testing_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()])

    data_transforms = {
        'train': training_transforms,
        'val': testing_transforms,
        'train_val': training_transforms,
        'test': testing_transforms
    }

    return data_transforms

# Create train and test patches sets
def get_data(trial, patchsize: tuple):
    patches_datasets = {x: pt.PatchesDataset(
        MD_split[x],
        crop_width = patchsize[0],
        crop_height = patchsize[1],
        ann_area_ratio = 0.5,
        transform=get_transforms(trial)[x])

        for x in ['train', 'val', 'train_val', 'test']}

    patches_dataloaders = {x: torch.utils.data.DataLoader(
                            patches_datasets[x],
                            batch_size=BATCHSIZE, 
                            shuffle=True,
                            num_workers=NUM_WORKERS)
        
        for x in ['train', 'val', 'train_val', 'test']}

    return patches_dataloaders

# Define the CNN model
def define_model(trial):

    if ARCHITECTURE == "resnet18":
        model = models.resnet18(weights='IMAGENET1K_V1')

        # Freeze the layers to prevent training them
        if TRAIN_FC_ONLY:
            for param in model.parameters():
                param.requires_grad = False


        # Set the last layer to predict 2 classes
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, 2)

    if ARCHITECTURE == "inceptionv3_USPTO":
        model = models.inception_v3(aux_logits=False)

        # Freeze the layers to prevent training them
        if TRAIN_FC_ONLY:
            for param in model.parameters():
                param.requires_grad = False

        # Set the last layer to predict 2 classes
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, 2)
        
        # Define the dictionary and list of keys to remove
        state_dict = torch.load("../cnc_small.pth") # Load Elsevier pre-trained model
        keys_to_remove = [key for key in state_dict.keys() if key.startswith("AuxLogits")]

        # Remove the keys
        state_dict = {key: value for key, value in state_dict.items() if key not in keys_to_remove}

        model.load_state_dict(state_dict)
        
    if ARCHITECTURE == "inceptionv3_IN":        
        model = models.inception_v3(weights='IMAGENET1K_V1', aux_logits=True)
        
        # We want to remove the auxiliary logits. We can't do this in model definition because of an Error, so we do it manually:
        state_dict = model.state_dict()
        keys_to_remove = [key for key in state_dict.keys() if key.startswith("AuxLogits")]

        # Remove the keys
        state_dict = {key: value for key, value in state_dict.items() if key not in keys_to_remove}
        
        #Re-insert the state_dict into a new model
        model = models.inception_v3(aux_logits=False)
        model.load_state_dict(state_dict)
        
        # Freeze the layers to prevent training them
        if TRAIN_FC_ONLY:
            for param in model.parameters():
                param.requires_grad = False
                
        # Set the last layer to predict 2 classes
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, 2)
        

    return model.to(DEVICE)


def objective(trial):
    # Early stopping
    triggertimes = 0
    best_epoch = 0
    best_val_loss = 9999

    model = define_model(trial)

    dataloaders = get_data(trial, INPUT_SIZE)

    # Get hyperparameter suggestions from Optuna
    optimizer_name = trial.suggest_categorical("optimizer", OPTIMIZER_OPTIONS)
    lr = trial.suggest_float("lr", LR_LOWER, LR_UPPER, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    criterion = torch.nn.CrossEntropyLoss()

    # Keep track of the latest f1 score, this will be returned
    last_f1_score = 0

    for epoch in range(MAX_EPOCHS):
        printtext = ""

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            # During this phase, keep track of the labels and preds of all batches for evaluation
            phase_running_labels = []
            phase_running_preds = []
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data in batches
            for inputs, labels, _ in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history only when in training phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1) # Get predictions from whichever value is larger (output 1 or 2)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Keep track of statistics for this batch
                running_loss += loss.item() * inputs.size(0) # Loss times batch size
                phase_running_labels.extend(labels.cpu().data)
                phase_running_preds.extend(preds.cpu())

            # Compute the val statistics for this epoch
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_f1 = metrics.f1_score(phase_running_labels, phase_running_preds, average="macro")
            printtext += (f"Epoch {epoch} - {phase} loss: {epoch_loss} F1: {epoch_f1}   ")

            
            if phase == 'val':
                print(printtext)
                
                # Early Stopping
                if epoch_loss > best_val_loss:
                    triggertimes += 1
                    print(f'Early stopping triggers: {triggertimes}')
                
                # Compare to best val values:
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_epoch = epoch
                    best_conf_matrix = metrics.confusion_matrix(phase_running_labels, phase_running_preds)
                    triggertimes = 0
                
                # We stop early if the epoch loss has not improved in the last X epochs
                if triggertimes >= STOPPING_PATIENCE:
                    print(f'Stopping early, best epoch: {best_epoch}\n')

                    # If we stop, note the optimal stopping epoch and best confusion matrix
                    trial.set_user_attr('stop_epoch', best_epoch)
                    trial.set_user_attr('confusion_matrix', best_conf_matrix)
                    return last_f1_score

                
                # Keep track of epoch's metrics for next round
                last_f1_score = epoch_f1

                # Report to trial for pruning
                trial.report(epoch_f1, epoch)
                # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    trial.set_user_attr('stop_epoch', epoch)
                    raise optuna.exceptions.TrialPruned(f"Trial was pruned at epoch {epoch}.")

    # If we did not stop early, set the user attributes in the last epoch, before returning
    trial.set_user_attr('stop_epoch', epoch)
    trial.set_user_attr('confusion_matrix', best_conf_matrix)            
    return last_f1_score

def objective_final(trial: optuna.trial.FrozenTrial):
    """
    This function evaluates performance on the test set, by training on both the training and validation sets.
    It returns the model, a classification report, the confusion matrix and the test set results.
    """

    model = define_model(trial)

    dataloaders = get_data(trial, INPUT_SIZE)

    # Get hyperparameter suggestions from Optuna
    optimizer_name = trial.suggest_categorical("optimizer", OPTIMIZER_OPTIONS)
    lr = trial.suggest_float("lr", LR_LOWER, LR_UPPER, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    criterion = torch.nn.CrossEntropyLoss()
    
    optimal_epochs = trial.user_attrs['stop_epoch']

    # Keep track of the latest f1 score, this will be returned
    last_f1_score = 0

    # Training loop
    for epoch in range(optimal_epochs):
        model.train()  # Set model to training mode

        # Iterate over data in batches, both training and validation data in this case
        for inputs, labels, _ in dataloaders['train_val']:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history only when in train or trainval
            with torch.set_grad_enabled(True):
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                # Get predictions from whichever value is larger (output 1 or 2)
                _, preds = torch.max(outputs, 1)

                loss.backward()
                optimizer.step()


    # Evaluation on test set
    model.eval()
    test_running_preds = []
    test_running_labels = []
    test_running_parents = []
    test_running_outputs0 = []
    test_running_outputs1 = []

    # Iterate over data in batches
    for inputs, labels, parent_ids in dataloaders['test']:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            # Get predictions from whichever value is larger (output 1 or 2)
            _, preds = torch.max(outputs, 1)
        
        # Save these values for exporting, analysis, reporting
        test_running_labels.extend(labels.cpu().numpy())
        test_running_preds.extend(preds.cpu().numpy())
        test_running_parents.extend(parent_ids)
        test_running_outputs0.extend(outputs.cpu().numpy()[:,0])
        test_running_outputs1.extend(outputs.cpu().numpy()[:,1])

    # Create DF with results, including output logits
    results = pd.DataFrame({
        'output_0': test_running_outputs0,
        'output_1': test_running_outputs1,
        'prediction': test_running_preds,
        'ground_truth': test_running_labels,
        'parent_image': test_running_parents
    })

    return (model, metrics.classification_report(test_running_labels, test_running_preds, output_dict=True),
            metrics.confusion_matrix(test_running_labels, test_running_preds), results)


# -------------- Specify Constants

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCHSIZE = 16
NUM_WORKERS = 2
STUDY_NAME = "R18_IN_FC"
STUDY_COMMENT = ""
N_TRIALS = 25
OPTIMIZER_OPTIONS = ["Adam", "SGD"]
TRAIN_FC_ONLY = True
LR_LOWER = 1e-5
LR_UPPER = 1e-3
ARCHITECTURE = "resnet18" # "inceptionv3_USPTO", "inceptionv3_IN" or "resnet18"
INPUT_SIZE = (224, 224)
STOPPING_PATIENCE = 5
MAX_EPOCHS = 50
MODEL = ""
        
if __name__ == '__main__':

    # Check if a study with this name already exists
    if os.path.isfile(f"Results/{STUDY_NAME}.pkl"):
        print("Study already exists, continuing...")
        study = joblib.load(f"Results/{STUDY_NAME}.pkl")
        
        if os.path.isfile(f"Results/{STUDY_NAME}_continued.pkl"):
            raise Exception("Choose a different name")
            
        # To prevent file saving issues later
        STUDY_NAME = STUDY_NAME + "_continued"

    else:
        print("Study does not exist yet, creating new study...")
        # Create Optuna study
        study = optuna.create_study(study_name=STUDY_NAME, direction="maximize")

    print("\n\nFinding Best Hyperparameters...\n\n")

    # Perform the Optuna hyperparameter search
    try:
        study.optimize(objective, n_trials=N_TRIALS)
    except:
        print("Hyperparameter search ended early due to Exception")

    # Save study to pkl file
    joblib.dump(study, f"Results/{STUDY_NAME}.pkl")

    print("\n\nEvaluating on Testset...\n\n")

    # Run on test set using the best hyperparameters found
    final_model, classi_report, confusion_mat, results = objective_final(study.best_trial)

    # Save results as csv
    results.to_csv(f"Results/{STUDY_NAME}.csv", index=False)

    # Save model
    torch.save(final_model, f"Results/{STUDY_NAME}.pt")

    with open(f"Results/{STUDY_NAME}.txt", 'x') as f:
        
        # Write statistics about the study
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        f.write(f"{STUDY_NAME}\n")
        f.write(f"{STUDY_COMMENT}\n\n")
        f.write("Study statistics: ")
        f.write(f"\n  Number of finished trials: {len(study.trials)}")
        f.write(f"\n  Number of pruned trials: {len(pruned_trials)}")
        f.write(f"\n  Number of complete trials: {len(complete_trials)}\n")

        f.write("\nBest validation trial:")
        trial = study.best_trial

        f.write(f"\n  Value: {trial.value}")

        f.write("\n  Params: ")
        for key, value in trial.params.items():
            f.write("    {}: {}\n".format(key, value))
        f.write(f"    optimal_epochs: {trial.user_attrs['stop_epoch']}\n")
        f.write(f"    confusion_matrix: {trial.user_attrs['confusion_matrix']}\n")

        f.write(f"\n\nTest Results\n")
        f.write(f"{json.dumps(classi_report, indent=2)}\n\n")

        tn, fp, fn, tp = confusion_mat.ravel()

        f.write(f"Label 0: [{tn} {fp}]\nLabel 1: [{fn} {tp}]\n")
        f.write(f"  Pred:   0   1\n")

        f.write(f"\ntn: {tn}\nfp: {fp}\nfn: {fn}\ntp: {tp}")

        f.write("\n\nOptuna Parameters:\n")
        f.write(f"  BATCHSIZE: {BATCHSIZE}\n")
        f.write(f"  NUM_WORKERS: {NUM_WORKERS}\n")
        f.write(f"  STUDY_NAME: {STUDY_NAME}\n")
        f.write(f"  N_TRIALS: {N_TRIALS}\n")
        f.write(f"  OPTIMIZER_OPTIONS: {OPTIMIZER_OPTIONS}\n")
        f.write(f"  LR range: {LR_LOWER} - {LR_UPPER}\n")
        f.write(f"  TRAIN_FC_ONLY: {TRAIN_FC_ONLY}\n")
        f.write(f"  ARCHITECTURE: {ARCHITECTURE}\n")
        f.write(f"  INPUT_SIZE: {INPUT_SIZE}\n")
        f.write(f"  STOPPING_PATIENCE: {STOPPING_PATIENCE}\n")
        f.write(f"  MAX_EPOCHS: {MAX_EPOCHS}\n")
        
        f.write(f"\n\n-------------- Image Level Results, simple method -------------\n\n")
        grouped_by_image = results.groupby(['parent_image']).agg({'ground_truth': max, 'prediction': max})
        f.write(f"{json.dumps(metrics.classification_report(grouped_by_image['ground_truth'], grouped_by_image['prediction'], output_dict=True), indent=4)}\n")

    with open(f"Results/{STUDY_NAME}.txt", 'r') as f:
        print(f.read())

