import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import random_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
import xgboost as xgb

from importlib import reload
import modules.markush as md # custom module
reload(md)
plt.rcParams['figure.figsize'] = (10, 5)

def preprocess_image(img):
    # Blur images very slightly to reduce effect of noise
    #img = cv2.blur(img, (3,3))

    return img

# A function for calculating features from a query image and matching it to a many template images
def orb_and_match_multiple(query_image, orb, matcher, templates, threshold=0.80):
    # We will have a list of matches, equal to the amount of template we will match against
    match_features = {}

    # Calculate keypoints and descriptor for the image we are querying.
    query_keypoints, query_descriptor = orb.detectAndCompute(query_image, None)

    for i in range(templates.shape[2]):
        # Get patch i from the templates and compute
        template_keypoints, template_descriptor = orb.detectAndCompute(templates[:,:,i], None)

        # Prevent knnMatch throwing an error
        if not (query_descriptor is None or template_descriptor is None):
        # Calculate matches, k is set to 2 to use Lowe's ratio test
            matches = matcher.knnMatch(query_descriptor, template_descriptor, k=2)
        else: # If either image has no descriptor
            matches = []

        # Ratio test as per Lowe's paper, filter out good matches
        distances = []
        match_count = 0
        for index in range(len(matches)):
            if len(matches[index]) == 2: # If there are two nearest neighbours
                m, n = matches[index]
                if m.distance < threshold * n.distance:  # 0.8 is threshold of ratio testing
                    distances.append(m.distance)        # We sum the distance of the best matches (m), if they are less than threshold * n.distance
                    match_count += 1       # Keep track of the amount of matches for a given template

        sorted_distances = sorted(distances)
        sorted_distances.extend([np.nan, np.nan, np.nan, np.nan, np.nan]) # Append 5 NaNs in case there are less than 5 matches
        match_features[f'temp{i}_match0'] = sorted_distances[0] # Distance of best match
        match_features[f'temp{i}_match1'] = sorted_distances[1] # Distance of second best match...
        match_features[f'temp{i}_match2'] = sorted_distances[2]
        match_features[f'temp{i}_match3'] = sorted_distances[3]
        match_features[f'temp{i}_match4'] = sorted_distances[4]
        match_features[f'temp{i}_nr_matches'] = match_count

        

    return match_features

# Function that converts a dataset with Markush images into a dataframe with features extracted from ORB
def create_feature_set(dataset, orb, matcher, templates):
    feature_set = []

    # For each image calculate the amount of matches and create a dataframe
    for i in range(len(dataset)):
        matches = orb_and_match_multiple(dataset[i]['image'], orb, matcher, templates)

        # We add the label to the matches dictionary, together they form the columns of our dataframe
        matches.update({'label': dataset[i]['label']})
        feature_set.append(matches) 
    return pd.DataFrame(feature_set)
    
def grid_search(MD, in_nr_orb_features, in_nr_patches, in_xgb_nr_estimators, in_xgb_max_depth, train_set, validation_set):

    # Create ORB object
    orb = cv2.ORB_create(nfeatures=in_nr_orb_features)

    # Create a Brute Force Matcher object.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = False)

    # Get random set of patches to use as templates for ORB
    templates = MD.random_markush_patches(in_nr_patches, give_context=False)

    # Create a train set dataframe with the features that we will train our classifier on
    train_set_features = create_feature_set(train_set, orb, matcher, templates)

    # Create a test set for testing on
    validation_set_features = create_feature_set(validation_set, orb, matcher, templates)

    # Define the XGBoost model
    xgb_model = xgb.XGBClassifier(n_estimators=in_xgb_nr_estimators, max_depth=in_xgb_max_depth, learning_rate=0.01, eval_metric='logloss')


    # Train the model on the training data
    xgb_model.fit(train_set_features.drop(columns='label'), train_set_features['label'])

    # Make predictions on the test data
    y_pred = xgb_model.predict(validation_set_features.drop(columns='label'))

    # Evaluate the model
    out_accuracy = accuracy_score(validation_set_features['label'], y_pred)

    scores = precision_recall_fscore_support(validation_set_features['label'], y_pred, labels=xgb_model.classes_)

    cm = confusion_matrix(validation_set_features['label'], y_pred)

    results = {
        'in_nr_orb_features': in_nr_orb_features,
        'in_nr_patches': in_nr_patches,
        'in_xgb_nr_estimators': in_xgb_nr_estimators,
        'in_xgb_max_depth': in_xgb_max_depth,

        'accuracy': out_accuracy, 

        'label_0_precision': scores[0][0],
        'label_0_recall': scores[1][0],
        'label_0_fscore': scores[2][0],
        'label_0_support': scores[3][0],

        'label_1_precision': scores[0][1],
        'label_1_recall': scores[1][1],
        'label_1_fscore': scores[2][1],
        'label_1_support': scores[3][1],
        'confusion': cm
    }

    return results