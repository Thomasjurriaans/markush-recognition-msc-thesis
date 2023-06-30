from torch.utils.data import Dataset
import pandas as pd
import torch
import os
from glob import glob
from skimage import io
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import cv2
import torchvision as tv


# Create a Custom Dataset for loading images from the Markush Dataset
class MarkushDataset(Dataset):
    """Dataset for classifying Markush structures"""

    def __init__(self, data):

        self.data = data

        # Extract patches from images
        self.markush_patches = np.array(self.extract_markush_patches(200, 200))
        self.markush_patches_no_context = np.array(self.extract_markush_patches_nocontext(200, 200))

    
    # Create MarkushDataset subset from another MarkushDataset
    @classmethod
    def from_parent(cls, parent: 'MarkushDataset', indices):
        data = parent.data.iloc[indices]

        return cls(data)


    @classmethod
    def from_dir(cls, image_dir, transform=None):
        """
            Args:
                image_dir (string): directory with Markush Images, Labels and NoMarkush images.
        """

        # Get the paths of all Markush images and add a label
        markush_path = os.path.join(image_dir, "Markush/*")
        Markush = pd.DataFrame(glob(markush_path), columns=['path'])
        Markush['label'] = 1

        # Get the paths of all Non-Markush images and add a label
        NoMarkush_path = os.path.join(image_dir, "NoMarkush/*")
        NoMarkush = pd.DataFrame(glob(NoMarkush_path), columns=['path'])
        NoMarkush['label'] = 0

        # Concatenate dataframes to get full path list
        paths = pd.concat([Markush, NoMarkush])

        # Extract image name from path explicitly
        paths['img_name'] = paths['path'].str.extract(r"([A-Za-z0-9-_]*\.[a-zA-Z]{3})")

        # Since the dataset is small, it will be more efficient to load them into memory already
        data = paths.apply(cls._load_image_into_memory, axis=1)

        # Load annotations
        data = cls._load_annotations(data)

        # Check whether all images labelled as Markush have annotations, no value should be NaN
        #assert(data[data['label'] == 1]['annotations'].isna().values.any() == False)

        return cls(data)
    
        
    # Helper function to load images into self.data during initialization
    @classmethod
    def _load_image_into_memory(cls, row):
        # Read image, convert to black/white
        image = cv2.imread(row['path'], cv2.IMREAD_GRAYSCALE)
        row['image'] = image
        return row
    
        
    # Load image annotations for Markush images
    @classmethod
    def _load_annotations(cls, data):
        annotations = pd.read_json('./data/training/Export2.json')                                      # Read json file with annotations
        annotations['img_name'] = annotations['image'].str.extract(r"([A-Za-z0-9-]*\.[a-zA-Z]{3})")     # Extract image names for matching

        annotations = annotations[['img_name', 'label']]                                                # Keep only relevant columns
        annotations = annotations.rename({'label': 'annotations'}, axis=1)                              # Rename label to annotations

        merged = pd.merge(left=data, right=annotations, how='left', on='img_name')                 # Merge annotations into data
        return merged
    

    # Return length of dataset
    def __len__(self):
        return len(self.data)
    
    
    # Get item from the dataset
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        return self.data.iloc[idx]

    # Load image directly from disk
    def load_image_from_disk(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return cv2.imread(self.paths.iloc[idx]['path'], cv2.IMREAD_GRAYSCALE)


    # Show an image with annotations
    def showimage(self, id):
        image = self.data.iloc[id]['image']                     # Get image data

        plt.imshow(image, cmap='gray')                          # Show image

        ax = plt.gca()                                          # Get current ax

        # If image contains annotations, draw them
        if isinstance(self.data.iloc[id]['annotations'], list):
            for ann in self.data.iloc[id]['annotations']:           # Iterate over dictionary with annotation features
                # Convert annotations to pixels instead of proportions
                x, y, width, height = self.annotation_px_conversion(ann)
                # Draw rectangle
                rect = Rectangle((x,y), width, height, linewidth=1, edgecolor='r', facecolor='none') # Create PLT rectangle patch
                ax.add_patch(rect)

        plt.show()

    # Convert annotations from proportions to pixels
    def annotation_px_conversion(self, ann):
        im_width = ann['original_width']
        im_height = ann['original_height']

        x = int(ann['x']/100 * im_width)     # Convert coordinates system for displaying, proportions to pixels
        y = int(ann['y']/100 * im_height)    # https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks
        width = int(ann['width']/100 * im_width)
        height = int(ann['height']/100 * im_height)
        
        # x and y are bottom left of annotation box
        return x, y, width, height


    # Extract all markush patches from the dataset
    def extract_markush_patches(self, patchwidth, patchheight):
        # We will store them in memory directly, since the dataset is very small
        markush_patches = []

        for i, row in self.data.iterrows():
            if isinstance(row['annotations'], list): # Only do this for image with annotations
                for ann in row['annotations']:      # For each annotation of each image...
                    # Get coordinates of annotation
                    x, y, width, height = self.annotation_px_conversion(ann)

                    # Add a white border equal to patch width and height to act as padding
                    image_with_border = cv2.copyMakeBorder(row['image'], patchheight, patchheight, patchwidth, patchwidth, cv2.BORDER_CONSTANT, value=255)


                    # Add patchheight and patchwidth because of the padding
                    ycenter = y+(height//2)+patchheight
                    xcenter = x+(width//2)+patchwidth

                    xoffset = patchwidth//2
                    yoffset = patchheight//2

                    
                    patch = image_with_border[ycenter-yoffset:ycenter+yoffset, xcenter-xoffset:xcenter+xoffset]

                    # Append to list
                    markush_patches.append(patch)

        return markush_patches
    
    # Extract all markush patches from the dataset, making everything outside the annotated box white
    def extract_markush_patches_nocontext(self, patchwidth, patchheight):
        # We will store them in memory directly, since the dataset is very small
        markush_patches_no_context = []

        for i, row in self.data.iterrows():
            if isinstance(row['annotations'], list):
                for ann in row['annotations']:      # For each annotation of each image...
                    # Get coordinates of annotation, x and y are bottom left corner of the annotation box
                    x, y, width, height = self.annotation_px_conversion(ann)

                    annotation_box_only = row['image'][y:y+height, x:x+width]

                    # Add a white border equal to patch width and height to act as padding
                    image_with_border = cv2.copyMakeBorder(annotation_box_only, patchheight, patchheight, patchwidth, patchwidth, cv2.BORDER_CONSTANT, value=255)

                    # The annotation is in the center, simply get the center from shape
                    ycenter = image_with_border.shape[0]//2
                    xcenter = image_with_border.shape[1]//2

                    xoffset = patchwidth//2
                    yoffset = patchheight//2

                    patch = image_with_border[ycenter-yoffset:ycenter+yoffset, xcenter-xoffset:xcenter+xoffset]

                    # Append to list
                    markush_patches_no_context.append(patch)

        return markush_patches_no_context

    # Get a single rando markush patch
    def random_markush_patch(self, give_context):
        if give_context:
            index = np.random.randint(0, len(self.markush_patches))
            return self.markush_patches[index,:,:]
        else:
            index = np.random.randint(0, len(self.markush_patches_no_context))
            return self.markush_patches_no_context[index,:,:]
        
    # Get a custom amount of markush patches
    def random_markush_patches(self, amount, give_context):
        patches = []
        for i in range(amount):  
            if give_context:
                index = np.random.randint(0, len(self.markush_patches))
                patches.append(self.markush_patches[index,:,:])
            else:
                index = np.random.randint(0, len(self.markush_patches_no_context))
                patches.append(self.markush_patches_no_context[index,:,:])
        return np.dstack(patches)



