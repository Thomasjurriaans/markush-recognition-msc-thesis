from modules.markush import MarkushDataset
from torch.utils.data import Dataset
import pandas as pd
import torch
from glob import glob
from skimage import io
import numpy as np
from scipy import ndimage
from collections import namedtuple
from PIL import Image
from torchvision import transforms

# Define a named tuple for use later, must be defined outside of class.
# params  crop_width, crop_height, ann_area_ratio, angle_variance
Box = namedtuple('Box', 'xmin ymin xmax ymax width height')

# Create a Custom Dataset for Patches from a MarkushDataset
class PatchesDataset(Dataset):
    """Dataset for Markush image Patches"""

    def __init__(self, MD: MarkushDataset, transform,  **params):

        # Convert parent dataset (full images) to patches that constitute those images
        self.data = self.MD_to_patches(MD, params)

        # Save transforms that are passed during initialization
        self.transform = transform

    # Return length of dataset
    def __len__(self):
        return len(self.data)
    
    # Get item from the dataset
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data.iloc[idx].to_dict()

        # If we have transforms, perform them each time an item is sampled
        if self.transform:
            sample['patch'] = self.transform(sample['patch'])

        return sample['patch'], sample['label'], sample['parent_image'], sample['parent_label']
    

    """
    Convert an entire MarkushDataset into a PatchesDataset
    """
    def MD_to_patches(self, MD, params):
        # Extract the patches from each image, save them in a PD dataframe
        patches = []
        for row in MD:
            image = Image.fromarray(row['image'])                           # Convert to PIL
            # Pad equal to the largest of the crop sizes, this ensures the crop never falls outside the image
            padding = max(params['crop_width'], params['crop_height'])

            # Pad the image with white
            image = transforms.functional.pad(image, 
                                                padding = padding, # Fill equal to whichever is larger
                                                fill = 255, # White
                                                padding_mode='constant'
                                            )
            
            # How many crops do we need to make to cover the entire image, excluding padding?
            width_nr_crops = image.size[0]//params['crop_width']
            height_nr_crops = image.size[1]//params['crop_height']

            # Iterate over the image in a grid pattern, taking crops
            for i in range(width_nr_crops):
                for j in range(height_nr_crops):
                    # Calculate crop location
                    x = i*params['crop_width']
                    y = j*params['crop_height']

                    # Calculate x and y for a second grid that is offset by half from original grid
                    x_off = x+(params['crop_width']//2)
                    y_off = y+(params['crop_height']//2)

                    # Create current crop for First grid, skipping the first row and column
                    if i > 0 and j > 0:
                        current = self.get_single_crop(MD, x=x, y=y, image=image, row=row, params=params)

                        # Don't save patches that are completely white
                        if np.average(current['patch']) < 255.0:
                            patches.append(current)

                    # Create current crop for second grid, skipping the last row and column
                    if i < width_nr_crops-1 and j < height_nr_crops-1:
                        current = self.get_single_crop(MD, x=x_off, y=y_off, image=image, row=row, params=params)

                        # Don't save patches that are completely white
                        if np.average(current['patch']) < 255.0:
                            patches.append(current)

        return pd.DataFrame(patches)



    """
    Get a single crop from an image with annotations, this is a helper function
    """
    def get_single_crop(self, MD, x, y, image, row, params):
        crop = Box(x, y, x+params['crop_width'], y+params['crop_height'], params['crop_width'], params['crop_height'])

        # Keep track of the label of this patch
        crop_contains_markush = 0
        
        # Pad equal to the largest of the crop sizes, this ensures the crop never falls outside the image
        padding = max(params['crop_width'], params['crop_height']) 

        # Calculate overlap with annotations, brute force:
        if isinstance(row['annotations'], list):
            for ann in row['annotations']:
                # Convert annotations to pixels instead of proportions
                ann_x, ann_y, ann_width, ann_height = MD.annotation_px_conversion(ann)

                annotation = Box(ann_x+padding, ann_y+padding, ann_x+ann_width+padding, ann_y+ann_height+padding, ann_width, ann_height)
                if self.calc_overlap(annotation, crop) > params['ann_area_ratio']:
                    crop_contains_markush = 1
                    break

        # Crop image
        image_crop = image.crop((crop.xmin, crop.ymin, crop.xmax, crop.ymax))
        
        # Return as dict with PIL image
        return {'patch': image_crop, 
                'label': crop_contains_markush,
                'parent_image': row['img_name'],
                'parent_label': row['label']}
    
    # function to calculate how much of the annotation's area is inside the crop
    def calc_overlap(self, ann, crop):
        ann_area = ann.width*ann.height


        dx = min(ann.xmax, crop.xmax) - max(ann.xmin, crop.xmin)
        dy = min(ann.ymax, crop.ymax) - max(ann.ymin, crop.ymin)

        if ann_area <= 0.0 and (dx >= 0) and (dy >= 0):
            print("??")
        if (dx >= 0) and (dy >= 0):
            return (dx*dy)/ann_area
        else:
            return 0.0
