
import torch
import os
import numpy as np

import cv2 as cv
from PIL import Image

def crop_object_with_mask(image):
   """Crops an image to the minimal bounding box containing the mask.
   """
   img = Image.fromarray(image)

   # Load the image
   img_array = np.array(image)
   # print(img_array[2])

   # Find non-white pixels
   mask_loc = np.any(img_array < 190, axis=2)  # Check all color channels

   # Find bounding box coordinates
   ymin, xmin = np.where(mask_loc)[0].min(), np.where(mask_loc)[1].min()
   ymax, xmax = np.where(mask_loc)[0].max() + 1, np.where(mask_loc)[1].max() + 1

   # Crop the image
   cropped_img = img.crop((xmin, ymin, xmax, ymax))

   return cropped_img


file_path = "F:\doctor\AlphaCLIP\image_wyn\straw_crop/3\crop_ini"
#file_path = "F:\doctor\segment-anything/brl_notebooks\isolating_masks/try3/1"
files = os.listdir(file_path)
output = "F:\doctor\AlphaCLIP\image_wyn\straw_crop/3\crop"


for file in files:
   image_path = os.path.join(file_path, file)
   image = cv.imread(image_path)
   image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
   croped_image = crop_object_with_mask(image)
   im = croped_image.convert("RGB")
   output_dir = os.path.join(output, file)
   im.save(output_dir)

print("done")

