import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


def getDisparity(left_img, right_img, patch_radius, min_disp, max_disp):
    """
    left_img and right_img are both H x W and you should return a H x W matrix containing the disparity d for
    each pixel of left_img. Set disp_img to 0 for pixels where the SSD and/or d is not defined, and for
    d estimates rejected in Part 2. patch_radius specifies the SSD patch and each valid d should satisfy
    min_disp <= d <= max_disp.
    """
    disparity_map = np.zeros(left_img.shape, dtype=np.float32)
    row_num, col_num = left_img.shape
    
    left_img_norm = left_img/255.0
    right_img_norm = right_img/255.0
    
    patch_size = 2 * patch_radius + 1       #patch size
    
    for i in range(patch_radius, row_num - patch_radius):
        for j in range(patch_radius + max_disp, col_num - patch_radius):
           left_patch = left_img_norm[i - patch_radius:i + patch_radius + 1, 
                                      j - patch_radius:j + patch_radius + 1]
           
           left_patch_flat = left_patch.ravel()[np.newaxis, :]
           
           right_patches = []
           valid_disps = []
            
            # Compute SSD for each disparity
           for d in range(min_disp, max_disp+1):
                
                #right patch is within bounds
                if j-d-patch_radius < 0 or j-d+patch_radius >= col_num:
                    continue
                
                # Extract patch from right image
                right_patch = right_img_norm[i - patch_radius:i + patch_radius + 1, 
                                            j - d - patch_radius:j - d + patch_radius + 1]
                
                right_patches.append(right_patch.ravel())
                valid_disps.append(d)
            
           if not right_patches:
                continue
            
            # Convert right patches to array for cdist
           right_patches = np.array(right_patches)  # Shape: (num_valid_disps, patch_size * patch_size)
            
            # Compute SSD using cdist (squared Euclidean distance)
           ssd_values = cdist(left_patch_flat, right_patches, metric='sqeuclidean').flatten()
            
            # Find disparity with minimum SSD
           min_ssd_idx = np.argmin(ssd_values)
           min_ssd = ssd_values[min_ssd_idx]
            
            # Store disparity if valid
           if min_ssd != float('inf'):
               disparity_map[i, j] = valid_disps[min_ssd_idx]
            
            # outlier remvoval
            
    
    return disparity_map