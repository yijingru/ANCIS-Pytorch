import numpy as np

def map_mask_to_image(mask, img, color):
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    mskd = img * mask
    clmsk = np.ones(mask.shape) * mask
    clmsk[:, :, 0] = clmsk[:, :, 0] * color[0] * 256
    clmsk[:, :, 1] = clmsk[:, :, 1] * color[1] * 256
    clmsk[:, :, 2] = clmsk[:, :, 2] * color[2] * 256
    img = img + 0.8 * clmsk - 0.8 * mskd
    return np.uint8(img)