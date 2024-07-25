import math
import skimage.transform
import numpy as np
import skimage.io
import os

'''
you use this just before passing any image to a CNN
which usually expects square images
however your input images can be of variable size
you don't want to just squash the images to a square
because you will lose valuable aspect ratio information
you want to resize while preserving the aspect ratio
these 2 functions perform this resizing behaviour
images are assumed to be formatted as Height, Width, Channels
we will use bound_image_dim to first bound the image to a range
the smallest dimension will be scaled up to the min_size
the largest dimension will be scaled down to the max_size
then afterwards you square_pad_image to pad the image to a square
filling the empty space with zeros
'''

'''
To preserve aspect ratio:
1. you must first resize to the target size by scale of the largest dimension. This will still be non-square still
2. For the smaller dimension, add padding to equal the target size
3. Do this for each images
'''

def bound_image_dim(image, min_size=None, max_size=None):
    if (max_size is not None) and \
       (min_size is not None) and \
       (max_size < min_size):
        raise ValueError('`max_size` must be >= to `min_size`')
    dtype = image.dtype
    (height, width, *_) = image.shape
    # scale the same for both height and width for fixed aspect ratio resize
    scale = 1
    # bound the smallest dimension to the min_size
    if min_size is not None:
        image_min = min(height, width)
        scale = max(1, min_size / image_min)
    # next, bound the largest dimension to the max_size
    # this must be done after bounding to the min_size
    if max_size is not None:
        image_max = max(height, width)
        if round(image_max * scale) > max_size:
            scale = max_size / image_max
    if scale != 1:
        image = skimage.transform.resize(
            image, (round(height * scale), round(width * scale)),
            order=1,
            mode='constant',
            preserve_range=True)
    return image.astype(dtype)


def square_pad_image(image, size):
    (height, width, *_) = image.shape
    if (size < height) or (size < width):
        raise ValueError('`size` must be >= to image height and image width')
    pad_height = (size - height) / 2
    pad_top = math.floor(pad_height)
    pad_bot = math.ceil(pad_height)
    pad_width = (size - width) / 2
    pad_left = math.floor(pad_width)
    pad_right = math.ceil(pad_width)
    return np.pad(
        image, ((pad_top, pad_bot), (pad_left, pad_right), (0, 0)),
        mode='constant')
    
def preprocess_image(file_path, output_size):
    # Read the image
    image = skimage.io.imread(file_path)
    
    # Resize the image maintaining the aspect ratio
    scale = output_size / max(image.shape[:2])
    new_shape = (round(image.shape[0] * scale), round(image.shape[1] * scale))
    resized_image = skimage.transform.resize(
        image, new_shape, mode='constant', preserve_range=True, anti_aliasing=True
    ).astype(image.dtype)
    
    # Calculate padding sizes
    pad_height = (output_size - resized_image.shape[0]) // 2
    pad_width = (output_size - resized_image.shape[1]) // 2
    
    # Apply padding to make the image square
    padded_image = np.pad(
        resized_image,
        (
            (pad_height, output_size - pad_height - resized_image.shape[0]),
            (pad_width, output_size - pad_width - resized_image.shape[1]),
            (0, 0)
        ),
        'constant',
        constant_values=0
    )
    
    return padded_image

if __name__ == '__main__':
    input_dir = './data/M2FPA/Test_Bins_all_pitch_cropped' # Set the directory paths

    # Set the desired size for padding
    desired_size = 112

    # Iterate over the files in the input directory
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            # Check for image files (you might want to check for specific extensions)
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                image = skimage.io.imread(file_path) # Read the image            
                squared_image = preprocess_image(file_path, desired_size) # Process the image
                skimage.io.imsave(file_path, squared_image) # Save the processed image


    print("Processing complete.")
        