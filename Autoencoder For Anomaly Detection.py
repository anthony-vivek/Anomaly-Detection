import os
import cv2
from PIL import Image
from glob import glob
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

# File paths
video_file = 'assignment3_video.avi'
image_folder = 'img_folder'
model_save_path = 'my_autoencoder_model.keras'
anomaly_img = r'img_folder\frame0800.jpg'
non_anomaly_img = r'img_folder\frame0500.jpg'


def convert_video_to_images(img_folder, filename=video_file):
    """
    Converts the video file (assignment3_video.avi) to JPEG images.
    Once the video has been converted to images, then this function doesn't
    need to be run again.
    Arguments
    ---------
    filename : (string) file name (absolute or relative path) of video file.
    img_folder : (string) folder where the video frames will be
    stored as JPEG images.
    """
    # Make the img_folder if it doesn't exist.'
    try:
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)
    except OSError:
        print('Error')

    # Make sure that the abscense/prescence of path
    # separator doesn't throw an error.
    img_folder = f'{img_folder.rstrip(os.path.sep)}{os.path.sep}'
    # Instantiate the video object.
    video = cv2.VideoCapture(filename)

    # Check if the video is opened successfully
    if not video.isOpened():
        print("Error opening video file")

    i = 0
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            im_fname = f'{img_folder}frame{i:0>4}.jpg'
            print('Captured...', im_fname)
            cv2.imwrite(im_fname, frame)
            i += 1
        else:
            break
    video.release()

    if i:
        print(f'Video converted\n{i} images written to {img_folder}')


def load_images(img_dir, im_width=60, im_height=44):
    """
    Reads, resizes and normalizes the extracted image frames from a folder.

    The images are returned both as a Numpy array of flattened images
    (i.e. the images with the 3-d shape (im_width, im_height, num_channels)
    are reshaped into the 1-d shape (im_width x im_height x num_channels))
    and a list with the images with their original number of dimensions
    suitable for display.

    Arguments
    ---------
    img_dir     : (string) the directory where the images are stored.
    im_width    : (int) The desired width of the image.
                        The default value works well.
    im_height   : (int) The desired height of the image.
                        The default value works well.
    Returns
    X           : (numpy.array) An array of the flattened images.
    images      : (list) A list of the resized images.
    """

    images = []
    fnames = glob(f'{img_dir}{os.path.sep}frame*.jpg')
    fnames.sort()
    for fname in fnames:
        im = Image.open(fname)
        # resize the image to im_width and im_height.
        im_array = np.array(im.resize((im_width, im_height)))
        # Convert uint8 to decimal and normalize to 0 - 1.
        images.append(im_array.astype(np.float32) / 255.)
        # Close the PIL image once converted and stored.
        im.close()

    # Flatten the images to a single vector
    X = np.array(images).reshape(-1, np.prod(images[0].shape))

    return X, images


convert_video_to_images(image_folder)

loaded_autoencoder = load_model(model_save_path)


def preprocess_image(img_path, target_size=(44, 60)):

    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array.astype(np.float32) / 255.
    img_array = img_array.reshape((1,) + img_array.shape)
    return img_array


def predict(img_path):

    # This threshold misclassifies some of the initial frames
    # that contain the boat but I still think it is the best
    # given that a lower threshold misclassifies frames without
    # the boat
    threshold = 0.506
    input_img = preprocess_image(img_path)

    # Reconstruct the input image
    reconstructed_img = loaded_autoencoder.predict(input_img)

    # Calculate the loss
    reconstruction_loss = (
        loaded_autoencoder.evaluate(input_img, reconstructed_img, verbose=0)
    )

    # I determined the threshold by plotting the loss
    # Compare the reconstruction loss with our threshold
    if reconstruction_loss > threshold:
        return True  # Anomaly
    else:
        return False  # Not anomaly


# Test the function on an anomaly
image_path = anomaly_img
result = predict(image_path)
print(result)

# Test the function on a non-anomaly
image_path = non_anomaly_img
result = predict(image_path)
print(result)