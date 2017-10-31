import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
import sklearn
from sklearn.model_selection import train_test_split

def loadDrivingLog(path='./data'):
    """
    Returns the lines from a driving log with base directory `path`.
    """
    lines = []
    with open(path + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines

def loadImages(log_line, path='./data', correction=0.2):
    """
    Returns the center, left, and right image including corresponding steering angles.
    """
    images = []
    angles = []

    # Extract every camera image in a line: center, left, right
    for i in range(3):
        source_path = log_line[i]
        filename = source_path.split('/')[-1]
        current_path = path + '/IMG/' + filename
        image = cv2.imread(current_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    
    # Extract and calculate steering angles
    steering_center = float(log_line[3])
    angles.append(steering_center)               # center camera
    angles.append(steering_center + correction)  # left camera
    angles.append(steering_center - correction)  # right camera
    return images, angles

def augmentImages(images, angles):
    """
    Returns augmented data based on the provided image and steering angle.
    Augmentation techniques include extraction of center, left, and right
    image as well as flipping the image/angle
    """
    augmented_images = images
    augmented_angles = angles

    # Data augmentation
    for i in range(len(images)):
        image = images[i]
        angle = angles[i]
        # Flip image
        image_flipped = np.fliplr(image)
        angle_flipped = -angle
        augmented_images.append(image_flipped)
        augmented_angles.append(angle_flipped)

    return augmented_images, augmented_angles

def nvidiaModel():
    """
    Returns a model using the NVIDIA architecture as shown in the lecture. 
    """
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

def generator(samples, batch_size=32):
    """
    Generate the required images and angles for training and validation.
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            batch_images = []
            batch_angles = []
            for batch_sample in batch_samples:
                images, angles = loadImages(batch_sample)
                images, angles = augmentImages(images, angles)
                batch_images.extend(images)
                batch_angles.extend(angles)
            
            X_train = np.array(batch_images)
            y_train = np.array(batch_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Load samples from driving log
samples = loadDrivingLog()

# Sample splitting
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print('Train samples: {}'.format(len(train_samples)))
print('Validation samples: {}'.format(len(validation_samples)))

# Create generators
train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

# Create model
model = nvidiaModel()

# Compile and train the model
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch= \
            len(train_samples), validation_data=validation_generator, \
            nb_val_samples=len(validation_samples), nb_epoch=3)

model.save('model.h5')

print(history_object.history.keys())
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.history['val_loss'])
