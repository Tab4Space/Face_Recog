import cv2, os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt


def generate_image_data(image, filePath):
    Generator = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-6,
        rotation_range=15.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.5,
        zoom_range=[0.8, 2.0],
        channel_shift_range=0.,
        fill_mode='nearest',
        cval=0.,
        horizontal_flip=True,
        vertical_flip=False,
        rescale=None,
        preprocessing_function=None
    )
    
    extendDimImage = image[np.newaxis, :, :, :]
    Generator.fit(extendDimImage)
    
    for i in range(500):
        generateImage = Generator.flow(extendDimImage).next().squeeze()
        changeImage = generateImage.astype(np.uint8)
        grayResizeImage = cv2.resize(cv2.cvtColor(changeImage, cv2.COLOR_RGB2GRAY), (64, 64))
        
        cv2.imwrite('./train/'+filePath.split('.')[0]+'.'+str(i)+'.png', grayResizeImage)


    