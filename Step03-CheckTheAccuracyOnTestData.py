# now , we will check our model within a new test data
# than , we will run a prediction on an image

from locale import normalize
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import keras
import cv2
import tensorflow as tf
TEST_DIR = "test"

# Parametros
# ===================
batch_size = 1

test_datagen = image.ImageDataGenerator( rescale= 1. / 255)
test_data = test_datagen.flow_from_directory(directory=TEST_DIR , target_size=(224,224) , batch_size=batch_size , class_mode='binary')

# lets print the classes :

print("test_data.class_indices: ", test_data.class_indices)

#load the saved model :

model = load_model('MyBestModel.h5')

#print(model.summary() )

acc = model.evaluate(x=test_data)[1]

print(acc)

# load an image from the test folder
#imagePath = "test/Camilo/prueba_Camilo2.jpg"
imagePath = "test/Keanu/prueba_Keanu1.jpg"

img = tf.keras.utils.load_img(imagePath,target_size=(224,224))

i = tf.keras.utils.img_to_array(img) # convert to array
i = i / 255 # -> normalize to our model
print(i.shape)

input_arr = np.array([i]) # add another dimention 
print(input_arr.shape)

# run the prediction
predictions = model.predict(input_arr)[0][0]
print(predictions)

# since it is binary if the result is close to 0 it is Camilo , and if it close to 1 it is Keanu
result = round(predictions)
if result == 0 :
    text = 'Camilo'
else :
    text = "Keanu"

print(text)

imgResult = cv2.imread(imagePath)
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(imgResult, text, (0,20), font, 0.8 , (255,0,0),2 )
cv2.imshow('img', imgResult)
cv2.waitKey(0)
cv2.imwrite("test/predictImage.jpg",imgResult)