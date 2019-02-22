#%%
import cv2 as cv2
import keras
# from keras.preprocessing.image.
import cv2
from os import listdir
from sklearn.preprocessing import StandardScaler
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras.utils import to_categorical
from keras.models import save_model

import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D, MaxPooling2D, Flatten, Activation, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

#%%

path = 'C:\\Users\\pedalo\\Documents\\Hackathon2LTU\\Tobacco\\train/'
dirs = listdir(path)

print(dir)

# path2 = 'C:\\Users\\pedalo\\Documents\\Hackathon2LTU\\Tobacco\\train\\ADVE'
# images = listdir(path2)

# print(images)

X = []
y = []

#%%
# for dir in dirs:
#     new_path = "C:\\Users\\pedalo\\Documents\\Hackathon2LTU\\Tobacco\\train\\"+dir+"\\"
#     # print(new_path) 
#     imagesList = listdir(new_path)

#     print(new_path)
#     for imgFile in imagesList:
#         # print(new_path + imgFile)
#         image = cv2.imread(new_path + imgFile, -1)
#         # image = np.float32(image)
#         # print(image)

#         if image is None:
#             print(new_path + imgFile)
#         else:
#             X.append(image)
#             y.append(dir)
#     # print(image)
#%%
print(len(X))
print(y[:5])

encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
#%%
print(X[:4])
print(dummy_y[:4])

#%%

# sc = StandardScaler()

# for i in range(len(X)):
#     # print(elem.shape)
#     sc.fit(X[i])

#     X[i] = sc.transform(np.float32(X[i]))



# encode class values as integers
#%%
# train = [np.array(elem) for elem in X]
# # print(train)

# for elem in X:
#     print(len(elem), len(elem[0]),len(elem[0][0]), elem[0][0][0])
# print(x_train.shape)
# for i in range(2783):
    # train = np.array([X[i]])
# X = np.array(X)
# print(len(train))
# for elem in X:
#     print((elem))

#%%
def get_im(path):
    # Load as grayscale
    img = cv2.imread(path, 0)
    # Reduce size
    # print(img)
    resized = cv2.resize(img, (128, 96))
    # print(resized)
    return resized
#%%

def load_train():
    X_train = []
    y_train = []
    y_encoded = []
    i = 0
    print('Read train images')
    for dir in dirs:
        print(f'Load folder {dir}')
        # i = 0
        # path = os.path.join('..','train', + str(j), '*.jpg')
        # files = glob.glob(path)
        # for fl in files:
        imagesList = listdir(new_path)
        # y_train.append(dir)
        for im in imagesList:
            img = get_im(new_path + im)
            X_train.append(img)
            y_train.append(dir)
            y_encoded.append(i)
            # print(X_train[0])
            # y_train.append(j)
        i+=1
    return X_train, y_train
#%%
train_data, train_target = load_train()
img_rows, img_cols = 96, 128
#%%
encoder = LabelEncoder()
encoder.fit(train_target)
encoded_Y = encoder.transform(train_target)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
print(dummy_y)
#%%

print(len(train_target))
print(train_target)
print(len(train_data))
#%%
# train_target = np.array(train_target, dtype=np.uint8)
train_data = np.array(train_data, dtype=np.uint8)
# train_target = np.array(train_target, dtype=np.uint8)
# print(train_target[0])
# train_target = np_utils.to_categorical(train_target, 10)
# train_target = to_categorical(train_target, num_classes=10)
train_data = train_data.reshape(train_data.shape[0], img_rows, img_cols, 1)
train_data = train_data.astype('float32')
train_data /= 255
print('Train shape:', train_data.shape)
print(train_target)
print(train_data.shape[0], 'train samples')




#%%


# X = X.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

model = Sequential()
# model.add(Conv2D())
model.add(Conv2D(32, (3, 3), input_shape=(96, 128, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(32, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dropout(0.3))

# model.add(Dense(400))
# model.add(Activation('relu'))
# model.add(Dropout(0.3))
# model.add(Dense(300))
# model.add(Activation('relu'))
# model.add(Dropout(0.3))

model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print(model.summary())
model.fit(train_data, dummy_y, batch_size=64, epochs=100, verbose=2, validation_split=0.1)

#%%
