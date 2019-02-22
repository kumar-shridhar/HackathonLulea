#%%
import cv2 as cv2
import keras
# from keras.preprocessing.image.
import cv2
from os import listdir
from sklearn.preprocessing import StandardScaler
import numpy as np



import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
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


for dir in dirs:
    new_path = "C:\\Users\\pedalo\\Documents\\Hackathon2LTU\\Tobacco\\train\\"+dir+"\\"
    # print(new_path) 
    imagesList = listdir(new_path)

    print(new_path)
    for imgFile in imagesList:
        # print(new_path + imgFile)
        image = cv2.imread(new_path + imgFile, -1)
        image = np.float32(image)

        # print(image)

        if image is None:
            print(new_path + imgFile)
        else:
            X.append(image)
            y.append(dir)
    # print(image)
#%%
print(len(X))
print(y[:5])
#%%

sc = StandardScaler()

for i in range(len(X)):
    # print(elem.shape)
    sc.fit(X[i])

    X[i] = sc.transform(X[i])


#%%
print(X[:4])

#%%
