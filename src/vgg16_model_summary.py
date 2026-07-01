from tensorflow.keras.applications import vgg16

vgg16model = vgg16.VGG16()  # vgg16 모델 네트워크 구조와 가중치까지 모두 로딩
vgg16model.summary()

predict_img_dir = '/home/sckit/deeplearning_prj/20260616/testimage_dataset/'
import os
os.chdir('/home/sckit/deeplearning_prj/20260616/testimage_dataset')

fileinfolist = []
for file in os.listdir():
    fileinfolist.append(predict_img_dir + file)

print(fileinfolist)

print(fileinfolist[0])

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

img = load_img(fileinfolist[6] , target_size = (224, 224))
image = img_to_array(img) # 이미지 객체를 넘파이 배열로 변경
print(image.shape)

image = image.reshape(1, 224, 224, 3)
print(image.shape)

image = vgg16.preprocess_input(image)
print(image)

pred = vgg16model.predict(image)
#print(pred)
# import numpy as np
# print( np.argmax(pred[0]) )
# print(vgg16model.classe)

labels = vgg16.decode_predictions(pred)
print(labels)