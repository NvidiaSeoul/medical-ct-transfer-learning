train_dir = '/home/sckit/deeplearning_prj/20260617/Covid19_CT_Image_dataset/train'
test_dir = '/home/sckit/deeplearning_prj/20260617/Covid19_CT_Image_dataset/test'

from tensorflow.keras.preprocessing.image import ImageDataGenerator

batch_size = 4
image_size = 224

# train 이미지 증강 유형 생성
train_image_generator = ImageDataGenerator(
                   rotation_range = 180,
                   width_shift_range=0.2,
                   height_shift_range=0.2,
                   horizontal_flip=True, 
                   vertical_flip=True
 )

# train image를 읽어 들이면서 image를 증강 시켜주는 제너레이터 생성
train_data_gen = train_image_generator.flow_from_directory(
    train_dir, # 불러올 이미지 경로
    batch_size = batch_size,
    shuffle = True,
    # 디렉토리 내부 이미지를 불러올때 어떤 형식으로 라벨링 해서 불러 올꺼냐?
    class_mode = 'categorical',    #  이진 분류 할 때 binary , 다중분류 떄는 ==> 'categorical'
    target_size = (image_size, image_size) # CNN 모델 입력 사이즈로 리사이즈 해라
)


# test 이미지 증강 없이 생성
test_image_generator = ImageDataGenerator()

# test image를 읽어 들이면서 제너레이터 생성
test_data_gen = test_image_generator.flow_from_directory(
    test_dir, # 불러올 이미지 경로
    batch_size = 4,
    shuffle = False,
    # 디렉토리 내부 이미지를 불러올때 어떤 형식으로 라벨링 해서 불러 올꺼냐?
    class_mode = 'categorical',    #  이진 분류 할 때 binary , 다중분류 떄는 ==> 'categorical'
    target_size = (image_size, image_size) # CNN 모델 입력 사이즈로 리사이즈 해라
)

# 디렉토리 별 자동 레벨링 정보를 갖고 있음
print(train_data_gen.class_indices) # {'Covid': 0, 'Normal': 1}

class_levels = list( train_data_gen.class_indices.keys() )
print(class_levels[0], class_levels[1])


# vgg16 모델의 가중치를 가져와서 전이학습 하는 모델 설계
from tensorflow.keras.applications import vgg16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten

# VGG16 모델의 Top층(FC layer) 의 가중치는 가져오지마. 재설계 할꺼야.
# include_top = False
vgg16_layer = vgg16.VGG16(weights='imagenet', include_top = False,
            input_shape=(image_size, image_size, 3))
#vgg16_layer.summary()

for layer in vgg16_layer.layers:
    layer.trainable  = False  # Vgg16 모델의 하단(특징추출역할 layer)는 학습되지 마세요.

newmodel = Sequential()
newmodel.add(vgg16_layer)

# 상단층 재 설계
newmodel.add( Flatten() )
newmodel.add( Dense(units=1024, activation='leaky_relu'))
newmodel.add( Dropout(0.3) )
newmodel.add( Dense(units=2, activation='softmax'))

newmodel.summary()
import tensorflow as tf
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5) # 0.00001
newmodel.compile(loss='categorical_crossentropy', optimizer = optimizer,
                 metrics = ['accuracy'])

import numpy as np
print( int( np.ceil(train_data_gen.samples / test_data_gen.batch_size ) ) )
# print( train_data_gen.samples )
# print( test_data_gen.batch_size )

# 모델 조기종료 등 콜백설정
# 학습 데이터가 적을때는 vgg16 가중치 그냥 동결해서 사용해야 함
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
chckpoint_cb = ModelCheckpoint('./CT_bestmodel.keras',
                               save_best_only=True)
earlystopping_cb = EarlyStopping(
    patience=3, restore_best_weights=True,
)

newmodel.fit(train_data_gen,
             steps_per_epoch = int( np.ceil(train_data_gen.samples / train_data_gen.batch_size ) ),
             epochs = 30,
             validation_data = test_data_gen,
             validation_steps = int( np.ceil(test_data_gen.samples / test_data_gen.batch_size ) ),
             verbose = 1,
              callbacks = [chckpoint_cb, earlystopping_cb]
             )
