from keras.models import Model
from keras.layers import Input, Conv2D, Output

mdl_height = 384/4
mdl_width = 512/4
batch_size = 16
verbose = 16
disp = 16
save = 32

layer_out = Output(shape=(mdl_height, mdl_width, 2))
layer1 = Input(shape=(mdl_height, mdl_width, 8))
layer2 = Conv2D(filters=32, kernel_size=7, strides=1, padding='same', activation='elu', use_bias=True)(layer1)
layer3 = Conv2D(filters=64, kernel_size=7, strides=1, padding='same', activation='elu', use_bias=True)(layer2)
layer4 = Conv2D(filters=128, kernel_size=7, strides=1, padding='same', activation='elu', use_bias=True)(layer3)
layer5 = Conv2D(filters=128, kernel_size=7, strides=1, padding='same', activation='elu', use_bias=True)(layer4)
layer6 = Conv2D(filters=128, kernel_size=7, strides=1, padding='same', activation='elu', use_bias=True)(layer5)
layer7 = Conv2D(filters=128, kernel_size=7, strides=1, padding='same', activation='elu', use_bias=True)(layer6)
layer8 = Conv2D(filters=256, kernel_size=7, strides=1, padding='same', activation='elu', use_bias=True)(layer7)
layer9 = Conv2D(filters=256, kernel_size=7, strides=1, padding='same', activation='elu', use_bias=True)(layer8)
layer10 = Conv2D(filters=256, kernel_size=7, strides=1, padding='same', activation='elu', use_bias=True)(layer9)
layer11 = Conv2D(filters=256, kernel_size=7, strides=1, padding='same', activation='elu', use_bias=True)(layer10)

pred2 = Conv2D(filters=2, kernel_size=7, strides=1, padding='same', activation='linear', use_bias=True)(layer2)
pred3 = Conv2D(filters=2, kernel_size=7, strides=1, padding='same', activation='linear', use_bias=True)(layer3)
pred4 = Conv2D(filters=2, kernel_size=7, strides=1, padding='same', activation='linear', use_bias=True)(layer4)
pred5 = Conv2D(filters=2, kernel_size=7, strides=1, padding='same', activation='linear', use_bias=True)(layer5)
pred6 = Conv2D(filters=2, kernel_size=7, strides=1, padding='same', activation='linear', use_bias=True)(layer6)
pred7 = Conv2D(filters=2, kernel_size=7, strides=1, padding='same', activation='linear', use_bias=True)(layer7)
pred8 = Conv2D(filters=2, kernel_size=7, strides=1, padding='same', activation='linear', use_bias=True)(layer8)
pred9 = Conv2D(filters=2, kernel_size=7, strides=1, padding='same', activation='linear', use_bias=True)(layer9)
pred10 = Conv2D(filters=2, kernel_size=7, strides=1, padding='same', activation='linear', use_bias=True)(layer10)
pred11 = Conv2D(filters=2, kernel_size=7, strides=1, padding='same', activation='linear', use_bias=True)(layer11)



