import numpy as np
from keras.datasets import mnist
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D


'''
此函数将标签值由数字变为位置显示，即为One Hot编码
example:
tran_y(9)返回数组的y_ohe数组为【0,0,0,0,0,0,0,0,0,1】
'''


def tran_y(y):
    y_ohe = np.zeros(10)
    y_ohe[y] = 1
    return y_ohe


'''
这一步用于读取mnist数据集中的训练样本的图片和结果和测试样本的图片和结果
X代表样本图片，Y代表标签值，train是训练样本，test是测试样本
'''
(X_train, Y_train) , (X_test, Y_test) = mnist.load_data()
print('训练样本图片大小:', X_train[0].shape)
print('第一个训练样本的标签值:', Y_train[0])

'''
这一段语句，首先将样本和测试图片的的尺寸从28*28的二维数组变为28*28*1的三维数组，然后是将样本图片的数据类型转变为32位浮点型
'''

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')


'''
对像素归一化
'''
X_train /= 255
X_test /= 255


'''
这两句语句的执行逻辑如下：
1，首先使用range函数生成从0到Y_train的长度的增一数组
2，然后从第一步生成的数组的第一个数字开始循环赋值给i，直到最后一个数字
3，根据每一次循环的i值，读取标签数组中的标签值，再将该标签值送入tran_y函数中，返回OneHot编码数组
4，将所有返回的OneHot数组依次排列，通过np.array函数赋值给Y_train_ohe
Y_test_ohr的赋值同理如上
'''

Y_train_ohe = np.array([tran_y(Y_train[i]) for i in range(len(Y_train))])
Y_test_ohe = np.array([tran_y(Y_test[i]) for i in range(len(Y_test))])


'''
生成序贯模型，加入以下层：
2D卷积层：64个过滤器，核函数尺寸为（3，3），步长为（1，1），，输入的样本大小为（28，28，1），使用relu函数作为激活函数
最大统计量2D池化层：大小为（2，2），将图片尺寸变为原来的二分之一
放弃层：0.5，对一半的输入参数权重不更新
以上层重复加入3次
扁平化层：将高维数组压缩成为2维数组
全连接层：128，向下一维输出128维的向量
全连接层：64，向下一维输出64维的向量
全连接层：32，向下一维输出32维的向量
全连接层：10，向下一维输出10维的向量
'''
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1),
                 padding='same', input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1),
                 padding='same', input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1),
                 padding='same', input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(10, activation="relu"))


'''
设置模型的损失函数为交叉熵函数，优化器是adagrad方法，性能评估是精确度
'''
model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])


'''
使用训练样本对模型进行训练，确认数据是测试样本，迭代次数为20次，每批样本的个数为128
'''
model.fit(X_train, Y_train_ohe,validation_data=(X_test, Y_test_ohe),
          epochs=20, batch_size=128)

'''
对测试样本进行测试，verbose参数设为0，表示不显示拟合过程中的输出状态
'''
score = model.evaluate(X_test, Y_test_ohe, verbose=1)

'''
显示模型，保存为'用于mnist的序贯模型.jpg'，显示每一层的网络输入输出参数
'''
plot_model(model, to_file='用于mnist的序贯模型.jpg', show_shapes='true')