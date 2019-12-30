import keras
from keras import layers
import matplotlib.pyplot as plt
import keras.datasets.mnist as mnist
import pandas as pd
import numpy as np



def load_data():
    path = 'source/mnist.npz'
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)

(train_image, train_label), (test_image, test_labels) = load_data()

print(train_image.shape)
# plt.imshow(train_image[0])
# plt.show()

model = keras.Sequential()
# （60000，28，28）-> (60000, 28*28)
model.add(layers.Flatten())
# 降维到64
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
# 降维到10
model.add(layers.Dense(10, activation='softmax'))

# 编译神经网络
model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['acc']
)

# 开始训练
model.fit(train_image, train_label, epochs=50, batch_size=512)

print('查看测试集')
# 测试一下 看起准确性
train_acc = model.evaluate(test_image, test_labels)
print('train_acc', train_acc)

print('train数据集上的得分')
test_acc = model.evaluate(train_image, train_label)
print('test_acc', test_acc)

result10 = np.argmax(model.predict(test_image[:10]), axis=1)
print(result10)

test_result10 = test_labels[:10]
print(test_result10)

# 模型的优化

