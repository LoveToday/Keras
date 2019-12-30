import keras
import numpy as np
import matplotlib.pyplot as plt

from keras import layers

x = np.linspace(0, 100, 30)
y = 3 * x + 7 + np.random.randn(30) * 6

print(x, y)

# plt.scatter(x, y)
# plt.show()

# 顺序模型
model = keras.Sequential()
model.add(layers.Dense(1, input_dim=1))

# model.summary()

# 编译模型
model.compile(optimizer='adam',
loss='mse'
)
# 训练模型
model.fit(x, y, epochs=5000)

model.predict(x)

plt.scatter(x, y, c='r')
plt.plot(x, model.predict(x))
plt.show()