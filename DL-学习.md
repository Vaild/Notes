# 深度学习

**为什么进行批归一化 BN？**

找到最快的梯度下降的路径。另外在tensorflow中批归异化是在连接层后面单独再加一下。





### wide & deep 模型

![img](https://pic4.zhimg.com/v2-509773d865632c1183f339833c585fd3_r.jpg)

wide靠记忆，deep靠特征推荐

**使用tensorflow搭建wide-deep模型**

```python
# 开始构建deep
input = keras.layers.Input(shape=xtrain.shape[1:])
hiddenl1 = keras.layers.Dense(100, activation='relu')(input)
hiddenl2 = keras.layers.Dense(100, activation='relu')(hiddenl1)
hiddenl3 = keras.layers.Dense(50, activation='relu')(hiddenl2)

# 这里的这个concat就是wide+deep 他让输入和经过计算的最后一个隐藏层3合并
concat = keras.layers.concatenate([input, hiddenl3])
output = keras.layers.Dense(1)(concat)
model = keras.models.Model(inputs=[input],
                           outputs=output)
```

wide & deep 多输入，为每个模型分别输入不同的数据。

```python
input_wide = keras.layers.Input(shape=[5])
input_deep = keras.layers.Input(shape=[6])

hiddenl1 = keras.layers.Dense(100, activation='relu')(input_deep)
hiddenl2 = keras.layers.Dense(100, activation='relu')(hiddenl1)
hiddenl3 = keras.layers.Dense(50, activation='relu')(hiddenl2)

concat = keras.layers.concatenate([input_wide, hiddenl3])
outputl = keras.layers.Dense(1)(concat)

# 再加一个可以多输出
outputl = keras.layers.Dense(1)(hiddenl3)

model = keras.models.Model(inputs=[input_wide, input_deep],
                           outputs=[outputl, outputl2])

model.compile(loss=keras.losses.MSE, optimizer=keras.optimizers.SGD(0.001))
model.summary()

xtrainwide = xtrain[:, :5]
xtraindeep = xtrain[:, 2:]
xvalidwide = xvalid[:,:5]
xvaliddeep = xvalid[:,2:]
xtestwide = xtest[:,:5]
xtestdeep = xtest[:,2:]
# 在最后训练的时候也要进行多输入
model.fit([xtrainwide, xtraindeep], ytrain, epochs=100,
          validation_data=([xvalidwide, xvaliddeep], yvalid))

```



### 超参数搜索（随机搜索、网格搜索、遗传算法搜索）

>学习率：学习率衰减

手动的话，就写for循环进行搜索。

 自动的网格搜索，用sklearn里面的GridSearchCV这个接口，但是他这个接口里的模型只能放的是sklearn的模型，所以我们要将深度学习的模型包装成sklearn的模型才能进行网格搜索。

但是，在包装的时候，要传入的是一个回调函数的接口。

遇到一个问题，输入层没有写激活函数，造成的结果就是：只有第一个epoch有loss，不能进行反向传播。



#### **包装为sklearn：**

==**tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=build_model)​**==

```python
# 建立一个深度学习的模型接口, 在写的时候与直接写没有什么不同
def build_model(hiddenlayer=1, learning_rate=0.01,
                layersize=30):
    model = keras.models.Sequential()
    # 先建立输入层, 输入层要记得写激活函数
    model.add(keras.layers.Dense(layersize, activation='relu', input_shape=xtrain.shape[1:]))
    for _ in range(hiddenlayer - 1):
        model.add(keras.layers.Dense(layersize, activation='relu'))
    model.add(keras.layers.Dense(1))
    loss = 'mse'
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(loss=loss, optimizer=optimizer)

    return model

# 将深度学习的模型包装为sklearn的模型
sklearn_model = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=build_model)
callbacks = [
    keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)
]

history = sklearn_model.fit(xtrain, ytrain, epochs=10,
                            validation_data=(xvalid, yvalid),
                            callbacks=callbacks)
```

#### 网格搜索代码

**用网格搜索的时候，将搜索的参数用字典写好。**

```python
from sklearn.model_selection import GridSearchCV
param_distribution = {
    'hiddenlayer': [1, 2, 3, 4, 5],
    'learning_rate': [1e-4, 5e-5, 1e-3, 5e-3, 1e-2],
    'layersize': [5, 25, 45, 65]
}

# 这里相当于重新写了一个新的模型
gridsearch_cv = GridSearchCV(sklearn_model, param_distribution, n_jobs=-1)
callbacks = [
    keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)
]
gridsearch_cv.fit(xtrain, ytrain, epochs=50, validation_data=(xvalid, yvalid),
                  callbacks=callbacks)
```



## tensorflow基础API

常量和变量？常量再次进行赋值的时候，地址改变。变量再次赋值地址不变。tensorflow里面不能对常量再次的赋值，也就是assign，除非用等号，但是等号之后地址发生变化。

tensorflow里面的constant，是常量。他是一个tensor 。constant只能转换规则的列表。

矩阵相乘用：`@`

tf 也可以对strings来进行处理。

**EagerTensor**：constant就是EagerTensor. 只能创建规则矩阵

```python
t = tf.constant([[1., 2., 3.], [4., 5., 6.]])
print(t @ tf.transpose(t))
print(t.numpy)
np_t = np.array([[1., 2., 3.], [4., 5., 7.]])
print(tf.constant(np_t))

t4 = tf.constant(['cafe', 'caffee', '咖啡'])
print(tf.strings.length(t4))
print(tf.strings.length(t4, unit='UTF8_CHAR'))
print(tf.strings.unicode_decode(t4,'utf8'))
```



**RaggedTensor**: 创建不规则的矩阵

```python
r = tf.ragged.constant([[11,12],[21,22,23],[],[41]])
print(r)
print(r[1])
print(r[2,:])

r2 = tf.ragged.constant([[51, 52], [], [71]])
print(r.shape, r2.shape)
print(tf.concat((r, r2), axis=0))
print(tf.concat((r, r2), axis=0).shape)
s4 = tf.constant([[10., 20.],
                  [30., 40.],
                  [50., 60.],
                  [70., 80.]])

```

**SparseTensor**： 创建稀疏矩阵

```python
s = tf.SparseTensor(indices=[[0, 1], [1, 0], [2, 3]], # 位置
                    values=[1., 2., 3.], # 值
                    dense_shape=[3, 4] # 维数
                   )
print(s)
print('=================================================')
print(tf.sparse.to_dense(s))

print(s.shape, s4.shape)
# 矩阵相乘
print(tf.sparse.sparse_dense_matmul(s, s4))

s5 = tf.SparseTensor(indices=[[0, 2], [0, 1], [2, 3]],
                     values=[1, 2, 3],
                     dense_shape=[3, 4])

# print(tf.sparse.to_dense(s5))
# 这里直接进行上个操作会报错，是因为，他的indices的顺序写的有问题， 我们需要先进行reorder之后再进行输出就可
s6 = tf.sparse.reorder(s5)
print(tf.sparse.to_dense(s6))
```



**Variable**: 他的value是个tensor， 可以用numpy转为numpy

```python
v = tf.Variable([[1., 2., 3.], [4., 5., 6.]])
print(v)
print(v.value())  # v.value 是一个tensor
print(v.numpy())
# 不能进行改变

#变量用 assign去进行改变。
```

### 自定义损失函数

自己写一个损失函数，然后再model.compile时赋给loss。

```python
# 自定义损失函数
def redef_mse(ytrue, ypred):
    return tf.reduce_mean(tf.square(ytrue - ypred))
# 自定义的损失函数再compile的时候放进去就可以
model.compile(loss=redef_mse, optimizer=keras.optimizers.SGD(0.01))

```



### 自定义层次

从 `keras.layers.Layer` 继承，改写`build` 和 `call` 函数

```python
class CustomizedDenseLayer(keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        self.units = units
        self.activation = keras.layers.Activation(activation)
        super(CustomizedDenseLayer, self).__init__(**kwargs)

    # build 是建立这个层的参数，包含w 和 b
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.units),
                                      initializer='uniform',
                                      trainable=True)

        self.bias = self.add_weight(name='bias',
                                    shape=(self.units,),
                                    initializer='zeros',
                                    trainable=True)
        super(CustomizedDenseLayer, self).build(input_shape)
	
    
	# 对参数进行正向计算
    def call(self, inputs, **kwargs):
        # 完成正向计算
        return self.activation(inputs @ self.kernel + self.bias)
```

 



### @tf.function 转为图结构

![img](https://img-blog.csdn.net/20171221100429378?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2luYXRfMzIwNDM0OTU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

两种方式把python函数转为tensorflow的图

- 转换函数

```python
# 将python函数通过转换的函数 转为 tf的图
scaled_elu_tf = tf.function(scaled_elu)
print(scaled_elu_tf)
print(scaled_elu_tf(tf.constant(-3.)))
print(scaled_elu_tf(tf.constant([-3, -9.36])))

print(scaled_elu_tf.python_function is scaled_elu)
```

- 装饰器

```python
@tf.function
def converge_to_2(n_iters):
    total = tf.constant(0.)
    increment = tf.constant(1.)
    for _ in range(n_iters):
        total += increment
        increment /= 2.0
    return total

print(converge_to_2)
print(converge_to_2(20))
print(converge_to_2(50))
```



### 自定义求导

#### 求一阶导

```python
# 求导
def f(x):
    return 3. * x ** 2 + 2 * x - 1

def approximate_derivative(f, x, eps=1e-3):
    return (f(x+eps) - f(x-eps)) / (eps * 2)

# 求偏导
def g(x1, x2):
    return (x1 + 5) * (x2 ** 2)

def approximate_gradient(g, x1, x2, eps=1e-3):
    dgx1 = approximate_derivative(lambda x: g(x, x2), x1, eps)
    dgx2 = approximate_derivative(lambda x: g(x1, x), x2, eps)
    return dgx1, dgx2

```

用内置的函数求导 `tf.GradientTape`

**必须使用`tf.Variable`进行求导**，不能用tf.constant, 但是`tf.constant`可以看

```python

x1 = tf.Variable(2.)
x2 = tf.Variable(3.)

# 只能用一次
with tf.GradientTape() as tape:
    z = g(x1, x2)

dzx1 = tape.gradient(z, x1)

# 可以永久使用， 但是最后要将资源释放掉  del
with tf.GradientTape(persistent=True) as tape:
    z = g(x1, x2)

dzx1 = tape.gradient(z, x1)
dzx2 = tape.gradient(z, x2)
print(dzx1, dzx2)
del tape

```

```python
# 使用constant 进行查看导数
c1 = tf.constant(2.)
c2 = tf.constant(3.)

with tf.GradientTape() as tape:
    tape.watch(c1)
    tape.watch(c2)
    z = g(c1, c2)

dxc1c2 = tape.gradient(z, [c1, c2])
print(dxc1c2)
```

#### 求二阶导

```python
# 求二阶导
x1 = tf.Variable(2.)
x2 = tf.Variable(3.)

with tf.GradientTape(persistent=True) as outtape:
    with tf.GradientTape(persistent=True) as innertape:
        z = g(x1, x2)
    innergrads = innertape.gradient(z, [x1, x2])
outgrads = [outtape.gradient(innergrad, [x1, x2]) for innergrad in innergrads]

print(innergrads, outgrads)

del innertape
del outtape
# 当然，不求偏导，求普通导数也是一样的
```

#### 模拟梯度下降

```python
def f(x):
    return 3. * x ** 2 + 2 * x - 1
x = tf.Variable(0.0)
learning_rate = 0.1
for _ in range(100):
    with tf.GradientTape() as tape:
        z = f(x)
    dzdx = tape.gradient(z, x)
    # print(dzdx)
    # 这里不能写 x = x.assign_sub（）
    x.assign_sub(learning_rate * dzdx)
print(x)
```

`GradientTape` & `optimizer`

```python
x = tf.Variable(1.)
learning_rate = 0.1
optimizer = keras.optimizers.SGD(lr=learning_rate)
for _ in range(100):
    with tf.GradientTape() as tape:
        z = f(x)
    dzdx = tape.gradient(z, x)
    # 这里不是很懂
    optimizer.apply_gradients([(dzdx, x)])
print(x)
```

更新打印

```python
for i in range(100000):
	print('\r i = ', i, end="") 
```



### 自定义模型训练

```python
epochs = 100
batch_size = 128
step_per_epoch = len(xtrain) // batch_size
optimizer = keras.optimizers.SGD(learning_rate=0.001)
metric = keras.metrics.MeanSquaredError()

def random_batch(x, y, batch_size):
    idx = np.random.randint(0, len(x), batch_size)
    return x[idx], y[idx]

model = keras.models.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape=xtrain.shape[1:]),
    keras.layers.Dense(1)
])

for epoch in range(epochs):
    # 每一个epoch都要对metric进行重置
    metric.reset_states()
    for step in range(step_per_epoch):
        x_batch, ybatch = random_batch(xtrain, ytrain, batch_size)

        with tf.GradientTape() as tape:
            # 通过模型得到预测值
            ypred = model(x_batch)
            # 因为这里得到的预测值是一个二维的，但是我们原有的标签是一个一维的
            ypred = tf.squeeze(ypred)

            # 这里是建立一个可以进行优化的函数
            loss = keras.losses.mean_squared_error(ybatch, ypred)
            metric(ybatch, ypred)
        # 求梯度， 我们为了更新的是模型的参数
        grads = tape.gradient(loss, model.variables)
        # 将梯度与参数打包，方便我们使用优化器进行参数更新。
        gradsa_variable = zip(grads, model.variables)
        # 使用优化器对参数进行更新
        optimizer.apply_gradients(gradsa_variable)
        print('\repoch', epoch , '  train_mse', metric.result().numpy(), end="")
    yvalidpred = model(xvalid)
    yvalidpred = tf.squeeze(yvalidpred)
    validmse = keras.metrics.mean_squared_error(yvalid, yvalidpred)
    print('\t', 'valid_mse: ', validmse)
```

#### 自定义层+自定义模型训练代码

```python
class cus_Dense(keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        self.units = units
        self.activation = keras.layers.Activation(activation)
        super(cus_Dense, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.units),
                                      initializer='uniform',
                                      trainable=True
                                      )
        self.bias = self.add_weight(name='bias',
                                    shape=(self.units, ),
                                    initializer='zeros',
                                    trainable=True)
        super(cus_Dense, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return self.activation(inputs @ self.kernel + self.bias)



epochs = 100
batch_size = 128
step_per_epoch = len(xtrain) // batch_size
metric = keras.metrics.MeanSquaredError()
optimizer = keras.optimizers.SGD(learning_rate=0.001)
# 获得随机数据
def random_batch(x, y, batch_size):
    idx = np.random.randint(0, len(x), batch_size)
    return x[idx], y[idx]


model = keras.models.Sequential(
    [cus_Dense(50, activation='relu', input_shape=xtrain.shape[1:]),
     cus_Dense(50, activation='relu'),
     cus_Dense(25, activation='relu'),
     cus_Dense(1)]
)


for epoch in range(epochs):
    metric.reset_states()
    for step in range(step_per_epoch):
        xbatch, ybatch = random_batch(xtrain, ytrain, batch_size)
        with tf.GradientTape() as tape:
            ypred = model(xbatch)
            ypred = tf.squeeze(ypred)
            loss = keras.losses.mean_squared_error(ybatch, ypred)
            metric(ybatch, ypred)
        grads = tape.gradient(loss, model.variables)
        grads_variables = zip(grads, model.variables)
        optimizer.apply_gradients(grads_variables)

        print('\r epoch', epoch, '   train_mse: ', metric, end="")

    yvalidpre = model(xvalid)
    yvalidpre = tf.squeeze(yvalidpre)
    validmse = tf.losses.mean_squared_error(yvalid, yvalidpre)
    print('\t', 'valid_mse: ', validmse)

```



## Tensorflow DataSets & tfRecord

tensorflow的数据结果 tfrecord，就是以二进制的形式写在磁盘上。

`np.c_[x, y]` 可以将数据集中的x和y 合并到一起

```python
import os
output_dir = 'generate_csv'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
def save2csv(output_dir, data, name_prefix, header=None, n_parts=10):
    path_format = os.path.join(output_dir, '{}_{:02d}.csv')
    filenames = []

    # 把数据分为nparts部分写道文件中去
    # np.array_split是不均等分割， enumerate可以直接返回当前的index和后面所跟的任何东西。
    for file_idx, row_indices in enumerate(np.array_split(np.arange(len(data)), n_parts)):
        path_csv = path_format.format(name_prefix, file_idx)
        filenames.append(path_csv)
        with open(path_csv, 'w', encoding='utf-8') as file:
            if header is not None:
                file.write(header + '\n')
            for row_index in row_indices:
                # 用逗号将字符串拼接起来
                # 这里的repr（）是将col转为一个字符串
                file.write(",".join([repr(col) for col in data[row_index]]))
                file.write('\n')
    return filenames

# np.c_把x和y合并起来
train_data = np.c_[xtrain, ytrain]
test_data = np.c_[xtest, ytest]
valid_data = np.c_[xvalid, yvalid]

header = housing.feature_names + ['target']
header_str = ','.join(header)
train_filename = save2csv('generate_csv', train_data, 'train_data', header_str, n_parts=20)
valid_filename = save2csv('generate_csv', valid_data, 'valid_data', header_str, n_parts=10)
test_filename = save2csv('generate_csv', test_data, 'test_data', header_str, n_parts=10)
```

###  读取csv

```python
# 解析一行
def parse_csv_line(line, n_fields=9):
    defs = [tf.constant([])] * n_fields

    parse_fields = tf.io.decode_csv(line, record_defaults=defs)
    x = tf.stack(parse_fields[:-1])
    y = tf.stack(parse_fields[-1:])
    return x, y

parse_csv_line('1,2,3,4,5,6,7,8,9', 9)


# 整个解析csv的流程
def csv_reader_dataset(filenames, n_readers=5,
                       batch_size=32, n_parse_threads=5,
                       shuffle_buffer_size=10000):
    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.repeat()
    dataset = dataset.interleave(
        lambda filename: tf.data.TextLineDataset(filename).skip(1),
        cycle_length = n_readers
    )
    dataset.shuffle(shuffle_buffer_size) #对数据进行洗牌，混乱
    #map，通过parse_csv_line对数据集进行映射，map只会给函数传递一个参数
    dataset = dataset.map(parse_csv_line,
                          num_parallel_calls=n_parse_threads)
    dataset = dataset.batch(batch_size)
    return dataset

train_set = csv_reader_dataset(train_filename, batch_size=3)
```

### tfrecord

>  二进制直接写，不需要进行转码

在20200821这部分。

还有estimator。



## 卷积神经网络CNN

> 可以用 selu ， 可以更快的收敛，并且效果也很好

### 卷积代码

卷积 -->  批归一化  -->  激活函数  -->  池化

```python
model = keras.models.Sequential([
    # 卷积
    keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu',
                        input_shape=(28, 28, 1)),
    keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
    # 池化
    keras.layers.MaxPool2D(pool_size=2),

])
for i in range(2):
    model.add(keras.layers.Conv2D(filters=64*(i+1), kernel_size=3, padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(filters=64*(i+1), kernel_size=3, padding='same', activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=2))

# 这里把维度信息展平
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
```

### 深度可分离卷积

卷积卷积 -->  批归一化  -->  激活函数  -->  1X1卷积  -->  批归一化  -->  激活函数

可以看到不同的视野。

#### 原理



#### 代码

```python
model = keras.models.Sequential([
    keras.layers.Conv2D(32, kernel_size=3, padding='same',
                        activation='selu',
                        input_shape=(28, 28, 1)),
    keras.layers.SeparableConv2D(filters=32, kernel_size=3,
                                 padding='same',
                                 activation='selu'),
    keras.layers.MaxPool2D(pool_size=2)
])

for i in range(2):
    model.add(keras.layers.SeparableConv2D(filters=64 * (i + 1),
                                           padding='same',
                                           kernel_size=3,
                                           activation='selu'))
    model.add(keras.layers.SeparableConv2D(filters=64 * (i + 1),
                                           padding='same',
                                           kernel_size=3,
                                           activation='selu'))
    model.add(keras.layers.MaxPool2D(pool_size=2))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='selu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

<img src="https://i.loli.net/2021/03/09/zqVb5egkwoXSsEK.png" alt="image-20210309204451643" style="zoom:50%;" />

刚开始训练虽然收敛很快，但是很容易过拟合。

所以我在卷积池化之后加了一个Dropout第一次drop25%， 在展平之前drop掉50%。

![image-20210310091228050](https://i.loli.net/2021/03/10/FfVPYetL5A1monD.png)

![image-20210310091521201](https://i.loli.net/2021/03/10/e2aShcLzuF8GHUC.png)



kaggle连接

## 循环神经网络RNN





hyperas与keras结合