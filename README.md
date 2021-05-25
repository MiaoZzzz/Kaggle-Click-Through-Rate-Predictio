Kaggle网站：https://www.kaggle.com/c/avazu-ctr-prediction/submissions?sortBy=date&group=successful&page=1&pageSize=100
同样，我们使用数据集前1000万条数据用于构建训练数据集和验证数据集，其中训练集和验证集的比例为0.8：0.2。
3.2.1  首先对数据集进行打乱，并且从原有特征中挑选特征进行特征工程。

```python
#打乱数据顺序
train_raw = train_raw.sample(frac=1).reset_index(drop=True)

#挑选训练特征
train_raw = train_raw[['click', 'hour','C1', 'banner_pos']]
train_raw = train_raw[:100000]

#划分训练集和验证集
train_df, val_df = train_test_split(train_raw, test_size=0.2)

#生成训练标签
train_labels = np.array(train_df.pop('click'))
bool_train_labels = train_labels != 0
val_labels = np.array(val_df.pop('click'))

#生成训练特征
train_features = np.array(train_df)
val_features = np.array(val_df)

#数据标准化
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
val_features = scaler.transform(val_features)

train_features = np.clip(train_features, -5, 5)
val_features = np.clip(val_features, -5, 5)

#查看正负样本比例
neg, pos = np.bincount(train_labels)
total = neg + pos
print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))
```
3.2.2 Oversample
由于数据中标签分布不平均，在80000个样本中正样本为13286个，所占比例为 16.61%。因此我们采用过采样的方法对小样本进行补全，使得正负样本的比例大概一致。

```python
#对小样本进行过采样
pos_features = train_features[bool_train_labels]
neg_features = train_features[~bool_train_labels]

pos_labels = train_labels[bool_train_labels]
neg_labels = train_labels[~bool_train_labels]

ids = np.arange(len(pos_features))
choices = np.random.choice(ids, len(neg_features))

res_pos_features = pos_features[choices]
res_pos_labels = pos_labels[choices]
```
3.2.3模型构建

使用tensorflow中keras.Sequential构建三层MLP模型，hidden lyaer神经元个数分别为16,16,1，激活函数分别为relu,relu,sigmoid。使用Adam优化器，损失函数为交叉熵损失。

```python
#构建训练模型
METRICS = [
      keras.metrics.BinaryAccuracy(name='accuracy'),
]


def make_model(metrics=METRICS, output_bias=None):
  if output_bias is not None:
    output_bias = tf.keras.initializers.Constant(output_bias)
  model = keras.Sequential([
      keras.layers.Dense(
          16, activation='relu',
          input_shape=(train_features.shape[-1],),kernel_initializer='he_normal'),
      
      keras.layers.Dense(16, activation='relu',kernel_initializer='he_normal'),
      
      #keras.layers.Dropout(0.5),
      keras.layers.Dense(1, activation='sigmoid',kernel_initializer='he_normal'),
                         #bias_initializer=output_bias),
  ])

  model.compile(
      optimizer=keras.optimizers.Adam(lr=0.000001),
      loss=keras.losses.BinaryCrossentropy(),
      metrics=metrics)

  return model
```
