import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier

# 分类结果可视化
def show_classification_result(model, feature, groundtruth):
    test_label = model.predict(
    feature, 
    prediction_type='Probability', 
    ntree_start=0, 
    ntree_end=model.get_best_iteration(),
    thread_count=-1, 
    verbose=None)
    positive = 0
    negative = 0
    positive_truth = 0
    negative_truth = 0
    total_width, n = 0.8, 2
    width = total_width / n
    x = np.arange(2)
    x = x - (total_width - width) / 2
    for i in range(len(test_label)):
        if test_label[i][0] > test_label[i][1]:
            negative += 1
        else:
            positive += 1
    for i in range(len(groundtruth)):
        if groundtruth[i] == 1:
            positive_truth += 1
        else:
            negative_truth += 1
    print(positive,negative, positive_truth, negative_truth)
    names = ['positive', 'negative']
    a = [positive, negative]
    b = [positive_truth, negative_truth]
    plt.bar(x, a, width=width, label='predict')
    plt.bar(x + width, b, width=width, label='ground-truth')
    plt.xticks(x, names)
    plt.legend()
    plt.show()
    # 计算precision，recall
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(test_label)):
        if test_label[i][0] > test_label[i][1] and groundtruth[i] == 1:
            fn += 1
        elif test_label[i][0] > test_label[i][1] and groundtruth[i] == 0:
            tn += 1
        elif test_label[i][0] <= test_label[i][1] and groundtruth[i] == 1:
            tp += 1
        elif test_label[i][0] <= test_label[i][1] and groundtruth[i] == 0:
            fp += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    print('f1 score = ', f1)
    plt.bar(['precision', 'recall'], [precision, recall])
    plt.show()

full_data = pd.read_csv('train.csv', sep=',', nrows=20)
print(full_data.columns)
full_test = pd.read_csv('test.csv', sep = ',')
test_feature = full_test.iloc[:, 1:]
# 90%作为训练集，10%作为测试集
n = int(len(full_data) * 0.9)
print(full_data.iloc[n:, 1])
# 训练集
train_feature = full_data.iloc[:n, 2:]
train_label = full_data.iloc[:n, 1]
#验证集
val_feature = full_data.iloc[n:, 2:]
val_label = full_data.iloc[n:, 1]
# 第一次训练
model = CatBoostClassifier(
    iterations = 50, 
    learning_rate = 0.5,
    task_type = 'GPU',
    loss_function = 'Logloss'
)
cat_features = list(range(0, 22))
model.fit(train_feature, y = train_label, cat_features = cat_features, eval_set=(val_feature, val_label), verbose = 10)
# 验证集上的可视化
show_classification_result(model, val_feature, val_label)
test_label = model.predict(
    test_feature, 
    prediction_type='Probability', 
    ntree_start=0, 
    ntree_end=model.get_best_iteration(),
    thread_count=-1, 
    verbose=None
)
test_id = full_test.iloc[:, 0:1]  # 得到id
test_id.join(pd.DataFrame(test_label))    # id和对应的预测结果合并成一个表
submission = pd.read_csv('sampleSubmission.csv')   # 输出表
submission["click"] = test_label[:, 1]   # 写入数据
submission.to_csv("submission.csv", index=False)
print(submission.head())

# 选择部分特征
train_feature = full_data.loc[:n, ['device_ip', 'site_id','app_id', 'C14', 'site_domain', 'device_id']]
train_label = full_data.loc[:n, ['click']]
val_feature = full_data.loc[n:, ['device_ip', 'site_id','app_id', 'C14', 'site_domain', 'device_id']]
cat_features = list(range(0, 6))
print(train_feature)
print(type(train_label))
print(val_feature)
print(val_label)

# 第二次训练
model = CatBoostClassifier(
    iterations = 50, 
    learning_rate = 0.5,
    task_type = 'GPU',
    loss_function = 'Logloss'
)
cat_features = list(range(0, 22))
model.fit(train_feature, y = train_label, cat_features = cat_features, eval_set=(val_feature, val_label), verbose = 10)
test_label = model.predict(
    test_feature, 
    prediction_type='Probability', 
    ntree_start=0, 
    ntree_end=model.get_best_iteration(),
    thread_count=-1, 
    verbose=None
)
test_id = full_test.iloc[:, 0:1]  # 得到id
test_id.join(pd.DataFrame(test_label))    # id和对应的预测结果合并成一个表
submission = pd.read_csv('sampleSubmission.csv')   # 输出表
submission["click"] = test_label[:, 1]   # 写入数据
submission.to_csv("submission.csv", index=False)
print(submission.head())

# 样本均衡
positive = 0
negative = 0
for i in train_label['click']:
    if i == 0:
        negative += 1
    else:
        positive += 1
print(positive, negative)

# 训练集
train_feature = full_data.iloc[:n, 2:]
train_label = full_data.iloc[:n, 1]
#验证集
val_feature = full_data.iloc[n:, 2:]
val_label = full_data.iloc[n:, 1]
# 删除训练集负样本
remove = negative - positive
drop_list = []
for i in range(len(train_label)):
    if train_label.iloc[i] == 0:
        drop_list.append(i)
        remove -= 1
        if remove == 0:
            break
train_feature.drop(drop_list, inplace=True)
train_label.drop(drop_list, inplace=True)
train_label.reset_index(drop=True, inplace=True)
train_feature.reset_index(drop=True, inplace=True)

# 删除验证集负样本
val_feature.reset_index(drop=True, inplace=True)
val_label.reset_index(drop=True, inplace=True)
positive = 0
negative = 0
positive_sample = set()
for i in range(len(val_label)):
    if val_label[i] == 0:
        negative += 1
    else:
        positive += 1
        positive_sample.add(i)

val_feature.reset_index(drop=True, inplace=True)
val_label.reset_index(drop=True, inplace=True)
remove = negative - positive
drop_list = []
for i in range(len(val_label)):
    if train_label.iloc[i] == 0:
        drop_list.append(i)
        remove -= 1
        if remove == 0:
            break
val_feature.drop(drop_list, inplace=True)
val_label.drop(drop_list, inplace=True)
val_label.reset_index(drop=True, inplace=True)
val_feature.reset_index(drop=True, inplace=True)

# 再次训练
test_feature = full_test.iloc[:, 1:]
test_label = model.predict(
    test_feature, 
    prediction_type='Probability', 
    ntree_start=0, 
    ntree_end=model.get_best_iteration(),
    thread_count=-1, 
    verbose=None
)
test_id = full_test.iloc[:, 0:1]  # 得到id
test_id.join(pd.DataFrame(test_label))    # id和对应的预测结果合并成一个表
submission = pd.read_csv('sampleSubmission.csv')   # 输出表
submission["click"] = test_label[:, 1]   # 写入数据
submission.to_csv("submission.csv", index=False)
print(submission.head())