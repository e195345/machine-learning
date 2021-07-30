#ライブラリのインポート
import numpy as np
from sklearn import svm, metrics
from scipy.io import arff
from sklearn.model_selection import train_test_split
import matplotlib as plt


#データ読み込み
filename = "Dry_Bean_Dataset.arff"
dataset, meta = arff.loadarff(filename)


#データの整形
ds=np.asarray(dataset.tolist(), dtype=np.str_)
features, data_class = np.split(ds, [16], axis=1)
features = np.array(features, dtype='float64')


#標準化
def standardization(x, axis=None, ddof=0):
    x_mean = x.mean(axis=axis, keepdims=True)
    x_std = x.std(axis=axis, keepdims=True, ddof=ddof)
    return (x - x_mean) / x_std

standard_features = standardization(features, axis=0)


#トレーニングデータとテストデータに分割
features_train, features_test, class_train, class_test = train_test_split(standard_features, data_class.flatten(), random_state = 0)


#モデルの構築
clf = svm.SVC(kernel='linear', tol=0.01)


#学習
clf.fit(features_train, class_train)
pre = clf.predict(features_test)
ac_score = metrics.accuracy_score(class_test, pre)
print("精度 : ", ac_score)

