import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.impute import SimpleImputer, KNNImputer
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten


def fill_missing_values(X, n_neighbors=75, method="KNN"):
    # normalization
    X_std = np.nanstd(X, axis=0, keepdims=True)
    X_ave = np.nanmean(X, axis=0, keepdims=True)
    X_norma = (X - X_ave) / X_std

    # use KNNImputer
    imputer = KNNImputer(missing_values=np.nan, n_neighbors=n_neighbors, weights='distance') if method == "KNN" \
        else SimpleImputer(missing_values=np.nan, strategy='median')

    X_norma_fixed = imputer.fit_transform(X_norma)

    return X_norma_fixed


def expand_dataset(X_train, y_train, num_class=4):
    sample = np.c_[X_train, y_train]
    sample_by_class = [sample[(sample[:, -1] == i)] for i in range(num_class)]
    sample_sizes = [sample_class.shape[0] for sample_class in sample_by_class]
    expand_ratio = np.round(np.max(np.array(sample_sizes)) / sample_sizes)
    expanded_sample_by_class = [np.repeat(sample_by_class[i], expand_ratio[i], axis=0) for i in range(num_class)]
    expanded_sample = np.concatenate(expanded_sample_by_class)
    return expanded_sample[:, :-1], expanded_sample[:, -1]

X_train_data = pd.read_csv('X_train_feature_fusion2.csv')
y_train_data = pd.read_csv('y_train.csv')
X_test_data = pd.read_csv('X_test_feature_fusion2.csv')

indices_test = np.array(X_test_data)[:,0]
indices_train = np.array(X_train_data)[:,0]
X_test = np.array(X_test_data)[:,1:]
y_train = np.array(y_train_data)[:,1]
X_train = np.array(X_train_data)[:,1:]

X_train = fill_missing_values(X_train, method="median")
X_test = fill_missing_values(X_test, method="median")
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)


def fit_model_and_pred(X_train, y_train, X_val, y_val, X_test):
    model = Sequential()
    model.add(Dense(32, activation='softmax', input_dimension=692))
    model.add(Dropout(rate=0.25))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)

    y_val_pred = model.predict(X_val)
    score = f1_score(y_val, y_val_pred, average='micro')
    y_pred = model.predict(X_test)
    return score, y_pred


def train_k_fold_pred(X, y, X_test, fold_num=10):
    kf = KFold(n_splits=fold_num, random_state=None, shuffle=False)
    kf.get_n_splits(X)
    test_score = 0.0
    y_pred_list = []

    for train_index, test_index in kf.split(X):
        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index]
        # X_train, y_train = expand_dataset(X_train, y_train)

        score, y_pred = fit_model_and_pred(X_train, y_train, X_val, y_val, X_test)
        y_pred_list.append(y_pred)
        print('The obtained validation f1 score is : ', score)
        test_score += score
    print("Validation score: %f" % (test_score / fold_num))

    y_pred_list = np.array(y_pred_list)
    y_test_predict = []
    for i in range(y_pred_list.shape[1]):
        item = y_pred_list[:, i]
        a = item[item == 0].shape
        b = item[item == 1].shape
        c = item[item == 2].shape
        d = item[item == 3].shape
        candidate = [a, b, c, d]
        y_test_predict.append(np.argmax(candidate))
    y_test_predict = np.array(y_test_predict)
    return test_score / fold_num, y_test_predict


def train_k_fold_pred_trick(X, y, X_test, fold_num=10):
    kf = KFold(n_splits=fold_num, random_state=None, shuffle=False)
    kf.get_n_splits(X)
    test_score = 0.0
    y_pred_list = []

    for train_index, test_index in kf.split(X):
        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index]
        score, y_pred = fit_model_and_pred(X_train, y_train, X_val, y_val, X_test)
        print('The obtained validation r1 score is : ', score)
        test_score += score
        if score > 0.81:
            y_pred_list.append(y_pred)
    print("Validation score: %f" % (test_score / fold_num))

    y_pred_list = np.array(y_pred_list)
    y_test_predict = []
    for i in range(y_pred_list.shape[1]):
        item = y_pred_list[:, i]
        a = item[item == 0].shape
        b = item[item == 1].shape
        c = item[item == 2].shape
        d = item[item == 3].shape
        candidate = [a, b, c, d]
        y_test_predict.append(np.argmax(candidate))
    y_test_predict = np.array(y_test_predict)
    return test_score / fold_num, y_test_predict

_, y_pred = train_k_fold_pred(X_train, y_train, X_test, fold_num=5)

sample =  pd.read_csv("sample.csv")
sample["y"] = y_pred
sample.to_csv("output.csv", index = False)