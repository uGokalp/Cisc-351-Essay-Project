from sklearn.model_selection import StratifiedKFold
from simpletransformers.classification import MultiLabelClassificationModel
import pandas as pd
import tensorflow as tf
import numpy as np
import sklearn


def to_letter(num):
    """Integer to string"""
    if num == 1:
        return 'one'
    elif num == 2:
        return 'two'
    elif num == 3:
        return 'three'
    elif num == 4:
        return 'four'
    elif num == 5:
        return 'five'
    elif num == 6:
        return 'six'
    elif num == 7:
        return 'seven'
    elif num == 8:
        return 'eight'
    elif num == 9:
        return 'nine'


def preprocess(df, nums=True):
    """Makes the data in the format the module accepts"""
    if nums:
        for i in range(len(df)):
            essay_set = to_letter(df.loc[i, "essay_set"])
            df.loc[i, "essay"] = f"This essay belongs to set {essay_set}. " + \
                                 df.loc[i, "essay"]
        df.loc[:, "text"] = df.essay.tolist()
    df["label"] = tf.keras.utils.to_categorical(df.score.tolist()).tolist()
    del df["essay"]
    del df["score"]
    del df["essay_set"]
    return df


def accuracy(y_true, y_pred):
    """Accuracy metric for roberta"""
    if len(y_true.shape) < 2:
        y_true = np.argmax(y_true)
    else:
        y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)

    return sklearn.metrics.accuracy_score(y_true, y_pred)


def loss_log(y_true, y_pred):
    """Returns log loss for Roberta"""
    y_true = y_true.tolist()

    return sklearn.metrics.log_loss(y_true, y_pred)


def cross_validate(model, X, y):
    """Applies 5-fold crossvalidation on Roberta and returns the average
    accuracy
    and the average loss"""
    skf = StratifiedKFold(5, random_state=42)
    acc_val = []
    loss_val = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        train = pd.concat([X_train, y_train])
        test = pd.concat([X_test, y_test])
        roberta = model('roberta', 'roberta-base',
                        num_labels=6,
                        args=args)
        roberta.train_model(train)
        result = roberta.eval_model(test, acc=accuracy, loss=loss_log)
        acc_val.append(result[1])
        loss_val.append(result[2])
    acc = np.array(acc_val).mean()
    loss = np.array(loss_val).mean()

    return acc, loss


if __name__ == '__main__':
    np.random.seed(seed=42)
    df = pd.read_csv("data/scaled.csv")
    df = preprocess(df)
    args = {'reprocess_input_data': True, 'overwrite_output_dir': True,
            'num_train_epochs': 1,
            'max_seq_length': 400, 'evaluate_during_training': False}
    acc, loss = cross_validate(MultiLabelClassificationModel, df.text,
                               df.label)
    print("Accuracy score: ", acc)
    print("Logloss score: ", loss)
