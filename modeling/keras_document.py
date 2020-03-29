import tensorflow as tf
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np


def get_model():
    """Creates a simple fully connected model"""
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(19,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(6, activation='sigmoid'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def cross_validate(X, y):
    """Applied 5-fold cross validation and returns the average accuracy
    and the average loss"""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc = []
    loss = []
    model = get_model()
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        print(X_train.shape)
        print(y_train.shape)
        y_cat = tf.keras.utils.to_categorical(y_train)
        model.fit(x=X_train, y=y_cat, batch_size=64, epochs=5, verbose=2,
                  validation_split=0.3)
        pred = model.predict_classes(X_test)
        pred_p = model.predict_proba(X_test)
        acc.append(accuracy_score(y_test, pred))
        loss.append(log_loss(y_test, pred_p))

    acc = np.array(acc).mean()
    loss = np.array(loss).mean()
    return acc, loss


if __name__ == '__main__':
    df = pd.read_csv("data/features2.csv")
    X = df.drop("score", axis=1)
    Y = df.score
    acc, loss = cross_validate(X, Y)
    print("Accuracy score: ", acc)
    print("Logloss score: ", loss)
