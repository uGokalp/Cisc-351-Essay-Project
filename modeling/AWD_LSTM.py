from fastai.text import *
from sklearn.model_selection import StratifiedKFold


def cross_val_dl(data, data_lm):
    """Applies Cross validation to the language model and returns the
    average accuracy
    and the average loss"""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc_val = []
    loss_val = []
    for train_index, val_index in skf.split(data.index, data.score):
        df_train = data.iloc[train_index]
        df_valid = data.iloc[val_index]
        fold = TextClasDataBunch.from_df(path='.', train_df=df_train,
                                         valid_df=df_valid,
                                         vocab=data_lm.train_ds.vocab,
                                         label_cols='score',
                                         text_cols=['essay', 'essay_set'],
                                         bs=64)
        learn = text_classifier_learner(fold, AWD_LSTM, drop_mult=0.3)
        learn.fit_one_cycle(5)
        loss, acc = learn.validate()
        acc_val.append(acc.numpy())
        loss_val.append(loss)
        acc = np.array(acc_val).mean()
        loss = np.array(loss_val).mean()

    return acc, loss


if __name__ == '__main__':
    np.random.seed(seed=42)
    df = pd.read_csv("../data/scaled.csv")
    df_text = df.iloc[np.random.permutation(len(df_text))]
    cut1 = int(0.8 * len(df_text)) + 1
    df_train, df_valid = df_text[:cut1], df_text[cut1:]
    data_lm = TextLMDataBunch.from_df(path='.',
                                      train_df=df_train,
                                      valid_df=df_valid,
                                      label_cols='score',
                                      text_cols='essay')
    lm = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)
    data = df.sample(len(df), random_state=42)
    acc, loss = cross_val_dl(data, data_lm)
    print("Accuracy score: ", acc)
    print("Logloss score: ", loss)
