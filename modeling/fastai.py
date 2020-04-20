from fastai import *
from fastai.tabular import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split


def grade_level(a):
    """Encoding Grade level information"""
    if a in [1, 5]:
        return 2
    elif a in [2, 3, 4, 6, 8]:
        return 1
    else:
        return 0


def essay_type(a):
    """Encoding essay_type"""
    if a in [1, 2, 7, 8]:
        return 0
    else:
        return 1


def create_holdout(df):
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        df.drop("score", axis=1), df.score, test_size=0.2, random_state=42)

    X_holdout["score"] = y_holdout
    X_train["score"] = y_train

    X_train.reset_index(drop=True, inplace=True)
    X_holdout.reset_index(drop=True, inplace=True)

    return X_train, X_holdout


def cross_val_dl(data, holdout):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc_val = []
    loss_val = []
    procs = [FillMissing, Categorify]
    cat_var = ["essay_set", "GradeLevel", "EssayType"]
    dep_var = "score"
    test = TabularList.from_df(holdout.drop("score", axis=1),
                               cat_names=cat_var, procs=procs)
    for train_index, val_index in skf.split(data.index, data.score):
        fold = TabularDataBunch.from_df(path='.', df=data, dep_var=dep_var,
                                        valid_idx=val_index, cat_names=cat_var,
                                        procs=procs)
        test = TabularList.from_df(holdout.drop("score", axis=1),
                                   cat_names=cat_var, procs=procs)
        fold.add_test(test, label=0)
        learn = tabular_learner(fold, layers=[1000, 500, 32], metrics=accuracy,
                                emb_drop=0.04, silent=True)
        learn.fit_one_cycle(5)
        learn.predict
        loss, acc = learn.validate()
        acc_val.append(acc.numpy())
        loss_val.append(loss)

    return acc_val, loss_val, learn


if __name__ == '__main__':
    np.random.seed(seed=42)
    df = pd.read_csv("../data/features.csv")
    encoder = LabelEncoder()
    df.loc[:, "essay_set"] = encoder.fit_transform(y=df.essay_set)
    df.loc[:, "GradeLevel"] = df.essay_set.apply(grade_level)
    df.loc[:, "EssayType"] = df.essay_set.apply(essay_type)

    data = df.sample(len(df), random_state=42)
    train, holdout = create_holdout(data)
    acc_val, loss_val, learn = cross_val_dl(train, holdout)

    print("Accuracy score: ", acc_val)
    print("Logloss score: ", loss_val)
