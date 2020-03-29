from fastai import *
from fastai.tabular import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold


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


def cross_val_dl(data):
    """Applies 5-fold crossvalidation for the fastai model and returns the
    average accuracy
    and the average loss"""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc_val = []
    loss_val = []
    procs = [FillMissing, Categorify]
    cat_var = ["essay_set", "GradeLevel", "EssayType"]
    dep_var = "score"
    for train_index, val_index in skf.split(data.index, data.score):
        fold = TabularDataBunch.from_df(path='.', df=data, dep_var=dep_var,
                                        valid_idx=val_index, cat_names=cat_var,
                                        procs=procs)
        learn = tabular_learner(fold, layers=[1000, 500, 32], metrics=accuracy,
                                emb_drop=0.04, silent=True)
        learn.fit_one_cycle(5)
        loss, acc = learn.validate()
        acc_val.append(acc.numpy())
        loss_val.append(loss)
        acc = np.array(acc_val).mean()
        loss = np.array(loss_val).mean()

    return acc, loss


if __name__ == '__main__':
    np.random.seed(seed=42)
    df = pd.read_csv("../data/features.csv")
    encoder = LabelEncoder()
    df.loc[:, "essay_set"] = encoder.fit_transform(y=df.essay_set)
    df.loc[:, "GradeLevel"] = df.essay_set.apply(grade_level)
    df.loc[:, "EssayType"] = df.essay_set.apply(essay_type)

    data = df.sample(len(df), random_state=42)
    acc, loss = cross_val_dl(data)

    print("Accuracy score: ", acc)
    print("Logloss score: ", loss)
