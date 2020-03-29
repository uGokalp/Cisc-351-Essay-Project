from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def lemmatize_text(text):
    """Lemmatizes the text and returns the new text"""
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    return " ".join([lemmatizer.lemmatize(w) for w in words])


def stem_text(text):
    """Stems the text and returns the new text"""
    stemmer = PorterStemmer()
    words = word_tokenize(text)
    return " ".join([stemmer.stem(w) for w in words])


def transform(df):
    """Applies preprocessing transformations on the dataframe returns a copy"""
    df_new = df.copy()
    df_new.loc[:, "essay"] = df_new.essay.str.replace("\d{1,4}", '')
    df_new.loc[:, "essay"] = df_new.essay.apply(stem_text)
    df_new.loc[:, "essay"] = df_new.essay.apply(lemmatize_text)
    return df_new


def document_term(Count):
    """Turns the count vectorizers into Bag of Words
        returns X and Y"""
    bow = pd.DataFrame(Count.toarray(),
                       columns=count_vectorizer.get_feature_names())
    bow.dropna(inplace=True)
    bow["score"] = df.score.tolist()
    bow["essay_set"] = df.essay_set.tolist()
    X = bow.drop("score", axis=1)
    Y = df.score
    return X, Y


def to_tfidf(Count):
    """Turns the count vectorizer into tfidf vectorizer
    returns X and Y"""
    tfidf_transformer = TfidfTransformer()
    tfidf = tfidf_transformer.fit_transform(Count)
    tfidf = pd.DataFrame(tfidf.toarray(),
                         columns=count_vectorizer.get_feature_names())
    tfidf["score"] = df.score.tolist()
    tfidf["essay_set"] = df.essay_set.tolist()

    tfidf.dropna(inplace=True)
    X = tfidf.drop("score", axis=1)
    Y = df.score
    return X, Y


def cross_validate(model, X, y):
    """Applied 5-fold cross validation and returns the average accuracy
    and the average loss"""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc = []
    loss = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        rf = model(random_state=42)
        rf.fit(X_train, y_train)
        pred = rf.predict(X_test)
        pred_p = rf.predict_proba(X_test)
        acc.append(accuracy_score(y_test, pred))
        loss.append(log_loss(y_test, pred_p))

    acc = np.array(acc).mean()
    loss = np.array(loss).mean()
    return acc, loss


if __name__ == '__main__':
    np.random.seed(seed=42)
    df = pd.read_csv("../data/scaled.csv")
    stop_words = list(set(stopwords.words("english")))

    df_new = transform(df)
    count_vectorizer = CountVectorizer(min_df=10, stop_words=stop_words)
    Count = count_vectorizer.fit_transform(df_new.essay)
    X, y = document_term(Count)
    acc, loss = cross_validate(RandomForestClassifier, X, y)

    print("Accuracy score: ", acc)
    print("Logloss score: ", loss)

    X, y = to_tfidf(Count)
    acc, loss = cross_validate(RandomForestClassifier, X, y)

    print("Accuracy score: ", acc)
    print("Logloss score: ", loss)
