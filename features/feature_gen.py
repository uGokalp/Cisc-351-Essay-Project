import nltk


def word_count(essay):
    """
    Returns the number of words
    """
    return len(word_tokenize(essay))


def long_words(essay):
    """
    Returns the number of words > 6 characters
    """
    text = essay.split(" ")
    counter = 0
    for word in text:
        if len(word) > 6:
            counter += 1
    return counter


def average_word(essay):
    """
    Returns the length of the average word
    """
    text = essay.split(" ")
    counter = []
    for word in text:
        counter.append(len(word))
    return np.mean(counter)


def quotation(essay):
    """
    Returns the number of quotations
    """
    text = word_tokenize(essay)
    counter = 0
    for word in text:
        if '"' in word or "'" in word:
            counter += 1
    if counter:
        counter = counter // 2
    return counter


def sentence_cnt(essay):
    """
    Returns the number of sentences
    """
    text = sent_tokenize(essay)
    return len(text)


def sentence_len(essay):
    """
    Returns the length of the sentence
    """
    text = sent_tokenize(essay)
    lengths = [len(sent) for sent in text]
    return np.mean(lengths)


def comma_cnt(essay):
    """
    Returns the number of commas
    """
    commas = re.findall(',', essay)
    if not commas:
        return 0
    else:
        return len(commas)


def noun_cnt(essay):
    """
    Returns the number of nouns
    """
    doc = nlp(essay)
    tags = [word.pos_ for word in doc if word.is_stop != True]
    nouns = tags.count("NOUN")
    return nouns


def verb_cnt(essay):
    """
    Returns the number of verbs
    """
    doc = nlp(essay)
    tags = [word.pos_ for word in doc if word.is_stop != True]
    verbs = tags.count("VERB")
    return verbs


def adj_cnt(essay):
    """
    Returns the number of adjectives
    """
    doc = nlp(essay)
    tags = [word.pos_ for word in doc if word.is_stop != True]
    adjectives = tags.count("ADJ")
    return adjectives


def adverb_cnt(essay):
    """
    Return the number of adverbs
    """
    doc = nlp(essay)
    tags = [word.pos_ for word in doc if word.is_stop != True]
    adverbs = tags.count("ADV")
    return adverbs


def punct_cnt(essay):
    """
    Return the number of punctiation
    """
    doc = nlp(essay)
    tags = [word.pos_ for word in doc if word.is_stop != True]
    puncts = tags.count("PUNCT")
    return puncts


def foreign_cnt(essay):
    """
    Returns the number foreign words

    """
    doc = nlp(essay)
    tags = [word.tag_ for word in doc if word.is_stop != True]
    foreign = tags.count("FW")
    return foreign


def get_sentiment(essay):
    """
    Return sentiment as (pos,neg)
    """
    sent = sentiment(essay)[0]
    if sent['label'] == 'NEGATIVE':
        neg = sent["score"]
        pos = 1 - neg
        return pos, neg
    else:
        pos = sent["score"]
        neg = 1 - pos
        return pos, neg


def spell_check(essay):
    sub = re.sub(r'[^\w\s]', '', essay)
    miss = spell.unknown(sub)
    if miss:
        return len(miss)
    else:
        return 0



if __name__ == '__main__':
    from nltk.tokenize import word_tokenize, sent_tokenize
    import re
    import numpy as np
    from transformers import pipeline
    import spacy
    from spellchecker import SpellChecker
    import pandas as pd
    from timeit import default_timer as timer

    print("Starting Timer")
    start = timer()
    print(start)

    nlp = spacy.load("en_core_web_sm")
    sentiment = pipeline("sentiment-analysis")
    spell = SpellChecker()

    path_to_essay = "../data/scaled.csv"
    data = pd.read_csv(path_to_essay)  # Contains raw essay
    df = pd.DataFrame(data.essay_set)  # the features

    df.loc[:, "WordCount"] = data.essay.apply(word_count)
    print("WordCount")
    df.loc[:, "LongWord"] = data.essay.apply(long_words)
    print("LongWord")
    df.loc[:, "AvgWord"] = data.essay.apply(average_word)
    print("AvgWord")
    df.loc[:, "Quotation"] = data.essay.apply(quotation)
    print("Quotation")
    df.loc[:, "SentenceCount"] = data.essay.apply(sentence_cnt)
    print("SentenceCount")
    df.loc[:, "SentenceLength"] = data.essay.apply(sentence_len)
    print("SentenceLength")
    df.loc[:, "CommaCount"] = data.essay.apply(comma_cnt)
    print("CommaCount")
    df.loc[:, "NounCount"] = data.essay.apply(noun_cnt)
    print("NounCount")
    df.loc[:, "AdverbCount"] = data.essay.apply(adverb_cnt)
    print("AdverbCount")
    df.loc[:, "AdjectiveCount"] = data.essay.apply(adj_cnt)
    print("AdjectiveCount")
    df.loc[:, "VerbCount"] = data.essay.apply(verb_cnt)
    print("VerbCount")
    df.loc[:, "PunctuationCount"] = data.essay.apply(punct_cnt)
    print("PunctuationCount")
    df.loc[:, "ForeignCount"] = data.essay.apply(foreign_cnt)
    print("ForeignCount")
    df.loc[:, "SpellingMistake"] = data.essay.apply(spell_check)
    print("SpellingMistake")
    df.loc[:, "Sentiment"] = data.essay.apply(get_sentiment)
    print("Sentiment")

    end = timer()
    delta = end-start
    if delta > 60:
        print(delta/60)
    else:
        print(delta)

    print("done")

    df.to_csv("../data/features.csv", index=False)
