import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

df = pd.read_csv("../data/scaled.csv")
features = pd.read_csv("../data/features2.csv")
sns.set_palette("muted")

bins = pd.qcut(features.NounCount, 3)
sns.barplot(x=bins, y=features.NounCount, data=features,
            estimator=lambda x: len(x) / len(df) * 100, hue="score")
plt.title("Count of Nouns and The Associated Grade")
plt.ylabel("Percent")
plt.show()

bins = pd.qcut(features.SentenceLength, 3, duplicates='drop')
sns.barplot(x=bins, y=features.SentenceLength, data=features,
            estimator=lambda x: len(x) / len(df) * 100, hue="score")
plt.title("Average Sentence Length and The Associated Grade")
plt.ylabel("Percent")
plt.show()

bins = pd.qcut(features.LongWord, 3, duplicates='drop')
sns.countplot(x=bins, data=features, hue="score")
plt.title("Count of Words longer than 6 characters and The Associated Grade")
plt.show()
