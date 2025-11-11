import pandas as pd
from collections import Counter
import re
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ----------------------------
# Load transcripts CSV
# ----------------------------
df = pd.read_csv("video_transcripts.csv")  # columns: video_url, transcript

# ----------------------------
# Preprocess transcripts
# ----------------------------
def preprocess(text):
    # lowercase
    text = str(text).lower()
    # remove punctuation except for basic sentence separators
    text = re.sub(r'[^\w\s]', '', text)
    return text

df['clean_transcript'] = df['transcript'].apply(preprocess)

# ----------------------------
# Word frequency across all videos
# ----------------------------
all_words = " ".join(df['clean_transcript']).split()
word_counts = Counter(all_words)
print("Top 20 most common words:")
print(word_counts.most_common(20))

# ----------------------------
# N-grams (2-grams and 3-grams)
# ----------------------------
vectorizer = CountVectorizer(ngram_range=(2,3))
X = vectorizer.fit_transform(df['clean_transcript'])
counts = X.toarray().sum(axis=0)
ngrams = vectorizer.get_feature_names_out()
ngram_counts = sorted(zip(ngrams, counts), key=lambda x: x[1], reverse=True)
print("\nTop 20 most common 2-3 word phrases:")
for phrase, count in ngram_counts[:20]:
    print(f"{phrase}: {count}")

# ----------------------------
# Sentiment analysis per video
# ----------------------------
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

df['polarity'], df['subjectivity'] = zip(*df['clean_transcript'].map(get_sentiment))

# Example: show first 5 videos sentiment
print("\nSentiment for first 5 videos:")
print(df[['video_url', 'polarity', 'subjectivity']].head())

# ----------------------------
# Word cloud of all transcripts
# ----------------------------
all_text = " ".join(df['clean_transcript'])
wc = WordCloud(width=1200, height=600, background_color='white').generate(all_text)

plt.figure(figsize=(12,6))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

# ----------------------------
# Save analysis results
# ----------------------------
df.to_csv("video_transcripts_analysis.csv", index=False)
print("Analysis saved to video_transcripts_analysis.csv")
