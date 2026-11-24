# Install required libraries:
# pip install -r requirements.txt

import nltk
import stanza
import spacy

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, ne_chunk, CFG
from nltk.parse import ChartParser
from nltk.util import ngrams
from nltk.sentiment import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# 1. TOKENIZATION USING NLTK, STANZA AND SPACY
# ============================================================

# Download required NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')
nltk.download('vader_lexicon')


# Sample paragraph
text = """Dr. A.P.J. Abdul Kalam was the 11th President of India and a renowned aerospace scientist.
He was born on October 15, 1931, in Rameswaram, Tamil Nadu.
Kalam played a key role in India's Pokhran-II nuclear tests in 1998.
He worked with ISRO and DRDO on missile development programs such as Agni and Prithvi.
In 2002, he became the President of India and served until 2007.
He received the Bharat Ratna, India's highest civilian award, in 1997.
Dr. Kalam inspired millions of students and delivered lectures at universities across the world."""


# -------------------- NLTK TOKENIZATION --------------------

print("\n========== NLTK TOKENIZATION ==========")

sentences = sent_tokenize(text)

print("\nSentence Tokenization:")
for i, sentence in enumerate(sentences, start=1):
    print(f"Sentence {i}: {sentence}")

print("\nWord Tokenization:")
for i, sentence in enumerate(sentences, start=1):
    words = word_tokenize(sentence)
    print(f"Sentence {i}: {words}")


# -------------------- STANZA TOKENIZATION --------------------

print("\n========== STANZA TOKENIZATION ==========")

stanza.download('en')
stanza_nlp = stanza.Pipeline(lang='en', processors='tokenize')

stanza_text = """Natural Language Processing is a subfield of Artificial Intelligence.
It deals with interaction between computers and human language."""

doc = stanza_nlp(stanza_text)

print("\nSentence Tokenization:")
for i, sentence in enumerate(doc.sentences, start=1):
    print(f"Sentence {i}: {sentence.text}")

print("\nWord Tokenization:")
for i, sentence in enumerate(doc.sentences, start=1):
    words = [word.text for word in sentence.words]
    print(f"Sentence {i}: {words}")


# -------------------- SPACY TOKENIZATION --------------------

print("\n========== SPACY TOKENIZATION ==========")

try:
    spacy_nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError(
        "Run: python -m spacy download en_core_web_sm"
    )

spacy_doc = spacy_nlp(stanza_text)

print("\nSentence Segmentation:")
for i, sentence in enumerate(spacy_doc.sents, start=1):
    print(f"Sentence {i}: {sentence.text}")

print("\nWord Tokenization:")
for i, sentence in enumerate(spacy_doc.sents, start=1):
    words = [token.text for token in sentence]
    print(f"Sentence {i}: {words}")


# ============================================================
# 2. STOPWORD REMOVAL
# ============================================================

print("\n========== STOPWORD REMOVAL ==========")

stop_words = set(stopwords.words('english'))

filtered_sentences_words = []

for sentence in sentences:
    words = word_tokenize(sentence)

    filtered_words = [
        word for word in words
        if word.lower() not in stop_words
    ]

    filtered_sentences_words.append(filtered_words)

print("\nSentences after Stopword Removal:")

for i, words in enumerate(filtered_sentences_words, start=1):
    print(f"Sentence {i}: {words}")


# ============================================================
# 3. STEMMING
# ============================================================

print("\n========== STEMMING ==========")

porter_stemmer = PorterStemmer()

stemmed_sentences_words = []

for words in filtered_sentences_words:
    stemmed_words = [
        porter_stemmer.stem(word)
        for word in words
    ]

    stemmed_sentences_words.append(stemmed_words)

print("\nSentences after Stemming:")

for i, words in enumerate(stemmed_sentences_words, start=1):
    print(f"Sentence {i}: {words}")


# ============================================================
# 4. LEMMATIZATION
# ============================================================

print("\n========== LEMMATIZATION ==========")

wordnet_lemmatizer = WordNetLemmatizer()

lemmatized_sentences_words = []

for words in filtered_sentences_words:
    lemmatized_words = [
        wordnet_lemmatizer.lemmatize(word)
        for word in words
    ]

    lemmatized_sentences_words.append(lemmatized_words)

print("\nSentences after Lemmatization:")

for i, words in enumerate(lemmatized_sentences_words, start=1):
    print(f"Sentence {i}: {words}")


# ============================================================
# 5. COMPARISON OF STEMMING AND LEMMATIZATION
# ============================================================

print("\n========== STEMMING VS LEMMATIZATION ==========")

for i in range(len(filtered_sentences_words)):

    print(f"\nSentence {i + 1}:")

    print("Original   :", filtered_sentences_words[i])
    print("Stemmed    :", stemmed_sentences_words[i])
    print("Lemmatized :", lemmatized_sentences_words[i])


# ============================================================
# 6. POS TAGGING USING NLTK
# ============================================================

print("\n========== POS TAGGING ==========")

tagged_sentences_words = []

for words in filtered_sentences_words:
    tagged_words = pos_tag(words)
    tagged_sentences_words.append(tagged_words)

print("\nSentences after POS Tagging:")

for i, tagged_words in enumerate(tagged_sentences_words, start=1):
    print(f"Sentence {i}: {tagged_words}")


# ============================================================
# 7. CHUNKING AND NAMED ENTITY RECOGNITION
# ============================================================

print("\n========== NAMED ENTITY RECOGNITION ==========")

named_entities_sentences = []

for tagged_words in tagged_sentences_words:
    named_entities = ne_chunk(tagged_words)
    named_entities_sentences.append(named_entities)

print("\nSentences with Named Entities:")

for i, named_entities in enumerate(named_entities_sentences, start=1):
    print(f"Sentence {i}:")
    print(named_entities)


# ============================================================
# 8. TF-IDF CALCULATION
# ============================================================

print("\n========== TF-IDF CALCULATION ==========")

documents = sentences

processed_docs = [
    " ".join([
        word.lower()
        for word in word_tokenize(doc)
        if word.isalnum()
    ])
    for doc in documents
]

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(processed_docs)

print("\nTF-IDF Matrix:")
print(tfidf_matrix.toarray())

print("\nFeature Names:")
print(vectorizer.get_feature_names_out())


# ============================================================
# 9. CYK / CHART PARSING
# ============================================================

print("\n========== CYK / CHART PARSING ==========")

grammar = CFG.fromstring("""
S -> NP VP
VP -> V NP
NP -> Det N
Det -> 'the' | 'a'
N -> 'cat' | 'dog'
V -> 'chased' | 'saw'
""")

parser = ChartParser(grammar)

sentence = "the cat chased the dog".split()

print("\nParsing Trees:")

for tree in parser.parse(sentence):
    print(tree)


# ============================================================
# 10. N-GRAMS (UNIGRAM, BIGRAM AND TRIGRAM)
# ============================================================

print("\n========== N-GRAM GENERATION ==========")

sample_text = "this is a simple nlp example for ngram generation"

tokens = word_tokenize(sample_text)

print("\nUnigrams:")
print(list(ngrams(tokens, 1)))

print("\nBigrams:")
print(list(ngrams(tokens, 2)))

print("\nTrigrams:")
print(list(ngrams(tokens, 3)))


# ============================================================
# 11. SENTIMENT ANALYSIS USING NLTK
# ============================================================

print("\n========== SENTIMENT ANALYSIS ==========")

sia = SentimentIntensityAnalyzer()

text_samples = [
    "I love this product! It is amazing.",
    "This is the worst experience I have ever had.",
    "The movie was okay, not great but not bad."
]

for sample in text_samples:

    score = sia.polarity_scores(sample)

    print("\nText:", sample)
    print("Scores:", score)

    if score['compound'] >= 0.05:
        print("Sentiment: Positive")

    elif score['compound'] <= -0.05:
        print("Sentiment: Negative")

    else:
        print("Sentiment: Neutral")