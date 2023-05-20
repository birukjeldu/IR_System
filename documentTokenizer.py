from collections import Counter
import re
import os
import math
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download NLTK
# nltk.download('punkt')
# nltk.download('stopwords')

DF = dict()

# Read the document from file
# filePath = 'files/doc1.txt'


# This function Performs All the Text Operations
def textOperation(filePath):
    # Read the document from file
    with open(filePath, 'r') as file:
        document = file.read()
        fileName = os.path.splitext(os.path.basename(filePath))[0]

    # Using regular expression to remove any punctuations
    document = re.sub(r'[^\w\s]', '', document)

    # Tokenization
    tokens = word_tokenize(document)

    # Elimination of stop words
    stop_words = set(stopwords.words('english'))

    filtered_tokens = []
    for word in tokens:
        if word.lower() not in stop_words:
            filtered_tokens.append(word)

    # Steeming the tokens
    stemmer = PorterStemmer()
    stemmed_tokens = []
    for token in filtered_tokens:
        stemmed_tokens.append(stemmer.stem(token))

    # for token in stemmed_tokens:
    #     print(token)
    DF[fileName] = stemmed_tokens
    # print(DF)


# def calculate_tfidf(docs):
#     vectorizer = TfidfVectorizer()
#     tfidf_matrix = vectorizer.fit_transform(docs)
#     feature_names = vectorizer.get_feature_names_out()

#     tfidf_scores = []
#     for i in range(len(docs)):
#         doc = docs[i]
#         scores = {}
#         for j, term in enumerate(feature_names):
#             score = tfidf_matrix[i, j]
#             if score > 0:
#                 scores[term] = score
#         tfidf_scores.append(scores)

#     return tfidf_scores


textOperation('files/doc1.txt')
textOperation('files/doc2.txt')
textOperation('files/doc3.txt')


# # Assuming you have a list of documents stored in the DF dictionary
# docs = list(DF.values())
# tfidf_scores = calculate_tfidf(docs[0])

# # Print the TF-IDF scores for each document
# for i, scores in enumerate(tfidf_scores):
#     doc_name = list(DF.keys())[i]
#     print("TF-IDF scores for {doc_name}:")
#     for term, score in scores.items():
#         print("Term: {term}, TF-IDF Score: {score:.4f}")
#     print()

# print(list(DF.values()))


def Calculate_TF(doc):
    # Count the frequency of each term
    term_counts = Counter(doc)

    # Calculate the total number of terms
    total_terms = len(doc)

    # Normalize the term frequencies
    tf = {}
    
    # Calculate the TF for each term
    for term, count in term_counts.items():
        tf[term] = count / total_terms

    return tf

def displayTF(tf):
    for term, tf_value in tf.items():
        print(f"Term: {term}, TF: {tf_value:.4f}")


for keys,val in DF.items():
    print("Document " + keys)
    tf = Calculate_TF(val)
    displayTF(tf)
    print('---------------------------------------')
    
input()

