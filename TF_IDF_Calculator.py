from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from documentTokenizer import DF,textOperation;
import pandas as pd

# textOperation('files/doc1.txt')
# textOperation('files/doc2.txt')
# textOperation('files/doc3.txt')
# textOperation('files/doc4.txt')

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

def Calculate_IDF(doc):
    # Create an instance of TfidfVectorizer
    vectorizer = TfidfVectorizer()

    # Fit the vectorizer on the document collection to calculate IDF
    vectorizer.fit(doc)
    
    idf = {}
    idf_values = vectorizer.idf_
    terms = vectorizer.get_feature_names_out()
    
    for i, term in enumerate(terms):
        idf[term] = idf_values[i]
        
    return idf

def Calculate_TF_IDF(doc):
    # Join the tokenized words back into a document
    document = ' '.join(doc)

    # Create an instance of TfidfVectorizer
    vectorizer = TfidfVectorizer()

    # Fit the vectorizer on the document and transform it into TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform([document])

    # Get the feature names (terms)
    feature_names = vectorizer.get_feature_names_out()

    # Get the TF-IDF scores
    tfidf_scores = tfidf_matrix.toarray()[0]

    # Create a dictionary to store the TF-IDF scores for each term
    tfidf = {}

    # Populate the dictionary with term-score pairs
    for i in range(len(feature_names)):
        term = feature_names[i]
        score = tfidf_scores[i]
        tfidf[term] = score

    return tfidf

def Matrix_TF_IDF(doc):
    # Join the tokenized words back into a document
    # document = ' '.join(doc)

    # Create an instance of TfidfVectorizer
    vectorizer = TfidfVectorizer()

    # Fit the vectorizer on the document and transform it into TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(doc)

    # Get the feature names (terms)
    feature_names = vectorizer.get_feature_names_out()
    tf_idf_array = tfidf_matrix.toarray()
    words_set = vectorizer.get_feature_names_out()
    df_tf_idf = pd.DataFrame(tf_idf_array, columns = words_set)
    return df_tf_idf


def displayTF_IDF(tfidf):
    for term, score in tfidf.items():
        print(f"Term: {term}, TF-IDF Score: {score:.4f}")
    
        
def displayIDF(idf):
    for term, idf_value in idf.items():
        print(f"Term: {term}, IDF: {idf_value:.4f}")
        

def displayTF(tf):
    for term, tf_value in tf.items():
        print(f"Term: {term}, TF: {tf_value:.4f}")
        

    
abc = [DF[keys] for keys in DF.keys()]
joined = []
for i in abc:
    joined.append(' '.join(i))
    
result = Matrix_TF_IDF(joined)

def displayResult():
    for keys,val in DF.items():
        print("Document " + keys)
        idf = Calculate_TF_IDF(val)
        displayTF_IDF(idf)
        print('---------------------------------------')
    print(result)
    
    
# displayResult()
# input()