from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import math

stop_words = set(stopwords.words('english'))
def Text_Operation(documents):
    
    # Create a PorterStemmer object.
    ps = PorterStemmer()

    # Tokenize the documents and convert them to lowercase.
    tokenized_documents = []
    for doc in documents:
        tokenized_documents.append(word_tokenize(doc.lower()))

# Filter the documents to remove stop words.
    filtered_documents = []
    for document in tokenized_documents:
        filtered_document = []
        for word in document:
            if word.lower() not in stop_words:
                filtered_document.append(ps.stem(word.lower()))
        filtered_documents.append(filtered_document)

    return filtered_documents


def search(query, inverted_index):
    ps = PorterStemmer()
    scores = defaultdict(int)
    query_tokenize = word_tokenize(query)
    filtered_query = []
    for word in query_tokenize:
        if word.lower() not in stop_words:
            filtered_query.append(ps.stem(word.lower()))

    for term in filtered_query:
        if term in inverted_index:
            for doc_id, tf_idf in inverted_index[term]:
                scores[doc_id] += tf_idf
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def Inverted_Index_TFIDF(docs):
    N = len(docs)
    terms = {}
    for i in range(N):
        for word in docs[i]:
            if word not in terms:
                terms[word] = {'doc_freq': 0, 'postings': []}
            if not terms[word]['postings'] or terms[word]['postings'][-1][0] != i:
                terms[word]['doc_freq'] += 1
                terms[word]['postings'].append([i, 0])
            terms[word]['postings'][-1][1] += 1

    for term in terms:
        df = terms[term]['doc_freq']
        idf = math.log(N / df)
        for posting in terms[term]['postings']:
            tf = posting[1]
            tf_idf = tf * idf
            posting[1] = tf_idf

    inverted_index = {}
    for term in terms:
        postings = terms[term]['postings']
        inverted_index[term] = []
        for posting in postings:
            inverted_index[term].append((posting[0], posting[1]))

    return inverted_index


filepath = [
    "files/doc1.txt",
    "files/doc2.txt",
    "files/doc3.txt",
    "files/doc4.txt"
]
document = []

for file_path in filepath:
    with open(file_path, "r") as f:
        document.append(f.read())


filtered_document = Text_Operation(document)
inverted_index = Inverted_Index_TFIDF(filtered_document)
def displayResult(term,q):
    docName = ['doc1','doc2','doc3','doc4']
    print(f"Relevant Documents for '{q}' ")
    if(len(term) == 0):
        print("!!! No result was found in the documents !!!!")
    else:
        count = 0
        for i in term:
            count +=1
            print(f"\t{count}. {docName[i[0]]} ---  {i[1]}")

while True:
    print("Enter a word to search(q - to exit ): ",end='')
    query = input()
    if query.lower() == 'q':
        break
    result=search(query,inverted_index)
    displayResult(result,query)
    print("\t----------------------------")
