from gensim import corpora, models, similarities

# List of documents in your corpus
corpus = [
    "Document 1: This is the first document",
    "Document 2: This document is the second document",
    "Document 3: And this is the third one",
]

# Tokenize the documents
tokenized_corpus = [doc.lower().split() for doc in corpus]

# Create a dictionary from the tokenized corpus
dictionary = corpora.Dictionary(tokenized_corpus)

# Create a TF-IDF model from the tokenized corpus
tfidf = models.TfidfModel(dictionary=dictionary)
corpus_tfidf = [tfidf[dictionary.doc2bow(tokens)]
                for tokens in tokenized_corpus]

# Create an index for the corpus
index = similarities.MatrixSimilarity(corpus_tfidf)

# Perform a query
query = "third"
query_bow = dictionary.doc2bow(query.lower().split())
query_tfidf = tfidf[query_bow]

# Get similarity scores between the query and corpus documents
sims = index[query_tfidf]

# Sort the similarity scores
sorted_sims = sorted(enumerate(sims), key=lambda item: -item[1])

# Retrieve the most similar document from the corpus
most_similar_doc_index = sorted_sims[0][0]
most_similar_doc = corpus[most_similar_doc_index]

print(f"Query: {query}")
print(f"Most similar document: {most_similar_doc}")

