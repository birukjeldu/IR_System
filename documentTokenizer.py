import re
import os
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download NLTK
# nltk.download('punkt')
# nltk.download('stopwords')

DF = dict()

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



textOperation('files/doc1.txt')
textOperation('files/doc2.txt')
textOperation('files/doc3.txt')
textOperation('files/doc4.txt')
# print(DF)



def diplayFinalTextOperation():
    print("Tokens Found in each document after applying Text Operations")
    for keys,val in DF.items():
        print("Document " + keys)
        print(val)
        print('---------------------------------------')


# diplayFinalTextOperation()
