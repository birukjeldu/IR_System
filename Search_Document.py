import pandas as pd
import TF_IDF_Calculator as tf
from nltk.stem import PorterStemmer


def wordOperation(text):
    stemmer = PorterStemmer()
    token = stemmer.stem(text)
    return token

userInputResult = dict()

def wordChecker(word,dictionay):
    for row in dictionay.items():
        if(word.lower()==row[0]):
            return row[1]
    return -1
    

# Is used to covert the panda dataframe into dictionary
result = tf.result
result = result.reset_index()
weightedTokens = []
for index,row in result.iterrows():
    weightedTokens.append(row.to_dict())
    
term = wordOperation("a")
def resultToDict(term):
    for row in weightedTokens:
        resultCounter = wordChecker(term,row)  
        userInputResult[weightedTokens.index(row)] = resultCounter
    

def displayResult(term):
    docName = ['doc1','doc2','doc3','doc4']
    print(f"Relevant Documents for term '{term}'")
    flag = 0
    count = 0
    for key, val in sortedUserResult.items():
        if(val > 0):
            flag = 1
            count += 1
            print(f"\t{count}. {docName[key]}.txt --- TF-IDF = {val:.4f}")
    
    if flag == 0:
        print("!!! No result was found in the documents !!!!")

while True:
    print("Enter a word to search(q - to exit ): ",end='')
    term = input()
    if term.lower() == 'q':
        break
    term = wordOperation(term)
    resultToDict(term)
    sortedUserResult = dict(sorted(userInputResult.items(),key=lambda x:x[1]))
    displayResult(term)
    print("\t----------------------------")
