import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Read the document from file
with open('files/doc1.txt', 'r') as file:
    document = file.read()

# Tokenization
tokens = word_tokenize(document)

# Elimination of stop words
stop_words = set(stopwords.words('english'))
# filtered_tokens = [word for word in tokens if word.casefold() not in stop_words]
filtered_tokens = []
for word in tokens:
    if word.lower() not in stop_words:
        filtered_tokens.append(word)

# Print the tokens
for token in filtered_tokens:
    print(token)
