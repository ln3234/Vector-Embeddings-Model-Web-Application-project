import wikipediaapi
import gensim
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stopwords.words('english')]
    return words

def train_save_model(text):
    processed_text = preprocess_text(text)
    sentences = nltk.sent_tokenize(text)
    processed_sentences = [preprocess_text(sentence) for sentence in sentences]
    model = Word2Vec(sentences=processed_sentences, vector_size=30, window=5, min_count=2)
    model.save("word2vec.model")

def get_wikipedia_content(page_title):
    user = 'LenaCodingAssingment/1.0 (ln3234pgt@students.nulondon.ac.uk)'
    wiki_wiki = wikipediaapi.Wikipedia(user, 'en')
    page = wiki_wiki.page(page_title)
    return page.text if page.exists() else "Page not found."

# Load wikipedia pages, make them one and run functions
page_titles = ["Romance", "Anger", "General Relativity", "Pop Music", "Romanticism", "Jerusalem", "Monarchy", "French revolution", "The Buddha", "Martial arts", "Gregorian chant", "Red panda", "Dentistry", "Cowboy Bebop"]
wikipedia_text = " "

for page in page_titles:
    wikipedia_text += get_wikipedia_content(page)

train_save_model(wikipedia_text)
