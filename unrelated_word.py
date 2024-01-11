from flask import Flask, render_template, request
from gensim.models import Word2Vec
import sys

app = Flask(__name__)

# Load the trained Word2Vec model
model = Word2Vec.load("word2vec.model")

@app.route("/", methods=["GET", "POST"])
def index():
    unrelated_word = []

    if request.method == "POST":
        word = request.form["word"]
        if word:
            try:
                # Get the least related word
                all_sims = model.wv.most_similar(word, topn=sys.maxsize)
                unrelated_word = list(reversed(all_sims[-1:]))
            except KeyError:
                unrelated_word = ["Word not found in the model."]

    return render_template("index.html", unrelated_word=unrelated_word)

if __name__ == "__main__":
    app.run(debug=True)
