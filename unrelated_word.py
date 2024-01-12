from flask import Flask, render_template, request
from gensim.models import Word2Vec

app = Flask(__name__)

# Load the trained Word2Vec model
model = Word2Vec.load("word2vec.model")

@app.route("/", methods=["GET", "POST"])
def index():
    unrelated_word = ""

    if request.method == "POST":
        word = request.form["word"]
        if word:
            try:
                # Get the least related word
                unrelated_word = model.wv.most_similar (negative=[word], topn=1)
            except KeyError:
                unrelated_word = ["Try a Different One!"]

    return render_template("index.html", unrelated_word=unrelated_word)

if __name__ == "__main__":
    app.run(debug=True)
