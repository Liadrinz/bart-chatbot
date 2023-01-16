from ahocorasick import Automaton

from flask import Flask, render_template, redirect, request
from bot import Bot


def web_app():
    app = Flask(__name__)
    bot = Bot()
    automaton = Automaton()
    with open("sensitive.txt") as fin:
        for idx, line in enumerate(fin):
            word = line.strip()
            automaton.add_word(word, (idx, word))
    automaton.make_automaton()

    @app.route("/")
    def index():
        return render_template("index.html")
    
    @app.route("/favicon.ico")
    def favicon():
        return redirect("/static/favicon.ico")

    @app.route("/chat", methods=["POST"])
    def chat():
        msgs = request.json["msgs"]
        context = "[SEP]".join([msg["msg"] for msg in msgs])
        resps = bot.chat(
            context,
            decode_method=request.json["decodeMethod"],
            num_beams=request.json["numBeams"],
            top_p=request.json["topp"],
            top_k=request.json["topk"]
        )
        return {"msg": resps[0], "candidates": resps[1:]}


    @app.route("/correct", methods=["POST"])
    def correct():
        msgs = request.json["msgs"]
        rn = request.json["rn"]
        rp = request.json["rp"]
        context = "[SEP]".join([msg["msg"] for msg in msgs])
        return {"loss": bot.correct(context, rn, rp)}
    
    @app.route("/feedback", methods=["POST"])
    def feedback():
        msgs = request.json["msgs"]
        r = request.json["r"]
        p = request.json["p"]
        context = "[SEP]".join([msg["msg"] for msg in msgs])
        return {"loss": bot.feedback(context, r, p)}
    
    return app

app = web_app()

if __name__ == "__main__":
    app.run("0.0.0.0", 8000, debug=True)
