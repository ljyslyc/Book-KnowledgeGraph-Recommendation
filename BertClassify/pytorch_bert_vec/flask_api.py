from flask import Flask, request, jsonify
from seq2vec.bert_vec import BertTextNet, BertSeqVec, AlbertTextNet
from scipy.spatial.distance import cosine

text_net = AlbertTextNet()  # 选择一个文本向量化模型
seq2vec = BertSeqVec(text_net)  # 将模型实例给向量化对象。
app = Flask(__name__)


@app.route('/api/simlarity/', methods=['GET', 'POST'])
def test_related():
    if request.method == 'GET':
        return 'bad request, please use post'
    else:
        question = request.form.get('question')
        answer = request.form.get('answer')
        question_vec = seq2vec.seq2vec(question)
        answer_vec = seq2vec.seq2vec(answer)
        sim = 1 - cosine(question_vec, answer_vec)
        return str(sim)


if __name__ == '__main__':
    app.run(
        host="0.0.0.0",
        port=7651,
    )
