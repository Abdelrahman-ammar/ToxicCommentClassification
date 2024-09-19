from flask import (redirect, 
                   Flask , 
                   jsonify,
                   request)
from preprocessing import (clean_sentence,
                           read_pickle,
                           read_model)

from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

app = Flask(__name__)

target_classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


model_path = "ToxicModelV2.h5"
tokenizer_path = "tokenizer.pkl"

MODEL = read_model(model_path)
tokenizer = read_pickle(tokenizer_path)

@app.route("/classify" ,  methods=["POST"])
def classify_comment():
    if request.method == "POST":
        data = request.json
        text = data.get('text')
        text = clean_sentence(text)
        text = tokenizer.texts_to_sequences(np.array([text]))
        text = pad_sequences(text,200,padding="post")
        predictions = MODEL.predict(text)
        predictions = (predictions > 0.5).astype(int)
        predictions = predictions[0]
        predected_classes = [target_classes[i] for i in range(len(predictions)) if predictions[i]==1]
        print(predected_classes)
        if predected_classes:
            return jsonify({"Classes" : predected_classes})
        else:
            return jsonify({"Classes": "healthy"})

if __name__ == '__main__':
    app.run(port=9000)