import torch

from flask import Flask, jsonify, request
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('DataAngelo/propoint_Final_project')
tokenizer.save_pretrained("./tokenizers")

model = BertForSequenceClassification.from_pretrained('DataAngelo/propoint_Final_project')
model.save_pretrained("./propoint_model")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def propoint_predict():
    query = request.json['query']
    input = tokenizer.encode_plus(query, add_special_tokens=True, return_tensors='pt')

    output = model(**input)
    logits = output[0]
    predicted_label = torch.argmax(logits, dim=1).tolist()[0]

    return jsonify({'label':predicted_label})

# run flask app
if __name__ == '__main__':
    app.run(debug=True)