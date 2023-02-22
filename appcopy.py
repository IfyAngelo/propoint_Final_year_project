import torch
import requests
import numpy as np

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
    #input_ids = torch.tensor(input).unsqueeze(0)

    output = model(**input)
    logits = output[0]
    predicted_label = torch.argmax(logits, dim=1).tolist()[0]

    return jsonify({'label':predicted_label})

# run flask app
if __name__ == '__main__':
    app.run(debug=True)

# input_text = 'Mtn 2g lagos'
# url = "http://localhost:8080/predict"

# response = requests.get(url, params={"input_text": input_text})
# predictions = response.json()
# print(predictions)


# user_input = st.text_area('Enter Choice: ')
# button = st.button('ProPoint')

# d = {
#     0: 'Poor',
#     1: 'Bad',
#     2: 'Average',
#     3: 'Good',
#     4: 'Very Good',
#     5: 'Excellent'
# }

# if user_input and button:
#     text = tokenizer([user_input], padding=True, truncation=True, max_length=512, return_tensors='pt')
#     output = model(**text)

#     st.write('Logits: ', output.logits)
#     y_pred = np.argmax(output.logits.detach().numpy(), axis=1)
    
#     st.write('ProPoint Predicts: ', d[y_pred[0]])