from flask import Flask, request, jsonify
from bs4 import BeautifulSoup
import requests
import pickle
import joblib
import re
vectorizer = joblib.load('E:\Programming Journey\ML Journey\OCR receipt checker\OCR receipt checker\model\lectorizer.pkl')
label_encoder = joblib.load('E:\Programming Journey\ML Journey\OCR receipt checker\OCR receipt checker\model\label_encoder.pkl')
model = joblib.load('E:\Programming Journey\ML Journey\OCR receipt checker\OCR receipt checker\model\model.pkl')

app = Flask(__name__)

@app.route('/parse', methods=['POST'])
def parse_info():
    data = request.get_json()
    url = data['url']

    response = requests.get(url)
    html_content = response.content
    soup = BeautifulSoup(html_content, 'html.parser')

    list_of_items = soup.find_all('li', style="border-bottom: 1px dotted #777; padding: 10px 0;")
    items = []
    for item in list_of_items:
        product_name = ' '.join(str(i) for i in item.find(class_="wb-all").get_text().split()[1:-1])
        price_text = item.find(class_="ready_ticket__item").get_text()
        price = float(re.search(r"=\s*(\d+\.\d{2})", price_text).group(1))

        items.append({
            'product': product_name,
            'price': price
        })
    print(items)
    return jsonify(items)

@app.route('/categorize', methods=['POST'])
def categorize():
    data = request.get_json()
    categorized_items = []
    for item in data:
        product = item['product']
        item_vec = vectorizer.transform([product])
        category_enc = model.predict(item_vec)
        category = label_encoder.inverse_transform(category_enc)
        categorized_items.append({
            'product': product,
            'price': item['price'],
            'category': category[0]
        })
    return jsonify(categorized_items)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')