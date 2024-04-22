from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import joblib
import requests

app = Flask(__name__)
CORS(app)

vectorizer = joblib.load('vectorizer.pkl')
model = joblib.load('trained_model.pkl')

url = "https://health.gov/myhealthfinder/api/v3/topicsearch.json"

topics_mapping = {
    "Get Your Teen Screened for Depression": 539,
    "Talk with Your Doctor About Depression": 540,
    "Get Your Child Screened for Anxiety": 34321,
    "Anxiety: Conversation Starters": 34691,
    "Talk with Your Doctor About Anxiety": 34692,
}

categories_mapping = {
    "Mental Health and Relationships": 20,
    "Mental Health": 109
}

def search_topics(input_words):
    topic_id = None
    for name, tid in topics_mapping.items():
        for word in input_words:
            if word in name.lower():
                topic_id = tid
                break
        if topic_id is not None:
            break
    
    if topic_id is None:
        return {'error': 'No matching topics found'}

    topic_url = f"{url}?TopicId={topic_id}"
    response = requests.get(topic_url)
    data = response.json()

    return data

def search_categories(input_words):
    category_ids = []
    for category, cid in categories_mapping.items():
        for word in input_words:
            if word in category.lower():
                category_ids.append(cid)
                break
    
    if not category_ids:
        return {'error': 'No matching categories found'}

    category_urls = [f"{url}?CategoryId={cid}" for cid in category_ids]
    category_results = {}
    for category_url in category_urls:
        response = requests.get(category_url)
        data = response.json()
        category_results.update(data)

    return category_results


@app.route('/mentaldisorder', methods=['POST'])
def detect_mental_disorder():
    user_input = request.json.get('input_text', '')

    input_vector = vectorizer.transform([user_input])

    predicted_disorder = model.predict(input_vector)

    return jsonify({'predicted_disorder': predicted_disorder.tolist()}), 200

@app.route('/topic_information', methods=['POST'])
def topic_information():
    user_input = request.json.get('input_text', '')

    topic_results = search_topics(user_input)

    return jsonify(topic_results), 200

@app.route('/category_information', methods=['POST'])
def category_information():
    user_input = request.json.get('input_text', '')

    category_results = search_categories(user_input)

    return jsonify(category_results), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)