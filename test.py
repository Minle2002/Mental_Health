import requests

url = "http://localhost:5000/topic_information"

data = {
    "input_text": "doctor and anxiety"
}

response = requests.post(url, json=data)

print(response.json())