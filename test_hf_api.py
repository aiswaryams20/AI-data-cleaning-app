import requests

api_url = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
headers = {"Authorization": "Bearer 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"}

payload = {"inputs": "This is so good!"}
response = requests.post(api_url, headers=headers, json=payload)

print("Status Code:", response.status_code)
print("Response Text:", response.text)
