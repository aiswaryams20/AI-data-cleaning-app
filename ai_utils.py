import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()  # Load .env file

def sentiment_analysis(text):
    api_key = os.getenv("HF_API_KEY")  # Safely load from environment variable
    print("API KEY:", api_key)  # Temporarily for debugging

    client = InferenceClient(api_key=api_key)

    try:
        result = client.text_classification(
            text,
            model="tabularisai/multilingual-sentiment-analysis"
        )

        if isinstance(result, list) and len(result) > 0:
            sorted_result = sorted(result, key=lambda x: x["score"], reverse=True)
            top_result = sorted_result[0]
            return {
                "label": top_result["label"],
                "confidence": round(top_result["score"] * 100, 2)
            }
        else:
            return {"error": "Empty response from API."}

    except Exception as e:
        print("API call error:", e)
        return {"error": str(e)}
