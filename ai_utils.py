from huggingface_hub import InferenceClient

def sentiment_analysis(text):
    client = InferenceClient(api_key="")

    try:
        result = client.text_classification(
            text,
            model="tabularisai/multilingual-sentiment-analysis"
        )

        if isinstance(result, list) and len(result) > 0:
            # Sort by score descending just in case it's not sorted
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
