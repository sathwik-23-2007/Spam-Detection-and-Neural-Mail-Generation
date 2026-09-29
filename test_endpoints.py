import requests
import json

BASE_URL = "http://127.0.0.1:5000"

def test_generate():
    url = f"{BASE_URL}/generate_mail"
    payload = {"type": "Formal", "topic": "Sick Leave"}
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if "email" in data:
                print("✅ /generate_mail endpoint working. Email snippet:", data["email"][:50] + "...")
            else:
                print("❌ /generate_mail endpoint returned unexpected data:", data)
        else:
            print(f"❌ /generate_mail endpoint failed with status {response.status_code}: {response.text}")
    except Exception as e:
        print(f"❌ /generate_mail endpoint error: {e}")

def test_classify():
    url = f"{BASE_URL}/detect_mail"
    payload = {"email_text": "Congratulations! You've won a lottery."}
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if "result" in data:
                print("✅ /detect_mail endpoint working. Result page loaded.")
                if data["result"] == "Spam":
                     print("✅ Classification correct (Spam detected).")
                else:
                     print(f"⚠️ Classification might be incorrect (expected Spam, got {data['result']}).")
            else:
                print("❌ /detect_mail endpoint returned unexpected content.")
        else:
            print(f"❌ /detect_mail endpoint failed with status {response.status_code}: {response.text}")
    except Exception as e:
        print(f"❌ /detect_mail endpoint error: {e}")

if __name__ == "__main__":
    print("Testing InboxAI Endpoints...")
    test_generate()
    test_classify()
