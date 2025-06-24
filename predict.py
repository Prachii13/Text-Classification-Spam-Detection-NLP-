import joblib

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

text = input("Enter your message: ")
vec = vectorizer.transform([text])
pred = model.predict(vec)[0]

print("📨 Prediction:", "Spam ❌" if pred else "Not Spam ✅")
