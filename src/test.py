import joblib

# Load model
model = joblib.load("models/model.pkl")

print("Spam Classifier Ready!")
print("Type 'exit' to stop")

while True:

    text = input("\nEnter message: ")

    if text.lower() == "exit":
        break

    prediction = model.predict([text])

    print("Prediction:", prediction[0])