import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

df = pd.read_csv("SMSSpamCollection", sep="\t", names=["label", "message"])
df["label"] = df["label"].map({"ham": 0, "spam": 1})
x=df["message"]
y=df["label"]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42, stratify=y)

vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
x_train_tfidf = vectorizer.fit_transform(x_train)
x_test_tfidf = vectorizer.transform(x_test)

print("\nTF-IDF Training Shape:",x_train_tfidf.shape)
print("TF-IDF Testing Shape:",x_test_tfidf.shape)

model=MultinomialNB()
model.fit(x_train_tfidf, y_train)

y_pred = model.predict(x_test_tfidf)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print("Accuracy :", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall   :", round(recall * 100, 2), "%")
print("F1 Score :", round(f1 * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Ham", "Spam"]
))

print("Total Messages:",len(df))
print("Training Messages:",len(x_train))
print("Testing Messages:",len(x_test))
print("Model Training Completed Successfully")

def predict_message(message):
    message_tfidf = vectorizer.transform([message])
    prediction = model.predict(message_tfidf)[0]

    if prediction == 1:
        return "SPAM"
    else:
        return "HAM"


test_messages = [
    "Congratulations! You have won a free iPhone. Claim your prize now!",
    "Hey, are we meeting at college tomorrow?",
    "URGENT! You have won 50000 rupees. Call now to claim your reward."
]

print("\n--- Custom Message Testing ---")

for message in test_messages:
    print("\nMessage:", message)
    print("Prediction:", predict_message(message))
print("\n--- Interactive Spam Detector ---")

while True:
    message = input("\nEnter a message (or type 'exit' to stop): ")

    if message.lower() == "exit":
        print("Exiting Spam Detector...")
        break

    prediction = predict_message(message)

    print("Prediction:", prediction)