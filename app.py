import tkinter as tk
from tkinter import scrolledtext
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
data = pd.read_csv("faq_dataset.csv")

questions = data['question']
answers = data['answer']

# Text preprocessing
def preprocess(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    return text

questions_clean = questions.apply(preprocess)

# TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions_clean)

# Chatbot response
def chatbot_response(user_input):
    user_input = preprocess(user_input)
    user_vector = vectorizer.transform([user_input])

    similarity = cosine_similarity(user_vector, X)
    index = similarity.argmax()

    if similarity[0][index] < 0.3:
        return "Sorry, I don't understand your question."

    return answers[index]

# Send message
def send_message(event=None):
    user_input = entry_box.get()

    if user_input.strip() == "":
        return

    chat_area.insert(tk.END, "You: " + user_input + "\n", "user")
    entry_box.delete(0, tk.END)

    response = chatbot_response(user_input)

    chat_area.insert(tk.END, "Bot: " + response + "\n\n", "bot")
    chat_area.yview(tk.END)

# Window
window = tk.Tk()
window.title("AI Chatbot")
window.geometry("500x550")
window.config(bg="#f0f0f0")

# Chat area
chat_area = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=60, height=25)
chat_area.pack(pady=10)

chat_area.tag_config("user", foreground="blue", font=("Arial", 10, "bold"))
chat_area.tag_config("bot", foreground="green", font=("Arial", 10))

# Input frame
input_frame = tk.Frame(window)
input_frame.pack(pady=10)

entry_box = tk.Entry(input_frame, width=35, font=("Arial", 12))
entry_box.pack(side=tk.LEFT, padx=5)

send_button = tk.Button(input_frame, text="Send", command=send_message)
send_button.pack(side=tk.LEFT)

# Enter key send
window.bind("<Return>", send_message)

window.mainloop()