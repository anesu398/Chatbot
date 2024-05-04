import os
import nltk
import numpy as np
import random
import string
import pickle
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('wordnet')

# Load and preprocess the corpus
with open('corpus.txt', 'r', encoding='utf-8') as file:
    corpus = file.read()

sent_tokens = nltk.sent_tokenize(corpus)
word_tokens = nltk.word_tokenize(corpus)

lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token.lower()) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Define greeting responses
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Load or create JSON file for storing new words
if os.path.exists('new_words.json'):
    with open('new_words.json', 'r') as file:
        try:
            new_words = json.load(file)
        except json.decoder.JSONDecodeError:
            new_words = []
else:
    new_words = []

# Generate response
def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)

    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]

    if(req_tfidf == 0):
        robo_response = robo_response + "I am sorry! I don't understand you."
        return robo_response
    else:
        robo_response = robo_response + sent_tokens[idx]
        return robo_response

# Chatting
flag = True
print("ROBO: My name is Robo. I will answer your queries about Chatbots. If you want to exit, type Bye!")

while(flag):
    user_response = input()
    user_response = user_response.lower()
    if(user_response != 'bye'):
        if(user_response == 'thanks' or user_response == 'thank you'):
            flag = False
            print("ROBO: You are welcome!")
        else:
            if(greeting(user_response) != None):
                print("ROBO: " + greeting(user_response))
            else:
                print("ROBO: ", end="")
                response_text = response(user_response)
                print(response_text)
                sent_tokens.remove(user_response)

                # Ask for feedback and update corpus if needed
                print("ROBO: Was this response helpful? (yes/no)")
                feedback = input().lower()
                if feedback == "no":
                    print("ROBO: Please provide the correct response:")
                    correct_response = input()
                    sent_tokens.append(correct_response)
                    print("ROBO: Thanks for providing feedback. I'll learn from that.")

                    # Update new words list
                    new_words.extend([word for word in nltk.word_tokenize(correct_response) if word not in new_words])

                    # Save new words to JSON file
                    with open('new_words.json', 'w') as file:
                        json.dump(new_words, file)
                else:
                    print("ROBO: Thanks for the feedback!")

    else:
        flag = False
        print("ROBO: Bye! Take care.")
