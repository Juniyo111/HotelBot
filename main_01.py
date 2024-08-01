import nltk
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
import torch
import string

# Download required NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

# Function to load data from text files
def load_data(questions_file, responses_file):
    with open(questions_file, 'r', encoding='utf-8') as qf:
        questions = qf.readlines()
    with open(responses_file, 'r', encoding='utf-8') as rf:
        responses = rf.readlines()
    return [q.strip() for q in questions], [r.strip() for r in responses]

# Function to preprocess text
def preprocess(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    return filtered_tokens

# Load the questions and answers for training
questions, responses = load_data('[Dataset] Module27(ques).txt', '[Dataset] Module27 (ans).txt')

# Combine questions and responses for training
documents = questions + responses
tagged_data = [TaggedDocument(words=preprocess(doc), tags=[str(i)]) for i, doc in enumerate(documents)]

# Doc2Vec 모델 학습
doc2vec_model = Doc2Vec(
    vector_size=150,
    window=10,
    alpha=0.025,
    min_alpha=0.00025,
    min_count=2,
    dm=1,
    epochs=500
)
doc2vec_model.build_vocab(tagged_data)
doc2vec_model.train(tagged_data, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)
doc2vec_model.save("d2v.model")
print("Doc2Vec Model Saved")

# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# BERT Model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Add a comprehensive list of greeting expressions
greetings = [
    "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
    "how are you", "howdy", "what's up", "greetings", "salutations",
    "hi there", "hello there", "hey there", "yo", "sup", "hiya", "good day",
    "good to see you", "how's it going", "how have you been", "long time no see",
    "nice to see you", "pleased to meet you", "top of the morning to you",
    "what's new", "what's happening", "how's everything", "how's life", "how's your day",
    "how's your day going", "how are things", "what's good", "what's cracking",
    "howdy-do", "hiya", "hey, what's up", "yo, what's good"
]

greeting_responses = [
    "Hello! How can I assist you today?",
    "Hi there! How can I help?",
    "Hey! What can I do for you?",
    "Good morning! How can I be of service?",
    "Good afternoon! How can I assist?",
    "Good evening! What can I help you with?",
    "I'm here to help! How can I assist you?",
    "Howdy! What can I do for you?",
    "What's up? How can I assist?",
    "Greetings! How can I help you today?",
    "Salutations! What can I assist you with?",
    "Hi there! How can I make your day better?",
    "Hello there! How can I assist you?",
    "Hey there! How can I help you today?",
    "Yo! How can I assist?",
    "Sup? How can I help?",
    "Hiya! What can I do for you?",
    "Good day! How can I assist?",
    "Good to see you! How can I help?",
    "How's it going? What can I do for you?",
    "I've been well, thank you! How can I assist?",
    "Long time no see! How can I help?",
    "Nice to see you! How can I assist?",
    "Pleased to meet you! How can I help?",
    "Top of the morning to you! How can I assist?",
    "What's new? How can I help?",
    "What's happening? How can I assist?",
    "How's everything? How can I help?",
    "How's life? How can I assist?",
    "How's your day? How can I help?",
    "How's your day going? How can I assist?",
    "How are things? What can I do for you?",
    "What's good? How can I help?",
    "What's cracking? How can I assist?",
    "Howdy-do! How can I help?",
    "Hiya! What can I assist you with?",
    "Hey, what's up? How can I help?",
    "Yo, what's good? How can I assist?"
] * 2  # To ensure there are as many responses as greetings

# Function to find the best matching response
def get_response(user_input, doc2vec_model, tfidf_vectorizer, tfidf_matrix, bert_model, questions, responses):
    user_tokens = preprocess(user_input)
    user_vector_doc2vec = doc2vec_model.infer_vector(user_tokens).reshape(1, -1)

    # TF-IDF similarity
    user_vector_tfidf = tfidf_vectorizer.transform([user_input])
    tfidf_similarities = cosine_similarity(user_vector_tfidf, tfidf_matrix).flatten()

    # BERT similarity
    user_vector_bert = get_bert_embeddings(user_input)
    bert_similarities = []
    for doc in documents:
        doc_vector_bert = get_bert_embeddings(doc)
        sim = cosine_similarity(user_vector_bert, doc_vector_bert)[0][0]
        bert_similarities.append(sim)
    
    # Combine similarities
    combined_similarities = (tfidf_similarities + bert_similarities) / 2

    max_sim = -1
    best_response = "Sorry, I don't understand that."

    for i, sim in enumerate(combined_similarities):
        if sim > max_sim:
            max_sim = sim
            best_response = responses[i if i < len(responses) else i - len(responses)]

    return best_response

# Load the Doc2Vec model
doc2vec_model = Doc2Vec.load("d2v.model")

print("Hotel Reception Chatbot is ready. Type 'exit' to end the conversation.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit', 'bye']:
        print("Goodbye!")
        break
    elif user_input.lower() in greetings:
        print("Bot:", np.random.choice(greeting_responses))
    else:
        response = get_response(user_input, doc2vec_model, tfidf_vectorizer, tfidf_matrix, bert_model, questions, responses)
        print("Bot:", response)
