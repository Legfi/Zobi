
#importing libraries
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tflearn
from tensorflow.python.framework import ops
import random
import json
import discord
import os
from dotenv import load_dotenv
import numpy

#reading the intents file
with open('intents.json') as file:
    data = json.load(file)

#making empty list to nlp prossesing
words = []
labels = []
docs_x = []
docs_y = []

#tokenizing patterns in intent file
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

#adding tags in labels list
        if intent["tag"] not in labels:
            labels.append(intent["tag"])

#stemming the word list and sorting it
words = [stemmer.stem(w.lower()) for w in words if w != "?"] 
words = sorted(list(set(words)))
labels = sorted(labels)

#preparing the training list and output list to train the model
training = []
output = []

#making a bag of world
out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

#making deeplearning model    
ops.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
#loading model if exist 
try:
    model.load("model.tflearn")
#fiting model if not exist
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    #saving the trained model
    model.save("model.tflearn")

#bag of word function
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)

#loading dotenv file to connecting bot to discord server
load_dotenv()
TOKEN = os.getenv('TOKEN')
client = discord.Client()

@client.event
async def on_message(message):

        if message.content.find("!") != -1:
            inp = message.content[1::]


            results = model.predict([bag_of_words(inp, words)])[0]
            results_index = numpy.argmax(results)
            tag = labels[results_index]

            #if the result has lower chance than 80% then ask to repeat
            if results[results_index] > 0.8:
                for tg in data["intents"]:
                    if tg['tag'] == tag:
                        responses = tg['responses']
            else:
                print("I didn't quite get that, try agin!")
            await message.channel.send(random.choice(responses))

client.run(TOKEN)