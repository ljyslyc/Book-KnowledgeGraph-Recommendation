# -*- coding: utf-8 -*-
import os
import xml.etree.ElementTree as etree
import pandas as pd
import csv
import getopt
import sys
import numpy as np
import json
import copy
from sklearn import utils
import re
from nltk.corpus import stopwords
import keras
import keras.preprocessing.text as kpt
import nltk
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Activation , LSTM , Input , Embedding
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, concatenate, Activation, Average
from keras.models import Model
from keras import optimizers
from keras.callbacks import CSVLogger


def getRelevantDirectories(argv):

	inputDir = ''
	outputDir = ''
	modelDir = ''

	try:
		opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
	except getopt.GetoptError:
		print( './myTrainingSoftware.py -i <inputdirectory> -o <outputdirectory>')
		print( 'The input directory should contain all the training files. \nThe output directory will be where the models are stored.')
		sys.exit(2)

	for opt, arg in opts:
		if opt == '-h':
			print( './myTrainingSoftware.py -i <inputdirectory> -o <outputdirectory>')
			print( 'The input directory should contain all the training files. \nThe output directory will be where the models are stored.')
			sys.exit()
		elif opt in ("-i", "--ifile"):
			inputDir = arg
		elif opt in ("-o", "--ofile"):
			outputDir = arg
	return inputDir, outputDir

def path_maker(path):
	if not os.path.exists(path):
		os.makedirs(path)

inputDir, outputDir = getRelevantDirectories(sys.argv[1:])

modelDir = os.path.abspath(outputDir)
inputDir = os.path.abspath(inputDir)

path_maker(modelDir)

xml_path = os.path.join(inputDir,'es/text/')
label_path = os.path.join(inputDir,'es/truth.txt')
label_file = open(label_path)
model_path = os.path.join(modelDir,'es_model.json')
dic_path = os.path.join(modelDir,'es_dictionary.json')
twit_path = os.path.join(modelDir,'twitter_es_data_set.csv')

with open(twit_path, 'w+') as csvfile:
	fieldnames = ['text','gender']
	writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
	writer.writeheader()
	for filename in os.listdir(xml_path):
		if not filename.endswith('.xml'): continue
		#----reading file name
		find_file_name = os.path.splitext(filename)[0]
		#print(find_file_name)
		#----finding author gender
		label_file = open(label_path)
		for row in label_file:
			if find_file_name in row:
				#print(find_file_name)
				#print(row.split(':::')[1])
				gender = row.split(':::')[1]
		
		#print(gender)

		#----reading xml full path
		xml_fullname = os.path.join(xml_path, filename)
		tree = etree.parse(xml_fullname)
		root = tree.getroot()
		#----reading xml file data
		for i in range(len(root[0])):
			writer.writerow({'text': str(root[0][i].text), 'gender': str(gender)})
csvfile.close()

FLAGS = re.MULTILINE | re.DOTALL


def stem(word):
		regexp = r'^(.*?)(ing|ious|ment)?$'
		stem, suffix = re.findall(regexp, word)[0]
		return stem

def tokenize(text):
    # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)
    
    #text = re_sub(r"#\S+", "hashtag")
    #text = re_sub(r"([!?.]){2,}", r" \1 ")
    #text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 ")
    #text = re_sub(r"([A-Z]){2,}", "allcaps")
    #text = re_sub(r'([\w!.,?();*\[\]":\”\“])([!.,?();*\[\]":\”\“])', r'\1 \2')
    #text = re_sub(r'([!.,?();*:\[\]":\”\“])([\w!.,?();*\[\]":\”\“])', r'\1 \2')
    #text = re_sub(r'(.)(<)', r'\1 \2')
    #text = re_sub(r'(>)(.)', r'\1 \2')
    #text = re_sub(r'[\'\`\’\‘]', r'')
    #text = re_sub(r'\\n', r' ')
    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", " ")
    #text = re_sub(r"#(\S+)", r" ") # replace #name with name
    text = re_sub(r"(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))", " em_positive ") # Smile -- :), : ), :-), (:, ( :, (-:, :')
    text = re_sub(r"(:\s?D|:-D|x-?D|X-?D)", " em_positive ") # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    text = re_sub(r"(<3|:\*)", " em_positive ") # Love -- <3, :*
    text = re_sub(r"(;-?\)|;-?D|\(-?;)", " em_positive ") # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    text = re_sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', " em_negative ") # Sad -- :-(, : (, :(, ):, )-:
    text = re_sub(r'(:,\(|:\'\(|:"\()', " em_negative ") # Cry -- :,(, :'(, :"(
    text = re_sub(r"(-|\')", "") # remove &
    text = re_sub(r"/"," / ")
    text = re_sub(r"@[0-9]+-", " ")
    text = re_sub(r"@\w+", " ")
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), " em_positive ")
    text = re_sub(r"{}{}p+".format(eyes, nose), " em_positive ")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), " em_negative ")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), " em_neutralface ")
    text = re_sub(r"<3"," heart ")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", " ")
    text = re_sub(r'-', r' ')
    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "  ")
#     text = re_sub(r"([pls?s]){2,}", r"\1")
#     text = re_sub(r"([plz?z]){2,}", r"\1")
    text = re_sub(r'\\n', r' ')
    text = re_sub(r" app "," application ")
    text = re_sub(r"app"," application")
    text = re_sub(r" wil "," will ")
    text = re_sub(r" im "," i am ")
    text = re_sub(r" al "," all ")
    #text = re_sub(r"<3","love")
    text = re_sub(r" sx "," sex ")
    text = re_sub(r" u "," you ")
    text = re_sub(r" r "," are ")
    text = re_sub(r" y "," why ")
    text = re_sub(r" Y "," WHY ")
    text = re_sub(r"Y "," WHY ")
    text = re_sub(r" hv "," have ")
    text = re_sub(r" c "," see ")
    text = re_sub(r" bcz "," because ")
    text = re_sub(r" coz "," because ")
    text = re_sub(r" v "," we ")
    text = re_sub(r" ppl "," people ") 
    text = re_sub(r" pepl "," people ")
    text = re_sub(r" r b i "," rbi ")
    text = re_sub(r" R B I "," RBI ")
    text = re_sub(r" R b i "," rbi ")
    text = re_sub(r" R "," ARE ")
    text = re_sub(r" hav "," have ")
    text = re_sub(r"R "," ARE ")
    text = re_sub(r" U "," you ")
    text = re_sub(r" 👎 "," em_negative ")
    text = re_sub(r"U "," you ")
    text = re_sub(r" pls "," please ")
    text = re_sub(r"Pls ","Please ")
    text = re_sub(r"plz ","please ")
    text = re_sub(r"Plz ","Please ")
    text = re_sub(r"PLZ ","Please ")
    text = re_sub(r"Pls","Please ")
    text = re_sub(r"plz","please ")
    text = re_sub(r"Plz","Please ")
    text = re_sub(r"PLZ","Please ") 
    text = re_sub(r" thankz "," thanks ")
    text = re_sub(r" thnx "," thanks ")
    text = re_sub(r"fuck\w+ "," fuck ")
    text = re_sub(r"f\*\* "," fuck ")
    text = re_sub(r"\*\*\*k "," fuck ")
    text = re_sub(r"F\*\* "," fuck ")
    text = re_sub(r"mo\*\*\*\*\* "," fucker ")
    text = re_sub(r"b\*\*\*\* "," blody ")
    text = re_sub(r" mc "," fucker ")
    text = re_sub(r" MC "," fucker ")
    text = re_sub(r" wtf "," fuck ")
    text = re_sub(r" ch\*\*\*ya "," fucker ")
    text = re_sub(r" ch\*\*Tya "," fucker ")
    text = re_sub(r" ch\*\*Tia "," fucker ")
    text = re_sub(r" C\*\*\*yas "," fucker ")
    text = re_sub(r"l\*\*\*\* ","shit ")
    text = re_sub(r" A\*\*\*\*\*\*S"," ASSHOLES")
    text = re_sub(r" di\*\*\*\*s"," cker")
    text = re_sub(r" nd "," and ")
    text = re_sub(r"Nd ","and ")
    text = re_sub(r"([!?!]){2,}", r"! ")
    text = re_sub(r"([.?.]){2,}", r". ")
    text = re_sub(r"([*?*]){2,}", r"* ")
    text = re_sub(r"([,?,]){2,}", r", ")
    text = re_sub(r"ha?ha", r" em_positive ")
    #text = re_sub(r"([!]){2,}", r"! ")
    #text = re_sub(r"([.]){2,}", r". ")
    #text = re_sub(r"([*]){2,}", r"* ")
    #text = re_sub(r"([,]){2,}", r", ")
    #text = re_sub(r"\n\r", " ") 
    text = re_sub(r"(ind[vs]pak)", " india versus pakistan ")
    text = re_sub(r"(pak[vs]ind)", " pakistan versus india ")
    text = re_sub(r"(indvsuae)", " india versus United Arab Emirates ")
    text = re_sub(r"[sS]hut[Dd]own[jnuJNU]", " shut down jnu ")
    #text = re_sub(r"ShutDownJNU", " shut down jnu ")
    #text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", " number ")
    #text = re_sub(r"(.)\1\1+", r"\1") # remove funnnnny --> funny
    return text

my_df = pd.read_csv(twit_path)
my_df['gender'] = my_df['gender'].map({'female\n': 0, 'male\n': 1})
stop = stopwords.words('spanish')
my_df['text'] = my_df['text'].apply(lambda x: " ".join(x for x in x.split(" ") if x not in stop))
my_df['text'] = my_df['text'].apply(lambda x: " ".join(tokenize(x) for x in x.split(" ")))
#my_df['text'] = my_df['text'].apply(lambda x: " ".join(x for x in text_processor.pre_process_doc(x) if x not in stop))
freq = pd.Series(' '.join(my_df['text']).split()).value_counts()[:50]
freq = list(freq.index)
my_df['text'] = my_df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

from keras.preprocessing.text import Tokenizer

train_x = my_df.text
train_y = my_df.gender

# create a new Tokenizer
max_num = 2000

tokenizer = Tokenizer(num_words= max_num)
# feed our tweets to the Tokenizer
tokenizer.fit_on_texts(train_x)

from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

#Tokenizers come with a convenient list of words and IDs
dictionary = tokenizer.word_index
# Let's save this out so we can use it later
with open(dic_path, 'w+') as dictionary_file:
    json.dump(dictionary, dictionary_file)

def convert_text_to_index_array(text):
    # one really important thing that `text_to_word_sequence` does
    # is make all texts the same length -- in this case, the length
    # of the longest text in the set.
    temp_wordIndices = []
    for word in kpt.text_to_word_sequence(text):
        if word in dictionary:
            temp_wordIndices.append(dictionary[word])
    return temp_wordIndices

allWordIndices = []
# for each tweet, change each token to its ID in the Tokenizer's word_index
for text in train_x:
    wordIndices = convert_text_to_index_array(text)
    allWordIndices.append(wordIndices)

# now we have a list of all tweets converted to index arrays.
# cast as an array for future usage.
allWordIndices = np.asarray(allWordIndices)

# create one-hot matrices out of the indexed tweets
train_x = tokenizer.sequences_to_matrix(allWordIndices, mode='binary')
# treat the labels as categories
train_y = keras.utils.to_categorical(train_y, 2)



model = Sequential()


model.add(Dense(1024, activation='relu',input_shape=(max_num,)))
model.add(Dropout(0.5))
model.add(Dense(512, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

# print(model.summary())

filepath="sequencing_the_data_try_n_error.{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
csv_logger = CSVLogger(os.path.join(modelDir,'es_model_log.csv'), append=True, separator=';')

model.fit(train_x, train_y,
    batch_size=32,
    epochs=2,
    verbose=1,
    validation_split=0.1,
    shuffle=True,callbacks = [csv_logger])

model_json = model.to_json()

with open(model_path, 'w+') as json_file:
    json_file.write(model_json)

model.save_weights(os.path.join(modelDir,'es_model.h5'))

# print('saved model!')

