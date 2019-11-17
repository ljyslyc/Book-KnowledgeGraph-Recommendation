# -*- coding: utf-8 -*-
import os
import xml.etree.ElementTree as etree
import csv
import getopt
import sys
import pandas as pd  
import numpy as np
import json
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
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer

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


def getRelevantDirectories(argv):

	inputDir = ''
	outputDir = ''
	modelDir = ''

	try:
		opts, args = getopt.getopt(argv,"hi:o:m:",["ifile=","ofile=","mfile="])
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
		elif opt in ("-m", "--mfile"):
			modelDir = arg
		elif opt in ("-o", "--ofile"):
			outputDir = arg
	return inputDir, outputDir, modelDir

inputDir, outputDir, modelDir = getRelevantDirectories(sys.argv[1:])

modelDir = os.path.abspath(modelDir)
inputDir = os.path.abspath(inputDir)
outputDir = os.path.abspath(outputDir)

def path_maker(path):
	if not os.path.exists(path):
		os.makedirs(path)

test_xml_path = os.path.join(inputDir,'en/text/')
result_xml_path = os.path.join(outputDir,'en/')
model_path = os.path.join(modelDir,'en_model.json')
dic_path = os.path.join(modelDir,'en_dictionary.json')
temp_twit = os.path.join(outputDir,'temp_twit.csv')
model_weight = os.path.join(modelDir,'en_model.h5')

path_maker(result_xml_path)

max_num = 2000

# read in your saved model structure
json_file = open(model_path, 'r')
loaded_model_json = json_file.read()
json_file.close()
# and create a model from that
model = model_from_json(loaded_model_json)
# and weight your nodes with your saved values
model.load_weights(model_weight)


# read in our saved dictionary
with open(dic_path, 'r') as dictionary_file:
	dictionary = json.load(dictionary_file)

def convert_text_to_index_array(text):
	words = kpt.text_to_word_sequence(text)
	wordIndices = []
	no_word = 0
	for word in words:
		if word in dictionary:
			wordIndices.append(dictionary[word])
		else:
			#print("'%s' not in training corpus; ignoring." %(word))
			not_found_word_list.append(word)
			no_word = no_word + 1
	return wordIndices,no_word

#----reading xml file data
for filename in os.listdir(test_xml_path):
	if not filename.endswith('.xml'): continue
	#----reading file name
	author_id = os.path.splitext(filename)[0]
	#----reading xml full path
	xml_fullname = os.path.join(test_xml_path, filename)
	tree = etree.parse(xml_fullname)
	root = tree.getroot()
	#----reading xml file data and storing in temp csv
	with open(temp_twit, 'w+') as csvfile:
		fieldnames = ['text']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
		for i in range(len(root[0])):
			writer.writerow({'text': (root[0][i].text).encode('utf-8')})
	csvfile.close()
	#--reading temp csv file
	my_df = pd.read_csv(temp_twit)
	my_df['text'] = my_df['text'].apply(lambda x: " ".join(stem(x) for x in x.split()))
	stop = stopwords.words('english')
	my_df['text'] = my_df['text'].apply(lambda x: " ".join(x for x in x.split(" ") if x not in stop))
	my_df['text'] = my_df['text'].apply(lambda x: " ".join(tokenize(x) for x in x.split(" ")))
	#my_df['text'] = my_df['text'].apply(lambda x: " ".join(x for x in text_processor.pre_process_doc(x) if x not in stop))
	# we're still going to use a Tokenizer here, but we don't need to fit it
	tokenizer = Tokenizer(num_words=max_num)
	# for human-friendly printing
	labels = ['female','male']
	# this utility makes sure that all the words in your input
	# are registered in the dictionary
	# before trying to turn them into a matrix.
	not_found_word_list = []
	male = 0
	female = 0
	for row in my_df.text:
		# okay here's the interactive part
		evalSentence = row
		# format your input for the neural net
		testArr,no_word = convert_text_to_index_array(evalSentence)
		input = tokenizer.sequences_to_matrix([testArr], mode='binary')
		# predict which bucket your input belongs in
		pred = model.predict(input)
		gender = labels[np.argmax(pred)]
		if gender == "male":
			male +=1
		else:
			female +=1
	if male >= female:
		gender = "male"
	else:
		gender = "female"
	text_to_write = """<author id='%s'\n\tlang='en'\n\tgender_txt='%s'\n/>"""% (author_id, gender)
	xml_result = os.path.join(result_xml_path, filename)
	fo = open(xml_result, "w+")
	fo.write( text_to_write )
	fo.close()