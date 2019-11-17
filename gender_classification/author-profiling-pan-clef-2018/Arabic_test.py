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
import nltk
import fasttext

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

test_xml_path = os.path.join(inputDir,'ar/text/')
result_xml_path = os.path.join(outputDir,'ar/')
model_path = os.path.join(modelDir,'ft_ar_model.bin')

path_maker(result_xml_path)

model = fasttext.load_model(model_path, encoding='utf-8',label_prefix='label__')

#----reading xml file data
for filename in os.listdir(test_xml_path):
	if not filename.endswith('.xml'): continue
	#----reading file name
	author_id = os.path.splitext(filename)[0]
	#----reading xml full path
	xml_fullname = os.path.join(test_xml_path, filename)
	tree = etree.parse(xml_fullname)
	root = tree.getroot()
	#----reading xml file data and storing in temp list
	ls = []
	for i in range(len(root[0])):
		ls.append(str(root[0][i].text).replace("\n"," "))

	male = 0
	female = 0
	gender_list = model.predict(ls)
	for gender in gender_list:
		gender = str(gender).replace("'", "").replace("[", "").replace("]", "")
		if gender == "male":
				male +=1
		else:
			female +=1

	if male >= female:
		gender = "male"
	else:
		gender = "female"
	text_to_write = """<author id='%s'\n\tlang='ar'\n\tgender_txt='%s'\n/>"""% (author_id, gender)
	xml_result = os.path.join(result_xml_path, filename)
	fo = open(xml_result, "w+")
	fo.write( text_to_write )
	fo.close()
