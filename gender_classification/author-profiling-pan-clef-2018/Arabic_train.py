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
import nltk
import fasttext


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

xml_path = os.path.join(inputDir,'ar/text/')
label_path = os.path.join(inputDir,'ar/truth.txt')
label_file = open(label_path)
model_path = os.path.join(modelDir,'ft_ar_model')
twit_path = os.path.join(modelDir,'twitter_ar_data_set.txt')

with open(twit_path, 'w+') as txtfile:
	for filename in os.listdir(xml_path):
		if not filename.endswith('.xml'): continue
		#----reading file name
		find_file_name = os.path.splitext(filename)[0]
		#----finding author gender
		label_file = open(label_path)
		for row in label_file:
			if find_file_name in row:
				gender = row.split(':::')[1]

		xml_fullname = os.path.join(xml_path, filename)
		tree = etree.parse(xml_fullname)
		root = tree.getroot()
		gender = str(gender).replace("\n","")
		for i in range(len(root[0])):
			txtfile.write("label__"+ gender + " " + str(root[0][i].text).replace("\n"," ")+"\n")
					
txtfile.close()

classifier = fasttext.supervised(twit_path, model_path, label_prefix='label__', lr=0.25, epoch=40, loss='softmax')