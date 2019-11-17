# author-profiling-pan-clef-2018
Author Profiling: Gender Identification in Twitter

Authorship analysis deals with the classification of texts into classes based on the stylistic choices of their authors.
Beyond the author identification and author verification tasks where the style of individual authors is examined, 
author profiling distinguishes between classes of authors studying their sociolect aspect, that is, how language is shared by
people. This helps in identifying profiling aspects such as gender, age, native language, or personality type.
Author profiling is a problem of growing importance in applications in forensics, security, and marketing.
E.g., from a forensic linguistics perspective one would like being able to know the linguistic profile of the author 
of a harassing text message (language used by a certain type of people) and identify certain characteristics
(language as evidence). Similarly, from a marketing viewpoint, companies may be interested in knowing, on the basis of 
the analysis of blogs and online product reviews, the demographics of people that like or dislike their products. 
The focus is on author profiling in social media since we are mainly interested in everyday language and how it reflects 
basic social and personality processes.

# Task

The focus was on gender identification in Twitter, where text and images may be used as information sources.
The languages:English, Spanish , and Arabic

# Training Corpus

To develop a software, PAN provided with a training data set that consisted of Twitter users labeled with gender. 
For each author, a total of 100 tweets and 10 images are provided. Authors are grouped by the language of their
tweets: English, Arabic and Spanish.

Dataset: https://s3.amazonaws.com/autoritas.pan/pan18-author-profiling-training-2018-02-27.zip

# Output

The software must take as input the absolute path to an unpacked dataset, and has to output for each document of the dataset
a corresponding XML file that looks like this:

  <author id="author-id"
	  lang="en|es|ar"
	  gender_txt="male|female"
	  gender_img="male|female"
	  gender_comb="male|female"
  />

# Code

For traing (over only text), the python files are:

Engilsh_train.py
Spanish_train.py
Arabic_train.py

For testing (over only text), the python files are:

Engilsh_test.py
Spanish_test.py
Arabic_test.py

# Reference

Rangel, F., Rosso, P., Montes-y Gómez, M., Potthast, M., Stein, B.: Overview of the 6th 
author profiling task at pan 2018: Multimodal gender identification in twitter. In: CLEF
2018 Labs and Workshops, Notebook Papers. CEUR Workshop Proceedings (2018),https://CEUR-WS.org
