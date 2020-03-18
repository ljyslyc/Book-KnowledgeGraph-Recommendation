import pandas as pd
import os
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn

loc = r'C:\Users\james\OneDrive\Documents\University\2017-18 Southampton\Data Mining\Group Coursework\Data'
        
path_1 = os.path.join(loc, r'Processed_Data\new_book_tags.csv')
path_2 = os.path.join(loc, r'tags.csv')

books_tags_data = pd.read_csv(path_1)
tags_data = pd.read_csv(path_2)

tag_id = []
#finds all unique popular tag ids
for tag in books_tags_data['tag_id']:
    if (tag not in tag_id):
        tag_id.append(tag)

"""
create dataframe containing all unique popular tag ids and their
corresponding name (uncleaned)
"""
tag_name = []
counter=0
for id_1 in tag_id:
    index = 0
    if (counter%200 == 0):
        print(counter)
    for id_2 in tags_data['tag_id']:
        if (id_1 == id_2):
            tag_name.append(tags_data['tag_name'][index])
        index += 1
    counter += 1

d = {'tag_id': tag_id, 'tag_name': tag_name}
popular_tags = pd.DataFrame(data=d)

def alphanumeric(string):
    """
    Returns True if string contains both letters
    and numbers
    """
    if (string.isalpha()):
        return False
    if (string.isnumeric()):
        return False
    else:
        return True

"""
clean the tag_names
"""
new_tags = popular_tags.copy() #makes a copy of dataframe to contain processed tags

index = 0
end = 0
for tag in popular_tags['tag_name'][:]:
    if (index%300 == 0):
        print(index)
    tag = tag.lstrip('-')
    tag += '-'
    tag = tag.lower()
    tag_words = []
    sub_word = ''
    for letter in tag:
        if ((letter != '-') and (letter != '_')):
            sub_word += letter
        else:
            #while sub_word is valid
            while ((sub_word != '') and (end != 1)):
                #contains numbers and letters
                if (alphanumeric(sub_word)):
                    sub_word = ''
                #numerical non-date
                if ((sub_word.isnumeric() == True) and 
                    ((len(sub_word) != 4) or (sub_word[0] not in [1,2]) )):
                    sub_word = ''
                #stop word
                if (sub_word in stopwords.words()):
                    sub_word = ''
                #less than 3 letters
                if (len(sub_word) < 3):
                    sub_word = ''
                #not about book contents
                if (sub_word in ['read','buy','available']):
                    sub_word = ''
                end = 1

            #if valid append
            if (sub_word == ''):
                pass
            else:
                tag_words.append(sub_word)
            sub_word = ''
            end = 0
    new_tags['tag_name'][index] = tag_words
    index += 1

"""
makes list of empty tags
and creates new dataframe excluding empty tags
"""
empty_tags = []
tag_id_cleaned = []
tag_name_cleaned = []
index = 0
for tag in new_tags['tag_name'][:]:
    if (tag == []):
        empty_tags.append(new_tags['tag_id'][index])
    else:
        tag_id_cleaned.append(new_tags['tag_id'][index])
        tag_name_cleaned.append(new_tags['tag_name'][index])
    index += 1

d_1 = {'tag_id': tag_id_cleaned, 'tag_name': tag_name_cleaned}
d_2 = {'tag_id': empty_tags}
cleaned_tags = pd.DataFrame(data=d_1)
empty_tags_df = pd.DataFrame(data=d_2)

write_path_1 = os.path.join(loc, r'Processed_Data\new_popular_tags.csv')
write_path_2 = os.path.join(loc, r'Processed_Data\cleaned_popular_tags.csv')
write_path_3 = os.path.join(loc, r'Processed_Data\empty_popular_tags.csv')

new_tags.to_csv(write_path_1, index=False)
cleaned_tags.to_csv(write_path_2, index=False)
empty_tags_df.to_csv(write_path_3, index=False)




