import pandas as pd
import os
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn

loc = r'C:\Users\james\OneDrive\Documents\University\2017-18 Southampton\Data Mining\Group Coursework\Data'
        
path_1 = os.path.join(loc, r'Processed_Data\new_book_tags.csv')
path_2 = os.path.join(loc, r'Processed_Data\empty_popular_tags.csv')

book_tags = pd.read_csv(path_1)
empty_tags = pd.read_csv(path_2)

new_count = []
new_goodreads_book_id = []
new_tag_id = []

length = len(book_tags['tag_id'])


for i in range(length):
    indicator = False
    if (i%1000 == 0):
        print(i)
    for tag in empty_tags['tag_id']:
        if (book_tags['tag_id'][i] == tag):
            indicator = True
            break
    if (indicator == False):
        new_count.append(book_tags['count'][i])
        new_goodreads_book_id.append(book_tags['goodreads_book_id'][i])
        new_tag_id.append(book_tags['tag_id'][i])

d = {'count': new_count, 'goodreads_book_id': new_goodreads_book_id, 'tag_id': new_tag_id}
cleaned_book_tags = pd.DataFrame(data=d)

write_path = os.path.join(loc, r'Processed_Data\cleaned_book_tags.csv')

cleaned_book_tags.to_csv(write_path, index=False)