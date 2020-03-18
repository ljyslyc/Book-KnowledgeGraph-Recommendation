import pandas as pd
import os

loc = r'C:\Users\james\OneDrive\Documents\University\2017-18 Southampton\Data Mining\Group Coursework\Data'
        
path = os.path.join(loc, r'books.csv')

books_data = pd.read_csv(path)
new_books_data = books_data.copy()

def contain_string(string, sub_string):
    length = len(string)
    indicator = False
    for i in range(length):
        if (sub_string[0] == string[i]):
            for j in range(1,len(sub_string)):
                if ((i+j)==length):
                    if (j == (len(sub_string)-1)):
                        return False
                    else:
                        break
                if (sub_string[j] == string[i+j]):
                    indicator = True
                else:
                    indicator = False
    return indicator

index = 0


length = len(books_data.index)

exclude = []
book_id_exclude = []
for i in range(length):
    if (i%1000 == 0):
        print(i)
    lang = books_data['language_code'][i]
    books_count = books_data['books_count'][i]
    ratings_count = books_data['ratings_count'][i]
    if (isinstance(lang, str)):
        if ((contain_string(lang,'en') == False)
                            or (books_count < 30) 
                            or (ratings_count < 10000)):
            exclude.append(i)
            book_id_exclude.append(books_data['goodreads_book_id'][i])
    else:
        exclude.append(i)
        book_id_exclude.append(books_data['goodreads_book_id'][i])

new_books_data = new_books_data.drop(new_books_data.index[exclude])

write_path = os.path.join(loc, r'Processed_Data\new_books_data.csv')
new_books_data.to_csv(write_path, index=False)
