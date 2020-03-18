import pandas as pd
import os

"""
This code relies heavily on the form of the data. Namely it will fail if 
the authors of the same book are not comma separated. It will also be inaccurate
or even fail if the same author for different books is not spelt in exactly the
same way.
"""


loc = r'C:\Users\james\OneDrive\Documents\University\2017-18 Southampton\Data Mining\Group Coursework\Data'
        
#path = os.path.join(loc, r'Sample\new_books_data.csv')
path = os.path.join(loc, r'Processed_Data\new_books_data.csv')

books_data = pd.read_csv(path)


def split(string):
    """
    Function takes input of a string and returns an array of strings
    the original string should be comma separated with a space after
    the comma in order for this function to be accurate.
    """
    names = []
    index = 0
    last = 0
    for letter in string:
        if ((letter == ',') or (index == (len(string) - 1))):
            if (index == (len(string) - 1)):
                names.append(string[last:(index+1)])
            else:
                names.append(string[last:index])
                last = index+2
        index += 1
    return names


unique_authors = []
count = 0
for name in books_data['authors']:
    if (count%1000 == 0):
        print(count)
    split_names = split(name)
    for author in split_names:
        if (author in unique_authors):
            pass
        else:
            unique_authors.append(author)
    count += 1

authors_books = []
length = len(books_data.index)

count = 0
length_2 = len(unique_authors)
for author in unique_authors:
    if (count%100 == 0):
        print(str(count)+'/'+str(length_2))
    books = []
    for i in range(length):
        split_names = split(books_data['authors'][i])
        if (author in split_names):
            books.append(books_data['goodreads_book_id'][i])
    authors_books.append(books)
    count += 1

d = {'author': unique_authors, 'book_id': authors_books}
books_by_author = pd.DataFrame(data=d)

#write_path = os.path.join(loc, r'Sample\books_by_author.csv')
write_path = os.path.join(loc, r'Processed_Data\books_by_author.csv')
books_by_author.to_csv(write_path, index=False)



