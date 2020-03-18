import pandas as pd
import os


loc = r'C:\Users\james\OneDrive\Documents\University\2017-18 Southampton\Data Mining\Group Coursework\Data'

path_1 = os.path.join(loc, r'ratings.csv')
path_2 = os.path.join(loc, r'Processed_Data\exclude_books.csv')

ratings_data = pd.read_csv(path_1)
exclude_ids = pd.read_csv(path_2)

exclude_ids = exclude_ids.sort_values(by=['book_id'])
ratings_data = ratings_data.sort_values(by=['user_id'], ascending=True)
ratings_data = ratings_data.set_index(ratings_data['user_id'])

new_ratings_data = ratings_data.copy()

length = len(ratings_data.index)

exclude_users = []
count = 1
index = 0
last_id = 1
for ids in ratings_data['user_id']:
    if (ids != last_id):
        if (count < 30):
            exclude_users.append(last_id)
        count = 1
        last_id = ids
    else:
        count += 1

d = {'user_id': exclude_users}
exclude_users_data = pd.DataFrame(data=d)

write_path = os.path.join(loc, r'Processed_Data\exclude_users_data.csv')
exclude_users_data.to_csv(write_path, index=False)

"""
for i in range(10):
    if (i%2 == 0):
        print(i)
    new_ratings_data = new_ratings_data.drop([exclude_ids['book_id'][i]])


for i in range(10,20):
    if (i%2 == 0):
        print(i)
    new_ratings_data = new_ratings_data.drop([exclude_ids['book_id'][i]])
    
for i in range(20,30):
    if (i%2 == 0):
        print(i)
    new_ratings_data = new_ratings_data.drop([exclude_ids['book_id'][i]])
    
for i in range(30,40):
    if (i%2 == 0):
        print(i)
    new_ratings_data = new_ratings_data.drop([exclude_ids['book_id'][i]])
"""
"""
for j in range(5):
    start = j*10
    end = (j+1)*10
    for i in range(start, end):
        if (i%2 == 0):
            print(i)
        new_ratings_data = new_ratings_data.drop([exclude_ids['book_id'][i]])
"""

