import pandas as pd
import os


loc = r'C:\Users\james\OneDrive\Documents\University\2017-18 Southampton\Data Mining\Group Coursework\Data'
        
path = os.path.join(loc, r'book_tags.csv')

book_tags_data = pd.read_csv(path)

new_goodbooks_id = []
new_tag_id = []
new_count = []

index = 0
sub_index = 1
max_num_tags = 6

new_goodbooks_id.append(book_tags_data['goodreads_book_id'][0])
new_tag_id.append(book_tags_data['tag_id'][0])
new_count.append(book_tags_data['count'][0])
max_count = book_tags_data['count'][0]

for count in book_tags_data['count']:
    if (index%100000 == 0):
        print(index)
    if (index == 0):
        pass
    else:
        if (book_tags_data['goodreads_book_id'][index] != new_goodbooks_id[-1]):
            max_count = count
            new_goodbooks_id.append(book_tags_data['goodreads_book_id'][index])
            new_tag_id.append(book_tags_data['tag_id'][index])
            new_count.append(count)
            sub_index = 0
        elif ((sub_index < max_num_tags) or (count>(0.02*max_count))):
            """
            adds top max_num_tags and all tags whose count is more than
            2% of the largest tag count for that book
            """
            new_goodbooks_id.append(book_tags_data['goodreads_book_id'][index])
            new_tag_id.append(book_tags_data['tag_id'][index])
            new_count.append(count)
            sub_index += 1
    index += 1

d_2 = {'goodreads_book_id': new_goodbooks_id, 'tag_id': new_tag_id, 'count': new_count}
new_book_tags = pd.DataFrame(data=d_2)

write_path = os.path.join(loc, r'Processed_Data\new_book_tags.csv')

new_book_tags.to_csv(write_path, index=False)