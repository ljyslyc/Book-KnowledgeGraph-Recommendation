cleaned_tags.csv
--------------------------------
Contains data for the most popular tags with insignificant tag data
removed.
FORMAT:
tag_id, tag_name

(tag_name is a list of strings where each element is a word in the tag)

cleaned_book_tags.csv
--------------------------------
Contains data for the most popular tags associated with
each book with insignifcant tags excluded.
FORMAT:
count, goodreads_book_id, tag_id

cleaned_books_data.csv
--------------------------------
Contains cleaned metadata about books. That is non english books, books
that have less than 30 copies and books with less than 10,000 ratings have been removed.
FORMAT:
book_id	goodreads_book_id	best_book_id	work_id	books_count	isbn	isbn13	authors	original_publication_year	original_title	title	language_code	average_rating	ratings_count	work_ratings_count	work_text_reviews_count	ratings_1	ratings_2	ratings_3	ratings_4	ratings_5	image_url	small_image_url







