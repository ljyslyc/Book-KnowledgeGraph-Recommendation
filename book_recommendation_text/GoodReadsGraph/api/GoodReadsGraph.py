import pandas as pd
import numpy as np

class Graph(object):
    def __init__(self, reads):
        """

        """
        # Edges in our graph
        self.reads = reads


    def _find_a_user(self, input_User, debug=False):
        """
        Find a user in the graph N hops from an author in the
        user's list of read authors:
        """
        _author_list = input_User.author_list
        if debug:
            print ("Method: _find_N_user : `_author_list`: ", _author_list)
        # Do we have any authors in the list?
        _n_authors = len(_author_list)
        if _n_authors > 0:
            # Pick an Author. Any Author.
            _reader_list = None
            while _reader_list == None:
                _next_author = _author_list[np.random.randint(_n_authors)] #Inclusive, random integer
                if debug:
                    print ("Method: _find_N_user : `_next_author` : ", _next_author)
                if len(_next_author.reader_list) > 1: # Is there anyone in this bucket?
                    _reader_list = _next_author.reader_list # Who reads this Author?

            _next_User = None
            while _next_User == None:
                _choice = _reader_list[np.random.randint(len(_reader_list))]
                if _choice != input_User:
                    _next_User = _choice # Make sure we do not pick ourselves
            if debug:
                print ("Method: _find_N_user : `_next_User`: ", _next_User)
            return _next_User # We finally made a choice!
        else:
            return None

    def _book2book(self, input_Book, N=3 , debug=False):
        """ Sanity Checker:
        Developer Function to quickly get similar, unpopular books
        recommended that is not based on a User. Simply input a book,
        go up the Author tree, and then find a random user, and their
        unpopular book.

        This is quicker, in theory, for testing out the predictions, as you
        don't have to build a User + Read objects for the Graph.
        """
        def _sort_tuple(tuple_val):
            """ For sorting our unread list based on popularity
                [(book1,popularity1),...,(bookn,popularityn)]
            """
            return tuple_val[1]

        out_recs = []
        for i in range(N):
            _reader_list = input_Book.reader_list
            _len_rl = len(_reader_list)
            _rand_User = _reader_list[np.random.randint(_len_rl)]
            _list = [(book, book.popularity, rating) for book, rating in _rand_User.shelf
                    if rating > 4] #NB; No filtering on read books, as no input user.
            _list = sorted(_list, key=_sort_tuple, reverse=False)
            unpopular_book, popularity, rating = _list[0]
            out_recs.append(unpopular_book)

        return out_recs

    def _find_a_book(self, input_User, two_hop=False, debug=False):
        """
        Given a user, recommend an unpopular book:
        1) Take an input_user, go to an Author node, grab another user that has
            read that author.
        2) Grab that User's book list and compare it to the input user.
        """
        def _sort_tuple(tuple_val):
            """ For sorting our unread list based on popularity
                [(book1,popularity1),...,(bookn,popularityn)]
            """
            return tuple_val[1]

        if debug:
            print ("Method: _find_a_book : `input_User`: ", input_User)

        _next_User = self._find_a_user(input_User, debug=debug)

        if two_hop:
            try:
                _two_hop = self._find_a_user(_next_User, debug)
                _next_User = _two_hop if _two_hop != input_User else _next_User
            except Exception as e:
                if debug:
                    print ("Method: _find_a_book : Exception at `two_hop`: ", input_User, e)

        if debug:
            print ("Method: _find_a_book : `_next_User`: ", _next_User)

        counter= 0
        while counter < 100:
            counter+=1

            """
            First, let's see how many books this user has read that
            the input_User has not AND is rated above 4 stars.
            NB: We could also add a maximum popularity here, just in case!
            This will form our set from which we can find unpopular books:
            """
            try:
                _unread_list = [(book, book.popularity, rating) for book, rating in _next_User.shelf
                                if book not in [_books for _books, _rating in input_User.shelf] and rating > 4]
                _n_unread = len(_unread_list)
                if debug:
                    print ("Method: _find_a_book : Length of the unread shelf: ", _n_unread)
            except Exception as e:
                print ("Method: _find_a_book : `_unread_list` threw an exception: ", _next_User, e)

            """
            Now, we take our unsorted, unread list of books, and sort them
            in ascending order. The first entry should be our best bet!
            """
            try:
                _unread_list = sorted(_unread_list, key=_sort_tuple, reverse=False)

                if debug:
                    if _n_unread > 1:
                        print ("Method: _find_a_book : Most unpopular book title, popularity, and rating ",
                               _unread_list[0][0].book_id, _unread_list[0][1])
                        print ("Method: _find_a_book : Most popular book title and popularity ",
                               _unread_list[_n_unread-1][0].book_id, _unread_list[_n_unread-1][1])
                    else:
                        print ("Method: _find_a_book : Most unpopular book title and popularity ",
                               _unread_list[0][0].book_id, _unread_list[0][1])
            except Exception as e:
                if debug:
                    print ("Method: _find_a_book : `_unread_list` sorting threw an exception: ", e)

            # So we may have found a good, rare book. Return it!
            unpopular_book, popularity, rating = _unread_list[0]
            if unpopular_book != None:
                return unpopular_book
        # Base case: We did not find any good books.
        return None

    def GrabNBooks(self, input_User, N=3, debug=False):
        """
        Our main class to find three unpopular books. Relies on two helper classes:
            _find_a_book: Grabs a rare books from a neighbor that reads similar books to you
            _find_a_user: Finds the neighbor to a book from your collection!

        If you enable two_hop = True in your calls, it can help preserve the privacy of your users,
            as you really start to jump around the graph. Want more privacy? Enable more random jumps.
        """
        if debug:
            print ("Method: GrabThreeBooks : Beginning :", input_User.user_id)

        RareBooks = []
        counter = 0
        while counter < 100:
            """
            try:
                _book = self._find_a_book(input_User, debug)

                if _book != None:
                    RareBooks.append(_book)
            except Exception as e:
                if debug:
                    print ("Method: GrabThreeBooks : Exception = ", e)
            """
            _book = self._find_a_book(input_User, debug=debug)
            RareBooks.append(_book)
            if len(RareBooks) == N:
                return RareBooks
            # Increase the counter so that we don't get stuck in a loop
            else:
                counter+=1
        #Base case in case something goes wrong...
        return None


class User(object):
    def __init__(self,user_id):
        self.user_id = user_id
        self.shelf = [] # Books read
        self.author_list = [] # Authors read

class Book(object):
    def __init__(self, book_id, Author, ratings_5, popularity, image_url):
        self.book_id = book_id
        self.author = Author
        self.author_id = Author.author_id
        self.ratings_5 = ratings_5 # Number of people that rated the book a 5
        self.popularity = popularity # What fraction of ratings does this book have?+
        self.image_url = image_url
        self.reader_list = [] #Users that read the book

    def add_reader(self,User):
        if User not in self.reader_list:
            self.reader_list.append(User) # User read this book

class Author(object):
    def __init__(self, author_id):
            self.author_id = author_id
            self.reader_list = [] #People who read the book

    def add_reader(self,User):
        if User not in self.reader_list:
            self.reader_list.append(User) # User read this book


class Read(object):
    def __init__(self, User, Book, Author, rating=None):
        """
        The edge connecting User, Book, and Author nodes
        """
        if Book not in User.shelf:
            User.shelf.append((Book, rating)) # User read this book and rated it.
        if Author not in User.author_list:
            User.author_list.append(Author)

        self.user = User
        self.book = Book
        self.author = Author
        self.rating = rating # Optional

        Book.add_reader(User)
        Author.add_reader(User)


def BuildGraph():
    """
    Now we use real data:
    `uir` : user,item,rating data
    `books`: meta information about each of the items (# of ratings, Author, etc.)
    """
    uir = pd.read_csv("api/data/goodbooks-10k-master/ratings.csv")

    books = pd.read_csv("api/data/goodbooks-10k-master/books.csv")
    books = books[(books["language_code"] == "eng") | (books["language_code"] == "en-US")]
    books["author_id"] = (books["authors"].astype("category")).cat.codes # Gives us an index

    """
    Let's build a few versions of popularity: overall ratings, text review counts, and
    fraction of all people that rated this book with 5-stars.
    """
    books["popularity_ratings"] = books["ratings_count"]/np.sum(books["ratings_count"])
    books["popularity_text_reviews"] = books["work_text_reviews_count"]/np.sum(books["work_text_reviews_count"])
    books["popularity_ratings5" ]= books["ratings_5"]/np.sum(books["ratings_5"])

    """
    Join these two dataframes together:
    1) This filters out non-English books
    2) Gives us book info as well as the Author
    """
    uir = pd.merge(uir, books[["book_id", "original_title",
                               "author_id","popularity_ratings","ratings_5", "image_url"]], on=["book_id"])

    """
    Let's build a catelog of Author objects first,
    as they do not depend on any other objects in our graph.
    """
    unique_authors = uir[["author_id"]].drop_duplicates()
    unique_authors["Author"] = [Author(aid) for aid in unique_authors["author_id"]]
    unique_authors = unique_authors.set_index("author_id", drop=True)

    """
    Now we can do the same for the users:

    """
    unique_users = uir[["user_id"]].drop_duplicates()
    unique_users["User"] = [User(uid) for uid in unique_users["user_id"]]
    unique_users = unique_users.set_index("user_id", drop=True)

    """
    We can build a set of dictionaries for easy reference later
    """
    user_dict = unique_users.to_dict("index")
    author_dict =  unique_authors.to_dict("index")

    """
    There are a number of ways we could proceed now, depending on our
    space and speed constraints. If we had memory limitations, we could
    save our unique_users and unique_authors dataframes as Dictionaries,
    then just call them whenever needed. I think for our relatively small
    dataset, we could just join them to our original dataframe:

      `uir = pd.merge(uir, unique_users, on=["user_id"])`
      `uir = pd.merge(uir, unique_authors, on=["author_id"])`

    and then process the Books inline with a list comprehension:

      `uir["Book"] = [Book(bid, aid, rat, pop , url) for bid, aid, rat, pop , url
                 in uir[["book_id","Author","ratings_5","popularity_ratings","image_url"]].values]`

    But I don't want to be too lazy here, so we will use the dictionary route:
    """

    unique_books = uir[["book_id", "original_title", "author_id", "ratings_5", "popularity_ratings",
                        "image_url"]].drop_duplicates()
    unique_books["Book"] = [Book(bid, author_dict[aid]["Author"], rat, pop, url) for bid, aid, rat, pop, url
                            in unique_books[
                                ["book_id", "author_id", "ratings_5", "popularity_ratings", "image_url"]].values]

    # Now that we have our Book objects, let's build it into a dictionary
    _unique_books = unique_books.set_index("book_id", drop=True)
    _unique_books = _unique_books.drop(["author_id", "ratings_5", "popularity_ratings", "image_url"],
                                       axis=1)  # Drop everything
    book_dict = _unique_books.to_dict("index")

    """
    We also need a title lookup for the User facing entries:
    1) Key is a title (lower_case!)
    2) Value is a Book `Object`
    """
    _unique_titles = unique_books.copy()
    _unique_titles["original_title"] = _unique_titles["original_title"].str.lower()
    _unique_titles = _unique_titles.drop(["author_id", "book_id", "ratings_5", "popularity_ratings", "image_url"],
                                         axis=1)  # Drop everything
    _unique_titles = _unique_titles.drop_duplicates("original_title").dropna()
    _unique_titles = _unique_titles.set_index("original_title", drop=True)
    titles_dict = _unique_titles.to_dict("index")

    """
    We can finally build our graph by assembling
    our collection of Read() objects and passing the
    list to our Graph: `Read(user, book1, author1) : `

    """

    read_list = [Read(user_dict[u]["User"], book_dict[b]["Book"], author_dict[a]["Author"], rating=int(r))
               for u, b, a, r in uir[["user_id","book_id","author_id", "rating"]].values]

    BigGraph = Graph(read_list)

    return BigGraph, titles_dict
