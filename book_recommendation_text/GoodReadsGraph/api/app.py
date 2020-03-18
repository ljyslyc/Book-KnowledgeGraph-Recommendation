import logging as logger
from flask import Blueprint, jsonify, request
from .GoodReadsGraph import BuildGraph

main = Blueprint('main', __name__)
logger.debug("App starting")

BigGraph, titles_dict = BuildGraph()
logger.debug("Graph is built service")

output_URL = ""

def large_url(url):
    """
    The URL path has an "s,m,l" in it, and
    we want the large url image
    """
    _split=url.split("/")
    _split[-2]=_split[-2][:-1]+"l"

    large_url = ""
    for segment in _split:
        large_url += segment +"/"

    return large_url[:-1]


@main.route('/input_book', methods=['POST'])
def input_book():
    # We get a request from our React App
    logger.debug("Starting service")
    _book = request.get_json()
    # We extract the title from our JSON
    _book_title = str(_book["title"])
    logger.debug("book: ", _book_title)
    # We use our dictionary to pull out a book Object
    try:
        _book_object = titles_dict[_book_title.lower()]["Book"]
        logger.debug(_book_object)
        # Grab the first entry, return it
        book_list = BigGraph._book2book(_book_object, N=1)
        book = book_list[0]
        logger.debug(book)

        # Set the global url to be the large version of the input url
        global ouput_URL
        ouput_URL = large_url(str(book.image_url))
        # Return the image URL to be displayed on our app
        #    NB: POST does not support anything else.
        return "Done", 201
    except Exception as e:
        logger.warning("Encountered exception: ", e)
        return "Error", 400

@main.route('/novel_novel', methods=['GET'])
def novel_novel():
    logger.debug("GET method returning output_url")
    return jsonify({"image_url": ouput_URL}), 200
