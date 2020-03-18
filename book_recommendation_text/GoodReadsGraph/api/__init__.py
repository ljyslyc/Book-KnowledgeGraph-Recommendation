import logging as logger
from flask import Flask

"""
To build this Flask app, navigate to the folder
above the `api/` directory:
    $ export FLASK_APP=api
    $ flask run
This should start the server running on
localhost:5000
"""

def create_app():
    app = Flask(__name__)

    from .app import main
    app.register_blueprint(main)

    logger.debug("App registered")
    return app
