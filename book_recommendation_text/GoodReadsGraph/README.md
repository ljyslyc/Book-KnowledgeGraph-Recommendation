# GoodReadsGraph
## Build a Graph Database of Good Reads book data to recommend novel novels to users:
### In the screenshot below, I entered _Neuromancer_ and got Ann Leckie's _Ancillary Mercy_. 
![](https://github.com/franckjay/GoodReadsGraph/blob/master/ReactApp/src/ReactApp.png)


### To run back-end:
* Activate your virtual environment in a python terminal
* Start flask server:
* `$ export FLASK_APP=api` 
* `$ flask run`
* Should be running on `localhost:5000`. You can test your API with `Postman`

### To run front-end (first-time only) in a new terminal:
No idea if there is a better way to do this, but:
* Follow Node installation instructions [here]("https://www.youtube.com/watch?v=06pWsB_hoD4&t=233s")
* Build a new React app folder (_do not call it_ `ReactApp/`):
* `$ npx create-react-app {FOO}`
* `$ npm start`
* This will start the app with all of the basic components needed
* The React App should show up on localhost:3000
* Copy+Replace all of the files in `ReactApp/` to your `{FOO}/` directory (e.g., `App.js`, `components/`)

### If you already did the first-time front-end build:
* `$ npm start`
* Anytime you edit/save any files inside the directory, the App on the local host will update
* Need to debug? Right click on your apps web page: `Inspect -> Console Tab` will show you what is up


TODO: Add a conda environment `.yml` file for the Flask dependencies
