Comparison of Three Recommender Systems on Goodreads Dataset
-------------------------------------------------------------
Jay Shi, William Su

Here's a guide to help you navigate through our files:

1. To understand all of our data, models, and implementation go to:
    1. FinalReport.pdf

2. For Data and Model Visualization, go to EDA&ModelViz.ipynb There we perform a number of tasks including:
    1. Visualizing the distrubtion of the average ratings
    2. Comparing the three matrices (shelved, isRead, and rating)
    2. Visualizing the three models (user-user, matrix factorization, neural network matrix factorization)
    through different plots
    3. Comparing all the models' MSE performance together

3. Data Cleaning, go to：
    1. Folder: ExtractingMatrices
        - json_to_csv_converter.py is a script we found from:
          https://github.com/Yelp/dataset-examples/blob/master/json_to_csv_converter.py
          it converts the raw data goodreads_interactions_fantasy_paranormal.json.gz to goodreads_interactions_fantasy_paranormal.csv file
        - extractThreeMatrices.ipynb notebook will show you how we
          converted a goodreads_interactions_fantasy_paranormal.csv file data
          into three sparse matrices:
              1. rating_matrix_fantasy.npz
              2. isRead_matrix_fantasy.npz
              3. shelved_matrix_fantasy.npz
        - BOOK_ID_TO_INT_fantasy.json and USER_ID_TO_INT_fantasy.json are json files that we saved
          from the extractThreeMatrices.ipynb notebook
              - in the notebook, we mapped the original id's (strings) to a specific range of int's that we
                use. this json file helps us map these int's back to the original id's
    2. Folder: ShrinkMatrices
        - the shrinkThreeMatricies.ipynb notebook will show you how
          we shrunk our original 3 sparse matrices
              1. rating_matrix_shrunk.npz
              2. isRead_matrix_shrunk.npz
              3. shelved_matrix_shrunk.npz
                - we decided to shrink the matrices because they were too large
                  and took too much computing power to run

4. UserUser model, go to
    1. Folder: UserUserModel
        - sweepUserUSer.ipynb notebook will show you how we build our weighted
          user-user model and also how we swept through the parameters w1 and w2:
              1. user_user_sweep1.json  user_user_sweep7.json contains json of key (w1, w2)
                 mapped to test_data MSE's

5. For MatrixFactorization model go to:
    1. Folder: MatrixFactorizationModel
        - sweep_params_matrix_fact.ipynb notebook will show you how we built the
          matrix factorizaton model based on a UV decomposition and alternating least squares
          and show you how we swept through the parameters K and reg
            1. matrix_factorization_all_params.json contains a json of key (K, reg) mapped to
              train_data MSE's and test_data MSE's
            2. best_param_50_niter.npy contains a numpy array that contains test_data MSE from iteration 1
              to iteration 50 using the best tuned K, reg pair we found from matrix_factorization_all_params.json

6.  To understand how we built our neural network-based matrix factorization model go to
    1. Folder: NeuralNetworkModel
        - nn_factorization.ipynb notebook will walk you through how the model is built
        - nn_history is the saved data of a model that we built
        - nn_model is the saved model that we built
            - we use this in our EDA&ModelViz notebook to visualize the model and data
              as it takes too long to retrain a model
        - nn_sweep.json is a json mapping latent spaces k that we swept to the test MSE
