import logging
import os 
import pandas as pd
import time

logging.basicConfig(
    format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt = '%m/%d/%Y %H:%M:%S',
    level = logging.INFO
)

logger = logging.getLogger(__name__)


def log_training(num_examples, train_batch_size, num_train_optimization_steps):
    """ logs the beginning of training """
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", num_examples)
    logger.info("  Batch size = %d", train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

def log_evaluating(num_eval_examples, eval_batch_size):
    """ Logs the beginning of the evaluation. """ 
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", num_eval_examples)
    logger.info("  Batch size = %d", eval_batch_size)

def log_result(output_dir, num_train_examples, num_test_examples, result):
    """ logs and writes the evalutated stats of the model """
    current_time = time.strftime("%m-%d_%I-%M%p")
    output_eval_file = os.path.join(output_dir,"eval_results", "result-{}.txt".format(current_time))
    with open(output_eval_file, "w") as writer:
        writer.write("train_examples = %d\n" % (num_train_examples,))
        writer.write("test_examples = %d\n\n" % (num_test_examples,))
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

def record_to_csv(output_dir, num_train_examples, num_test_examples, result):
    """ creates (if necessary) and stores results in a 'combined.csv' file """
    path = os.path.join(output_dir, "combined.csv")
    result["num_train_examples"] = num_train_examples
    result["num_test_examples"] = num_test_examples
    results = pd.Series(result)
    past_results = pd.DataFrame()
    if os.path.exists(path):
        past_results = pd.read_csv(path)
    results = past_results.append(results, ignore_index=True)
    results.to_csv(path, index=False)    
