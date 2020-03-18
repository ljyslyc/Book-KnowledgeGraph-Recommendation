import sys

import numpy as np

sys.path.append('../')

from seq2vec.common_function import preprocess_data
from seq2vec.bert_vec import BertSeqVec, AlbertTextNet, BertTextNet
from scipy.spatial.distance import cosine
from scipy import stats
import csv


def get_body_sim_ts(text_net, body, test=False):
    bert_seq_vec = BertSeqVec(text_net)
    lines = preprocess_data(body)
    last_vec = None
    sims = []
    last_line = None
    for line in lines:
        if len(line) > 200:
            line = line[:200]
        vec = bert_seq_vec.seq2vec(line)
        if last_vec is None:
            last_vec = vec
            last_line = line
        else:
            sim = cosine(vec, last_vec)
            sims.append(sim)
            last_vec = vec
            if False:
                print(last_line)
                print(line)
                print(sim)
            last_line = line
    sims = np.array(sims)
    if test:
        print(sims)
    avgs = np.mean(sims)
    return sims, avgs


def get_fft(ts_data):
    fr = np.fft.fft(ts_data)
    fr = np.abs(fr) / len(ts_data)
    print(max(fr[1:]))


if __name__ == '__main__':
    text_net = BertTextNet()
    with open('/home/hetao/Documents/text_style/from_db_uncut.csv', 'r') as inf, \
            open('/home/hetao/Documents/text_style/is_daodu/from_db_bert.csv', 'w') as outf:
        reader = csv.DictReader(inf)
        writer = csv.DictWriter(outf, list(reader.fieldnames) + ['avg', 'median', 'skew', 'top_mean'])
        writer.writeheader()
        for row in reader:
            one_sim_ts, avg = get_body_sim_ts(text_net, row['body'])
            # plt.hist(one_sim_ts, bins=len(one_sim_ts)//3)
            # plt.show()
            median = np.median(one_sim_ts)
            skew = stats.skew(one_sim_ts)
            sorted_ts = np.sort(one_sim_ts)
            top_percent = sorted_ts[-len(sorted_ts) // 2:-1]
            top_mean = np.mean(top_percent)
            row['avg'] = avg
            row['median'] = median
            row['skew'] = skew
            row['top_mean'] = top_mean
            writer.writerow(row)
