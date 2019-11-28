import re
import numpy
import  nltk
# nltk.download('stopwords')
import nltk.stem
from nltk.corpus import stopwords
def get_TwoWords():
    # M = dict()
    WORDS = dict()
    Two_Word = dict()
    Possibly = dict ()
    str = "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+"
    # str = "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~#￥%……&*（）]+"

    with open('train_txt.csv', 'r') as f:
        txt = f.read ().splitlines ()
        for line in txt:
            temp = line.split(',')
            words = temp[2].split(' ')
            # M[temp[1]] = re.sub(str, "", words)
            i = 0
            first = re.sub(str, "", words[0])

            # 前一个单词
            for word in words:
                word = re.sub (str, "", word)
                s = nltk.stem.SnowballStemmer('english')
                word = s.stem(word)
                if word not in WORDS.keys ():
                    WORDS[word] = 1
                else:
                    WORDS[word] += 1
                if (i == 0):
                    i += 1
                    continue
                if first + " " + word not in Two_Word.keys ():
                    Two_Word[first + " " + word] = 1
                else:
                    Two_Word[first + " " + word] += 1
                first = word

        # for item in Two_Word.items():
        #     first = item[0].split(" ")
        #     Possibly[item[0]] = item[1]/(WORDS[first[0]])
        WORDS = dict(sorted (WORDS.items (), key=lambda d: d[1], reverse=True))
        # Possibly = dict(sorted (Possibly.items (), key=lambda d: d[1], reverse=True))
        Two_Word = dict (sorted (Two_Word.items (), key=lambda d: d[1], reverse=True))
        i = 1
        exc = {' '}
        # for item in Two_Word.items ():
        #     if item in exc:
        #         continue
        #     print (item)
        #     if (i > 100):
        #         break
        #     i += 1
        # for item in WORDS.items ():
        #     if item[0] in stopwords.words('english'):
        #         continue
        #     print(item)
        #     if (i > 100):
        #         break
        #     i += 1
    return WORDS, Two_Word




