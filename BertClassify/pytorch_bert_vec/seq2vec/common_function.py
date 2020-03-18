import re


def split_sentence(sentence):  # 段落分成句子，去掉句子中的空格，去掉句子特别短的句子。
    pattern = r"[。！!？?；;～…◆★]+"
    split_clauses = re.split(pattern, sentence)
    punctuations = re.findall(pattern, sentence)
    punctuations.append('')
    half_out = [''.join(x) for x in zip(split_clauses, punctuations)]
    output = []
    m = r'\s+'
    for item in half_out:
        cleaned_item = re.sub(m, '', item)
        if len(cleaned_item) < 6:
            continue
        else:
            output.append(cleaned_item)
    return output

def load_stopwords(stopwords=None):
    """
    加载停用词
    :param stopwords:str,停用词字典位置。
    :return:
    """
    if stopwords:
        with open(stopwords, 'r', encoding='utf-8') as f:
            return set([line.strip() for line in f])
    else:
        return []


def preprocess_data(body):
    """
    文本预处理，将段落分成句子，对句子进行分词。
    :param body: 新闻正文
    :return:
    """
    sentences = []
    paragraphs = body.split('\n')
    for paragraph in paragraphs:
        sentences_paragraph = split_sentence(paragraph)
        sentences += sentences_paragraph
    return sentences
