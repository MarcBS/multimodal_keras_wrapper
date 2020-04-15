# -*- coding: utf-8 -*-
import re
import sys

__all__ = [
    'tokenize_basic',
    'tokenize_aggressive',
    'tokenize_icann',
    'tokenize_montreal',
    'tokenize_soft',
    'tokenize_none',
    'tokenize_none_char',
    'tokenize_CNN_sentence',
    'tokenize_questions',
    'detokenize_none',
    'detokenize_bpe',
    'detokenize_none_char',
]


def tokenize_basic(caption, lowercase=True):
    """
    Basic tokenizer for the input/output data of type 'text':
       * Splits punctuation
       * Optional lowercasing

    :param caption: String to tokenize
    :param lowercase: Whether to lowercase the caption or not
    :return: Tokenized version of caption
    """

    punct = [u'.',
             u';',
             u',',
             u'，',
             u"/",
             u'[',
             u']',
             u'"',
             u'{',
             u'}',
             u'(',
             u')',
             u'=',
             u'+',
             u'\\',
             u'_',
             u'-',
             u'>',
             u'<',
             u'@',
             u'`',
             u'¿',
             u'?',
             u'¡',
             u'!']

    def processPunctuation(inText):
        """
        Removes punctuation.
        """
        outText = inText
        for p in punct:
            outText = outText.replace(p, u' ' + p + u' ')
        return outText

    resAns = caption.lower() if lowercase else caption
    resAns = resAns.replace(u'\n', u' ')
    resAns = resAns.replace(u'\t', u' ')
    resAns = processPunctuation(resAns)
    resAns = re.sub(u'[ ]+', u' ', resAns)
    return resAns


def tokenize_aggressive(caption, lowercase=True):
    """
    Aggressive tokenizer for the input/output data of type 'text':
    * Removes punctuation
    * Optional lowercasing

    :param caption: String to tokenize
    :param lowercase: Whether to lowercase the caption or not
    :return: Tokenized version of caption
    """
    punct = [u'.',
             u';',
             u',',
             u'，',
             u"/",
             u'[',
             u']',
             u'"',
             u'{',
             u'}',
             u'(',
             u')',
             u'=',
             u'+',
             u'\\',
             u'_',
             u'-',
             u'>',
             u'<',
             u'@',
             u'`',
             u'¿',
             u'?',
             u'¡',
             u'!']

    def processPunctuation(inText):
        outText = inText
        for p in punct:
            outText = outText.replace(p, '')
        return outText

    resAns = caption.lower() if lowercase else caption
    resAns = processPunctuation(resAns)
    resAns = re.sub('[ ]+', ' ', resAns)
    resAns = resAns.strip()
    return resAns


def tokenize_icann(caption):
    """
    Tokenization used for the icann paper:
    * Removes some punctuation (. , ")
    * Lowercasing

    :param caption: String to tokenize
    :return: Tokenized version of caption
    """
    return tokenize_soft(caption)


def tokenize_montreal(caption):
    """
    Similar to tokenize_icann
        * Removes some punctuation
        * Lowercase

    :param caption: String to tokenize
    :return: Tokenized version of caption
    """
    tokenized = re.sub('[.,"\n\t]+', '', caption.strip())
    tokenized = re.sub('[\']+', " '", tokenized)
    tokenized = re.sub('[ ]+', ' ', tokenized)
    tokenized = map(lambda x: x.lower(), tokenized.split())
    tokenized = " ".join(tokenized)
    return tokenized


def tokenize_soft(caption, lowercase=True):
    """
    Tokenization used for the icann paper:
        * Removes very little punctuation
        * Lowercase
    :param caption: String to tokenize
    :param lowercase: Whether to lowercase the caption or not
    :return: Tokenized version of caption
    """
    tokenized = re.sub(u'[\n\t]+', u'', caption.strip())
    tokenized = re.sub(u'[.]+', u' . ', tokenized)
    tokenized = re.sub(u'[,]+', u' , ', tokenized)
    tokenized = re.sub(u'[!]+', u' ! ', tokenized)
    tokenized = re.sub(u'[?]+', u' ? ', tokenized)
    tokenized = re.sub(u'[{]+', u' { ', tokenized)
    tokenized = re.sub(u'[}]+', u' } ', tokenized)
    tokenized = re.sub(u'[(]+', u' ( ', tokenized)
    tokenized = re.sub(u'[)]+', u' ) ', tokenized)
    tokenized = re.sub(u'[[]+', u' [ ', tokenized)
    tokenized = re.sub(u'[]]+', u' ] ', tokenized)
    tokenized = re.sub(u'["]+', u' " ', tokenized)
    tokenized = re.sub(u'[\']+', u" ' ", tokenized)
    tokenized = re.sub(u'[ ]+', u' ', tokenized)
    # tokenized = u" ".join(tokenized)
    return tokenized if not lowercase else tokenized.lower()


def tokenize_none(caption):
    """
    Does not tokenizes the sentences. Only performs a stripping

    :param caption: String to tokenize
    :return: Tokenized version of caption
    """
    tokenized = re.sub('[\n\t]+', '', caption.strip())
    return tokenized


def tokenize_none_char(caption):
    """
    Character-level tokenization. Respects all symbols. Separates chars. Inserts <space> sybmol for spaces.
    If found an escaped char, "&apos;" symbol, it is converted to the original one
    # List of escaped chars (by moses tokenizer)
    & ->  &amp;
    | ->  &#124;
    < ->  &lt;
    > ->  &gt;
    ' ->  &apos;
    " ->  &quot;
    [ ->  &#91;
    ] ->  &#93;
    :param caption: String to tokenize
    :return: Tokenized version of caption
    """

    def convert_chars(x):
        """
        Convert spaces to the '<space>' token.
        """
        if x == ' ':
            return u'<space>'
        else:
            return x

    tokenized = re.sub(u'[\n\t]+', u'', caption.strip())
    tokenized = re.sub(u'[ ]+', u' ', tokenized)
    tokenized = [convert_chars(char) for char in tokenized]
    tokenized = u" ".join(tokenized)
    return tokenized


def tokenize_CNN_sentence(caption):
    """
    Tokenization employed in the CNN_sentence package
    (https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py#L97).
    :param caption: String to tokenize
    :return: Tokenized version of caption
    """
    tokenized = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", caption)
    tokenized = re.sub(r"\'s", " \'s", tokenized)
    tokenized = re.sub(r"\'ve", " \'ve", tokenized)
    tokenized = re.sub(r"n\'t", " n\'t", tokenized)
    tokenized = re.sub(r"\'re", " \'re", tokenized)
    tokenized = re.sub(r"\'d", " \'d", tokenized)
    tokenized = re.sub(r"\'ll", " \'ll", tokenized)
    tokenized = re.sub(r",", " , ", tokenized)
    tokenized = re.sub(r"!", " ! ", tokenized)
    tokenized = re.sub(r"\(", " \( ", tokenized)
    tokenized = re.sub(r"\)", " \) ", tokenized)
    tokenized = re.sub(r"\?", " \? ", tokenized)
    tokenized = re.sub(r"\s{2,}", " ", tokenized)
    return tokenized.strip().lower()


def tokenize_questions(caption):
    """
    Basic tokenizer for VQA questions:
        * Lowercasing
        * Splits contractions
        * Removes punctuation
        * Numbers to digits

    :param caption: String to tokenize
    :return: Tokenized version of caption
    """
    contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
                    "couldnt": "couldn't",
                    "couldn'tve": "couldn’t’ve", "couldnt’ve": "couldn’t’ve", "didnt": "didn’t",
                    "doesnt": "doesn’t",
                    "dont": "don’t", "hadnt": "hadn’t", "hadnt’ve": "hadn’t’ve", "hadn'tve": "hadn’t’ve",
                    "hasnt": "hasn’t", "havent": "haven’t", "hed": "he’d", "hed’ve": "he’d’ve", "he’dve": "he’d’ve",
                    "hes": "he’s", "howd": "how’d", "howll": "how’ll", "hows": "how’s", "Id’ve": "I’d’ve",
                    "I’dve": "I’d’ve", "Im": "I’m", "Ive": "I’ve", "isnt": "isn’t", "itd": "it’d",
                    "itd’ve": "it’d’ve",
                    "it’dve": "it’d’ve", "itll": "it’ll", "let’s": "let’s", "maam": "ma’am", "mightnt": "mightn’t",
                    "mightnt’ve": "mightn’t’ve", "mightn’tve": "mightn’t’ve", "mightve": "might’ve",
                    "mustnt": "mustn’t",
                    "mustve": "must’ve", "neednt": "needn’t", "notve": "not’ve", "oclock": "o’clock",
                    "oughtnt": "oughtn’t",
                    "ow’s’at": "’ow’s’at", "’ows’at": "’ow’s’at", "’ow’sat": "’ow’s’at", "shant": "shan’t",
                    "shed’ve": "she’d’ve", "she’dve": "she’d’ve", "she’s": "she’s", "shouldve": "should’ve",
                    "shouldnt": "shouldn’t", "shouldnt’ve": "shouldn’t’ve", "shouldn’tve": "shouldn’t’ve",
                    "somebody’d": "somebodyd", "somebodyd’ve": "somebody’d’ve", "somebody’dve": "somebody’d’ve",
                    "somebodyll": "somebody’ll", "somebodys": "somebody’s", "someoned": "someone’d",
                    "someoned’ve": "someone’d’ve", "someone’dve": "someone’d’ve", "someonell": "someone’ll",
                    "someones": "someone’s", "somethingd": "something’d", "somethingd’ve": "something’d’ve",
                    "something’dve": "something’d’ve", "somethingll": "something’ll", "thats": "that’s",
                    "thered": "there’d", "thered’ve": "there’d’ve", "there’dve": "there’d’ve",
                    "therere": "there’re",
                    "theres": "there’s", "theyd": "they’d", "theyd’ve": "they’d’ve", "they’dve": "they’d’ve",
                    "theyll": "they’ll", "theyre": "they’re", "theyve": "they’ve", "twas": "’twas",
                    "wasnt": "wasn’t",
                    "wed’ve": "we’d’ve", "we’dve": "we’d’ve", "weve": "we've", "werent": "weren’t",
                    "whatll": "what’ll",
                    "whatre": "what’re", "whats": "what’s", "whatve": "what’ve", "whens": "when’s", "whered":
                        "where’d", "wheres": "where's", "whereve": "where’ve", "whod": "who’d",
                    "whod’ve": "who’d’ve",
                    "who’dve": "who’d’ve", "wholl": "who’ll", "whos": "who’s", "whove": "who've", "whyll": "why’ll",
                    "whyre": "why’re", "whys": "why’s", "wont": "won’t", "wouldve": "would’ve",
                    "wouldnt": "wouldn’t",
                    "wouldnt’ve": "wouldn’t’ve", "wouldn’tve": "wouldn’t’ve", "yall": "y’all",
                    "yall’ll": "y’all’ll",
                    "y’allll": "y’all’ll", "yall’d’ve": "y’all’d’ve", "y’alld’ve": "y’all’d’ve",
                    "y’all’dve": "y’all’d’ve",
                    "youd": "you’d", "youd’ve": "you’d’ve", "you’dve": "you’d’ve", "youll": "you’ll",
                    "youre": "you’re", "youve": "you’ve"}
    punct = [';', r"/", '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\',
             '_', '-', '>', '<', '@', '`', ',', '?', '!']
    commaStrip = re.compile("(\d)(,)(\d)")
    periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
    manualMap = {'none': '0', 'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
                 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'}
    articles = ['a', 'an', 'the']

    def processPunctuation(inText):
        """
        Process (remove) the punctuation.
        """
        if isinstance(inText, str) and sys.version_info < (3, 0):
            inText = inText.decode("utf-8").encode("utf-8")
        outText = inText
        for p in punct:
            if (p + ' ' in inText or ' ' + p in inText) or (re.search(commaStrip, inText) is not None):
                outText = outText.replace(p, '')
            else:
                outText = outText.replace(p, ' ')
        outText = periodStrip.sub("", outText, re.UNICODE)
        return outText

    def processDigitArticle(inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = manualMap.setdefault(word, word)
            if word not in articles:
                outText.append(word)
            else:
                pass
        for wordId, word in list(enumerate(outText)):
            if word in contractions:
                outText[wordId] = contractions[word]
        outText = ' '.join(outText)
        return outText

    resAns = caption.lower()
    resAns = resAns.replace('\n', ' ')
    resAns = resAns.replace('\t', ' ')
    resAns = resAns.strip()
    resAns = processPunctuation(resAns)
    resAns = processDigitArticle(resAns)

    return resAns


def detokenize_none(caption):
    """
    Dummy function: Keeps the caption as it is.
    :param caption: String to de-tokenize.
    :return: Same caption.
    """
    if isinstance(caption, str) and sys.version_info < (3, 0):
        caption = caption.decode('utf-8')
    return caption


def detokenize_bpe(caption, separator=u'@@'):
    """
    Reverts BPE segmentation (https://github.com/rsennrich/subword-nmt)
    :param caption: Caption to detokenize.
    :param separator: BPE separator.
    :return: Detokenized version of caption.
    """
    if isinstance(caption, str) and sys.version_info < (3, 0):
        caption = caption.decode('utf-8')
    bpe_detokenization = re.compile(u'(' + separator + u' )|(' + separator + u' ?$)')
    detokenized = bpe_detokenization.sub(u'', caption).strip()
    return detokenized


def detokenize_none_char(caption):
    """
    Character-level detokenization. Respects all symbols. Joins chars into words. Words are delimited by
    the <space> token. If found an special character is converted to the escaped char.
    # List of escaped chars (by moses tokenizer)
        & ->  &amp;
        | ->  &#124;
        < ->  &lt;
        > ->  &gt;
        ' ->  &apos;
        " ->  &quot;
        [ ->  &#91;
        ] ->  &#93;
    :param caption: String to de-tokenize.
        :return: Detokenized version of caption.
    """

    detokenized = re.sub(u' ', u'', caption)
    detokenized = re.sub(u'<space>', u' ', detokenized)
    return detokenized
