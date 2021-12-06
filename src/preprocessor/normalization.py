# https://github.com/neologd/mecab-ipadic-neologd/wiki/Regexp.ja から引用・一部改変
from __future__ import unicode_literals
import re
import unicodedata

def unicode_normalize(cls, s):
    pt = re.compile('([{}]+)'.format(cls))

    def norm(c):
        return unicodedata.normalize('NFKC', c) if pt.match(c) else c

    s = ''.join(norm(x) for x in re.split(pt, s))
    s = re.sub('－', '-', s)
    return s

def remove_extra_spaces(s):
    s = re.sub('[ 　]+', ' ', s)
    blocks = ''.join(('\u4E00-\u9FFF',  # CJK UNIFIED IDEOGRAPHS
                      '\u3040-\u309F',  # HIRAGANA
                      '\u30A0-\u30FF',  # KATAKANA
                      '\u3000-\u303F',  # CJK SYMBOLS AND PUNCTUATION
                      '\uFF00-\uFFEF'   # HALFWIDTH AND FULLWIDTH FORMS
                      ))
    basic_latin = '\u0000-\u007F'

    def remove_space_between(cls1, cls2, s):
        p = re.compile('([{}]) ([{}])'.format(cls1, cls2))
        while p.search(s):
            s = p.sub(r'\1\2', s)
        return s

    s = remove_space_between(blocks, blocks, s)
    s = remove_space_between(blocks, basic_latin, s)
    s = remove_space_between(basic_latin, blocks, s)
    return s

def normalize_neologd(s):
    s = normalize_pdf(s)
    s = s.strip()
    

    s = unicode_normalize('０-９Ａ-Ｚａ-ｚ｡-ﾟ', s)

    def maketrans(f, t):
        return {ord(x): ord(y) for x, y in zip(f, t)}

    s = re.sub('[˗֊‐‑‒–⁃⁻₋−]+', '-', s)  # normalize hyphens
    s = re.sub('[﹣－ｰ—―─━ー]+', 'ー', s)  # normalize choonpus
    s = re.sub('[~∼∾〜〰～]+', '〜', s)  # normalize tildes (modified by Isao Sonobe)
    s = s.translate(
        maketrans('!"#$%&\'()*+,-./:;<=>?@[¥]^_`{|}~｡､･｢｣',
              '！”＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［￥］＾＿｀｛｜｝〜。、・「」'))

    s = remove_extra_spaces(s)
    s = unicode_normalize('！”＃＄％＆’（）＊＋，－．／：；＜＞？＠［￥］＾＿｀｛｜｝〜', s)  # keep ＝,・,「,」
    s = re.sub('[’]', '\'', s)
    s = re.sub('[”]', '"', s)
    return s


def remove_extra_dot(s):
    while len(re.findall('\.\.\.', s))>0:
        s = s.replace('...', '.')
    while len(re.findall('\。\。', s))>0:
        s = s.replace('。。', '。')
    return s


def normalize_pdf(s):
    s = s.replace('\uf0e1\uf0e2', '​')
    s = s.replace('\uf0e2', '​')
    s = s.replace('\uf0c6', '→')
    s = s.replace('\uf0ae', '→')
    s = s.replace('⇒', '→')
    s = s.replace('\uf0b7', '·')
    s = s.replace('\uf06c', '·')
    s = s.replace('\uf02a', '·')
    s = s.replace('●', '·')
    s = s.replace('☆', '·')
    s = s.replace('＊', '·')
    s = s.replace('\uf074', '◀')
    s = s.replace('\uf075', '▶')
    s = s.replace('\uf020', '​')
    s = s.replace('\uf0be', '-​')
    s = s.replace('\uf0e3', '​')
    s = s.replace('\uf0e4', '​')
    s = s.replace('\uf0bd', '|​')
    s = s.replace('\u3000​', ' ')
    s = s.replace('®', '​')
    s = s.replace('™​', '​')
    s = s.replace('©', '​')
    s = s.replace('△', '​◯')
    s = s.replace('○', '​◯')
    s = s.replace('【', '​[')
    s = s.replace('】', '​]')
    s = s.replace('「', '​[')
    s = s.replace('」', '​]')
    s = s.replace('※', '​·')
    s = s.replace('・', '​·')
    s = s.replace('〓', '​[一時停止]')
    s = remove_extra_dot(s)
    return s

def remove_brackets(text):
    text = re.sub(r"(^【[^】]*】)|(【[^】]*】$)", "", text)
    return text
    
def normalize_text(text):
    assert "\n" not in text and "\r" not in text
    text = text.replace("\t", " ")
    text = text.strip()
    text = normalize_neologd(text)
    text = text.lower()
    return text