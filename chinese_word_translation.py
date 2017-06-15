# coding=utf-8
from file_preprocessor import Reader,XmlBatchReader
from xml.etree import cElementTree as ET
from urllib2 import quote
from urllib import urlencode
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
def collect_words():
    vocab_set = set()
    try:
        i = 0
        while True:
            r = XmlBatchReader(id=i)
            question_xml = r.question_xml[0][0]
            answer_xml = r.answer_xml[0][0]
            for sentence in question_xml:
                for token in sentence[0]:
                    word = token.findtext('word')
                    if word[0] >= u'\u4e00'  and word[0] <= u'\u9FA5':
                        #print word
                        vocab_set.add(word)
            for sentence in answer_xml:
                for token in sentence[0]:
                    word = token.findtext('word')
                    if word[0] >= u'\u4E00' and word[0] <= u'\u9FA5':
                        #print word
                        vocab_set.add(word)
            print 'set size',len(vocab_set)
            i += 10000
    except IOError:
        print "%d pairs total" % i

    f = open('vocab.txt','w')
    for word in vocab_set:
        f.write(word.encode('utf-8')+'\n')
    f.close()

def parse_file(path = 'data/vocab.txt'):
    f = open(path,'r')
    pf = None
    vocabs = f.readlines()
    i = 0
    for word in vocabs:
        if i % 300 == 0:
            if pf:
                pf.close()
            pf = open('data/vocab_parsed/vocab_%d.txt' % i,'w')
        pf.write(word)
        i += 1
    f.close()


# -*- coding: utf-8 -*-

import urllib2
import hashlib
import json
import random

def tras():
    class Baidu_Translation:
        def __init__(self):
            self._q = ''
            self._from = ''
            self._to = ''
            self._appid = 0
            self._key = ''
            self._salt = 0
            self._sign = ''
            self._dst = ''
            self._enable = True

        def GetResult(self):
            self._q.encode('utf-8')
            m = (str(Trans._appid) + Trans._q + str(Trans._salt) + Trans._key).encode('utf-8')
            m_MD5 = hashlib.md5(m)
            Trans._sign = m_MD5.hexdigest()
            Url_1 = 'http://api.fanyi.baidu.com/api/trans/vip/translate?'
            Url_2 = 'q=' + self._q + '&from=' + self._from + '&to=' + self._to + '&appid=' + str(
                Trans._appid) + '&salt=' + str(Trans._salt) + '&sign=' + self._sign

            PostUrl = (Url_1+Url_2).decode()
            TransRequest = urllib2.Request(PostUrl)
            print PostUrl,type(PostUrl)
            #print PostUrl
            TransResponse = urllib2.urlopen(TransRequest)
            TransResult = TransResponse.read()
            data = json.loads(TransResult)
            if 'error_code' in data:
                print 'Crash'
                print 'error:', data['error_code']
                return data['error_msg']
            else:
                self._src = data['trans_result'][0]['src']
                self._dst = data['trans_result'][0]['dst']
                return self._src,self._dst

        def ShowResult(self, result,i):
            f = open('data/vocab_translated/vocab_%d.txt'%i,'w')
            result = result.replace('.', '\n')
            result = result.replace('。', '\n')
            f.write(result)
            f.close()
            print 'Done %d' % i

        def Welcome(self):
            self._q = 'Welcome to use icedaisy online translation tool'
            self._from = 'zh'
            self._to = 'en'
            self._appid = '20170522000048730'
            self._key = 'TOa4k_bZdDVeEzWvqN5o'
            self._salt = random.randint(10001, 99999)

        def StartTrans(self):
            i = 119700
            while True:
                try:
                    f = open('data/vocab_parsed/vocab_%d.txt' % i)
                    f2 = open('data/vocab_translated/vocab_%d.txt' % i,'w')

                    self._q = f.read().replace('\n','！')
                    src,dst = self.GetResult()
                    dst = dst.replace('！', '\n')
                    dst = dst.replace('!', '\n')
                    f2.write(dst)
                    f.close()
                    f2.close()
                    i += 300
                except ValueError:
                    i+=300
                    pass



    Trans = Baidu_Translation()
    Trans.Welcome()
    Trans.StartTrans()

def merge():
    import pickle
    try:
        i = 0
        trans_dict = {}
        while True:
            zh = open('data/vocab_parsed/vocab_%d.txt' % i)
            en = open('data/vocab_translated/vocab_%d.txt' % i)
            zh_line = zh.readlines()
            en_line = en.readlines()
            if len(zh_line) != len(en_line):
                print 'error file ',i
            else:
                for r in range(len(zh_line)):
                    trans_dict[zh_line[r].strip()] = en_line[r].strip()
            i+=300

    except IOError:
        print i
        pass

    pickle_file = open('dictionary.pkl','wb')
    pickle.dump(trans_dict,pickle_file)

merge()