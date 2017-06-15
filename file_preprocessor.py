# coding=utf-8
import csv
import os
import xml.etree.cElementTree as ET
from sematic_tree import Sematic_tree
from collections import OrderedDict

# Author: AuCson, 0515
# Purpose:

def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False

def is_number(uchar):
    """判断一个unicode是否是数字"""
    if uchar >= u'\u0030' and uchar <= u'\u0039':
        return True
    else:
        return False

def is_alphabet(uchar):
    """判断一个unicode是否是英文字母"""
    if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
        return True
    else:
        return False

def is_punct(uchar):
    if uchar in [u' ',u',',u'.',u'?',u'"',u':',u';',u'!',u'"',u'(',u')',u'[',u']',u'{'
                 u'}',u'<',u'>',u'+',u'-',u'=',u'/',u'%',u'。',u'，',u'？',u'！',
                 u'《',u'》',u'：',u'；',u'“',u'”',u'）',u'（',u'【',u'】',u'#',u'——',
                 u'_']:
        return True
    else:
        return False

def is_legal(uchar):
    """判断是否非汉字，数字和英文字符"""
    if (is_chinese(uchar) or is_number(uchar) or is_alphabet(uchar) or is_punct(uchar)):
        return True
    else:
        return False

class Reader:
    def __init__(self,txt_path,test=False):
        self.labels = []
        self.raw_questions = []
        self.raw_answers = []
        self.test = test
        self.read_file(txt_path)
        self.aq_dict = {}
        self.al_dict = {}
        self.aq_offset_dict = {}
        self.al_offset_dict = {}
    def read_file(self,txt_path):
        with open(txt_path,'r') as f:
            reader = csv.reader(f,delimiter='\t',quoting=csv.QUOTE_NONE)
            if not self.test:
                for row in reader:
                    self.labels.append(row[0])
                    self.raw_questions.append(row[1])
                    self.raw_answers.append(row[2])
            else:
                for row in reader:
                    self.raw_questions.append(row[0])
                    self.raw_answers.append(row[1])
        #self.make_dict()
    def to_seperate_file(self):
        i = 0
        question_file,answer_file,label_file = None,None,None
        while i < len(self.raw_questions):
            if i % 10000 == 0:
                if question_file and answer_file and label_file:
                    question_file.close()
                    answer_file.close()
                    if not self.test:
                        label_file.close()
                if not self.test:
                    question_file = open('data/package/question_raw_train_%d.txt' % i,'w')
                    answer_file = open('data/package/answer_raw_train_%d.txt' % i,'w')
                    label_file = open('data/package/label_raw_train_%d.txt' % i, 'w')
                else:
                    question_file = open('data/package_test/question_raw_test_%d.txt' % i,'w')
                    answer_file = open('data/package_test/answer_raw_test_%d.txt' % i, 'w')
            #print type(self.raw_questions[i])
            if not self.raw_answers[i].endswith('。'):
                self.raw_answers[i] += '。'

            u = self.raw_answers[i].decode('utf-8')
            s = []
            k = 0
            while k < len(u):
                uchar = u[k]
                if is_legal(uchar):
                    s.append(uchar.encode('utf-8'))
                k+=1
            self.raw_answers[i] = ''.join(s)


            u = self.raw_questions[i].decode('utf-8')
            s = []
            k = 0
            while k < len(u):
                uchar = u[k]
                if is_legal(uchar):
                    s.append(uchar.encode('utf-8'))
                k+=1
            self.raw_questions[i] = ''.join(s)

            question_file.write(self.raw_questions[i]+'\n')
            answer_file.write(self.raw_answers[i]+'\n')
            if not self.test:
                label_file.write(self.labels[i] + '\n')
            i += 1
        question_file.close()
        answer_file.close()
        if not self.test:
            label_file.close()
    def make_dict(self):
        self.aq_dict=dict(zip((_ for _ in self.raw_answers),(_ for _ in self.raw_questions)))
        if not self.test:
            self.al_dict=dict(zip((_ for _ in self.raw_answers),(_ for _ in self.labels)))

#处理已经标注好的数据
class XmlBatchReader:
    def __init__(self,id,test=False):
        self.offset_aq_dict = OrderedDict()
        self.offset_al_dict = OrderedDict()
        self.test = test
        self.read_xml_batch(id)

    def read_xml_batch(self,id):
        if not self.test:
            raw_answer = open('data/package/answer_raw_train_%d.txt' % id)
            raw_question = open('data/package/question_raw_train_%d.txt' % id)
            raw_label = open('data/package/label_raw_train_%d.txt' % id)
        else:
            raw_answer = open('data/package_test/answer_raw_test_%d.txt' % id)
            raw_question = open('data/package_test/question_raw_test_%d.txt' % id)
        i = 0
        char_offset_ans,char_offset_ques = 0,0
        prev_ques = None
        prev_ques_offset = 0
        while True:
            ans = raw_answer.readline()
            if not ans:
                break
            ques = raw_question.readline()
            self.offset_aq_dict[char_offset_ans] = char_offset_ques if prev_ques!=ques else prev_ques_offset
            if not self.test:
                self.offset_al_dict[char_offset_ans] = int(raw_label.readline()[-2])
            #print [self.offset_al_dict[char_offset_ans]],type(self.offset_al_dict[char_offset_ans])
            #raw_input()
            if prev_ques!=ques:
                prev_ques_offset = char_offset_ques
                prev_ques = ques
            char_offset_ans += len(ans.decode('utf-8'))
            char_offset_ques += len(ques.decode('utf-8'))
            i+=1
        print 'len:',len(self.offset_aq_dict)
        #print self.offset_aq_dict
        if not self.test:
            self.answer_xml = ET.parse('data/parsed/answer_raw_train_%d.txt.xml' % id).getroot()
            self.question_xml = ET.parse('data/parsed/question_raw_train_%d.txt.xml' % id).getroot()
            self.answer_cnt = self.count_sentence(self.answer_xml)
        else:
            self.answer_xml = ET.parse('data/parsed_test/answer_raw_test_%d.txt.xml' % id).getroot()
            self.question_xml = ET.parse('data/parsed_test/question_raw_test_%d.txt.xml' % id).getroot()
            self.answer_cnt = self.count_sentence(self.answer_xml)
    def count_sentence(self,xml):
        cnt = 0
        for _ in xml[0][0]:
            cnt += 1
        return cnt


def preview(path):
    pf = open('preview.txt.xml','w')
    f = open(path,'r')
    for _ in range(2000):
        s = f.readline()
        pf.write(s+'\n')