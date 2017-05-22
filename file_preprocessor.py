# coding=utf-8
import csv
import os
import xml.etree.cElementTree as ET
from sematic_tree import Sematic_tree
from collections import OrderedDict

# Author: AuCson, 0515
# Purpose:

class Reader:
    def __init__(self,txt_path):
        self.labels = []
        self.raw_questions = []
        self.raw_answers = []
        self.read_file(txt_path)
        self.aq_dict = {}
        self.al_dict = {}
        self.aq_offset_dict = {}
        self.al_offset_dict = {}
    def read_file(self,txt_path):
        with open(txt_path,'r') as f:
            reader = csv.reader(f,delimiter='\t',quoting=csv.QUOTE_NONE)
            for row in reader:
                self.labels.append(row[0])
                self.raw_questions.append(row[1])
                self.raw_answers.append(row[2])
        #self.make_dict()
    def to_seperate_file(self):
        i = 0
        question_file,answer_file,label_file = None,None,None
        while i < len(self.raw_questions):
            if i % 10000 == 0:
                if question_file and answer_file and label_file:
                    question_file.close()
                    answer_file.close()
                    label_file.close()
                question_file = open('data/package/question_raw_train_%d.txt' % i,'w')
                answer_file = open('data/package/answer_raw_train_%d.txt' % i,'w')
                label_file = open('data/package/label_raw_train_%d.txt' % i, 'w')
            question_file.write(self.raw_questions[i]+'\n')
            answer_file.write(self.raw_answers[i]+'\n')
            label_file.write(self.labels[i] + '\n')
            i += 1
        question_file.close()
        answer_file.close()
        label_file.close()
    def make_dict(self):
        self.aq_dict=dict(zip((_ for _ in self.raw_answers),(_ for _ in self.raw_questions)))
        self.al_dict=dict(zip((_ for _ in self.raw_answers),(_ for _ in self.labels)))

#处理已经标注好的数据
class XmlBatchReader:
    def __init__(self,id):
        self.offset_aq_dict = OrderedDict()
        self.offset_al_dict = OrderedDict()
        self.read_xml_batch(id)
    def read_xml_batch(self,id):
        raw_answer = open('data/package/answer_raw_train_%d.txt' % id)
        raw_question = open('data/package/question_raw_train_%d.txt' % id)
        raw_label = open('data/package/label_raw_train_%d.txt' % id)
        i = 0
        char_offset_ans,char_offset_ques = 0,0
        while True:
            ans = raw_answer.readline()
            if not ans:
                break
            ques = raw_question.readline()
            self.offset_aq_dict[char_offset_ans] = char_offset_ques
            self.offset_al_dict[char_offset_ans] = raw_label.readline()
            char_offset_ans += len(ans.decode('utf-8'))
            char_offset_ques += len(ques.decode('utf-8'))
            i += 1
        print 'i',i
        print 'len:',len(self.offset_aq_dict)
        #print self.offset_aq_dict
        print self.offset_aq_dict[600]
        print sorted(self.offset_aq_dict.keys()[0:100])
        self.answer_xml = ET.parse('data/parsed/answer_raw_train_%d.txt.xml' % id).getroot()
        self.question_xml = ET.parse('data/parsed/question_raw_train_%d.txt.xml' % id).getroot()
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