# coding=utf-8
import csv
import os
import xml.etree.cElementTree as ET
from sematic_tree import Sematic_tree

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
        question_file,answer_file = None,None
        while i < len(self.raw_questions):
            if i % 10000 == 0:
                if question_file and answer_file:
                    question_file.close()
                    answer_file.close()
                question_file = open('data/package/question_raw_train_%d.txt' % i,'w')
                answer_file = open('data/package/answer_raw_train_%d.txt' % i,'w')
            question_file.write(self.raw_questions[i]+'\n')
            answer_file.write(self.raw_answers[i]+'\n')
            i += 1
        question_file.close()
        answer_file.close()
    def make_dict(self):
        self.aq_dict=dict(zip((_ for _ in self.raw_answers),(_ for _ in self.raw_questions)))
        self.al_dict=dict(zip((_ for _ in self.raw_answers),(_ for _ in self.labels)))

#处理已经标注好的数据
class XmlBatchReader:
    def __init__(self,id):
        self.offset_aq_dict = {}
        self.offset_al_dict = {}
        self.read_xml_batch(id)
    def read_xml_batch(self,id):
        raw_answer = open('data/package/answer_raw_train_%d.txt' % id)
        raw_question = open('data/package/question_raw_train_%d.txt' % id)
        raw = open('data/BoP2017-DBQA.train.txt')
        try:
            offset_ans = raw_answer.tell()
            ans = raw_answer.readline()
            offset_ques = raw_question.tell()
            ques = raw_question.readline()
            label = raw.readline()[0]
            self.offset_aq_dict[offset_ans] = offset_ques
            self.offset_al_dict[offset_ans] = int(label)
        except EOFError:
            print 'len:',len(self.offset_aq_dict)
            pass
        self.answer_xml = ET.parse('data/parsed/answer_raw_train_%d.txt.xml' % id).getroot()
        print type(self.answer_xml),type(self.answer_xml[0])
        self.question_xml = ET.parse('data/parsed/question_raw_train_%d.txt.xml' % id).getroot()
        print self.answer_xml
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