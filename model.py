# coding=utf-8
from file_preprocessor import XmlBatchReader, Reader
from xml.etree import cElementTree as ET
import cow
import numpy as np
import re
from sematic_tree import Sematic_tree
debug1 = []
debug2 = []


class SentenceStructureModel:
    def __init__(self, path='data/BoP2017-DBQA.train.txt'):
        # self.raw_data = Reader(path)
        self.parsed_data = XmlBatchReader(id=0)
        self.n = self.parsed_data.answer_cnt
        self.xml_ans_ref = {}
        self.xml_ques_ref = {}
        self.last_question = None
        # Features PART I: NER features:
        self.repeat_cnt = []
        self.unique_cnt = []  # unique in answer
        # Features PART II: Word level features
        self.sobj_sim = []  # compute average(max(similarity)) for each nn.
        self.dobj_sim = []  # compute average(max(similarity)) for each vv.
        self.n_synset_sim = []  # non-zero only if A is the hypernym of B.
        self.label = []
        self.ml_model = None

    def origin_sentence(self, sentence_root):
        origin = ''
        for token in sentence_root[0]:
            origin += token.findtext('word')
        # print origin
        return origin

    def get_sim_score(self, word, word_list):
        sim = [cow.get_sim(word, item) for item in word_list] + [0.0]
        return max(sim)

    def question_pattern(self,dependency_root,pos_root):
        patterns = [
            (0,[u'什么',u'哪[\u4e00-\u9fa5]*',u'谁'],None,['det']), #什么地方？什么时候？ 定语
            (1,[u'多[\u4e00-\u9fa5]*',u'几[\u4e00-\u9fa5]*'],None,None),#有多少 距离，数量，程度
            (2,[u'什么',u'哪[\u4e00-\u9fa5]*',u'谁'],None,['dobj']), #是什么？ 做了什么？ 宾语
            (3,[u'什么',u'哪[\u4e00-\u9fa5]*',u'谁'],None,['nsubj']), #哪些是？ 什么是？ 主语
            (4,[u'怎[\u4e00-\u9fa5]*'],None,None) #询问方式
        ]
        def find(pattern):
            word_match = False
            for token in pos_root:
                #print token.tag,token.attrib
                word = token.findtext('word')
                if not word:
                    print 'not word'
                    return False
                for reg in pattern[1]:
                    if re.match(reg,word.decode('utf-8')):
                        word_match = True
                if not word_match:
                    continue
                if pattern[3] is None:
                    return True
                for dep in dependency_root:
                    if dep.attrib['type'] in pattern[3]:
                        return True
            return False

        for pattern in patterns:
            if find(pattern):
                return pattern[0]
        return -1

    def key_query_word_feature(self,xml_q,xml_a):
        token_q = next(xml_q.iterfind('tokens'))
        dep_q = next(xml_q.iterfind('dependencies'))
        token_a = next(xml_a.iterfind('tokens'))
        dep_a = next(xml_a.iterfind('dependencies'))
        def find_dependency_pair(deps,type_list):
            for dep in deps:
                if dep.attrib['type'] in type_list:
                    return dep.findtext('governor'),dep.findtext('dependent')
            return None
        def find_token(tokens,pos_list):
            for token in tokens:
                if token.findtext('POS') in pos_list:
                    return token
        def ner(tokens):
            l = []
            for token in tokens:
                if token.findtext('NER') != 'O':
                    l.append(token.findtext('word'))
            return l
        #记录上一个问题是什么，判断是否是重复的问题
        new_question = False
        q = self.origin_sentence(xml_q)
        if q != self.last_question:
            new_question = True
            self.last_question = q
        word_list = [token.findtext('word') for token in token_a]
        #答案的第一句话都是总结性话语，里面的NER是无用的。
        if new_question:
            self.useless_ner = [word for word in ner(token_a)]
            for word in self.useless_ner:
                print word
        #句子的主要成分是[时间，地点][主语][谓语][宾语]
        #找到动宾关系
        pair = find_dependency_pair(dep_q,['dobj'])
        if pair is None:
            dobj_score = 0.0
        else:
            print pair[0],pair[1]
            if pair[0] in ['有','是']: #存在性动词，只看宾语
                dobj_score = self.get_sim_score(pair[1],word_list)
            else:
                s1 = self.get_sim_score(pair[0],word_list)
                s2 = self.get_sim_score(pair[1],word_list)
                dobj_score = (s1+s2)/ 2
        #找到主谓关系,NP(名词词组)，VP(动词词组)
        sobj_score = 0.0
        tree = Sematic_tree()
        tree.s = xml_q.findtext('parse').encode('utf-8')
        tree.build_tree_from_root()
        VP_l = tree.find_tag('VP')
        print [node.tag for node in tree.flatten]
        leaf_word = []
        for VP in VP_l:
            NP = tree.find_nearest_tag(VP,'NP')
            if NP:
                t_leaf_word = []
                tree.find_all_leaf_word(NP,t_leaf_word)
                for item in t_leaf_word:
                    if item not in leaf_word:
                        leaf_word.append(item)
        for word in leaf_word:
            word = word.decode('utf-8')
            if word in self.useless_ner:
                continue
            print word, 'leaf word'
            s = self.get_sim_score(word, word_list)
            sobj_score += s
        print sobj_score,'sobj_score'
        print dobj_score,self.origin_sentence(xml_q),self.origin_sentence(xml_a)
        return sobj_score,dobj_score

    def extract_feature_single(self, xml_q, xml_a, label):
        ner_q, ner_a = [], []
        words_q, words_a = [], []
        pos_q, pos_a = {}, {}
        for token in xml_q[0]:
            words_q.append(token.findtext('word'))
            #print token.findtext('word')
            ner_t = token.findtext('NER')
            if ner_t in ['NUMBER','DATE',u'NUMBER',u'DATE']:
                ner_q.append(token.findtext('word'))
            pos_t = token.findtext('POS')
            if not pos_q.get(pos_t, None):
                pos_q[pos_t] = []
            pos_q[pos_t].append(token.findtext('word'))

        for token in xml_a[0]:
            words_a.append(token.findtext('word'))
            ner_t = token.findtext('NER')
            if ner_t in ['NUMBER','DATE',u'NUMBER',u'DATE']:
                ner_a.append(token.findtext('word'))
            pos_t = token.findtext('POS')
            if not pos_a.get(pos_t, None):
                pos_a[pos_t] = []
            pos_a[pos_t].append(token.findtext('word'))
        # ner feature
        ner_rpt,ner_unq = self.extract_ner_feature(ner_q,ner_a)
        print ner_rpt,ner_unq,';'
        # pos/word similarity feature
        # TODO: Information Entropy should be taken into consideration; common words in answer set should be less significant.
        #NN_score,VV_score = self.extract_pos_feature(pos_q,pos_a)
        sobj_score,dobj_score = self.key_query_word_feature(xml_q,xml_a)
        self.repeat_cnt.append(float(ner_rpt))
        self.unique_cnt.append(float(ner_unq))
        self.sobj_sim.append(float(sobj_score))
        self.dobj_sim.append(float(dobj_score))
        #self.n_sim.append(float(NN_score))
        #self.v_sim.append(float(VV_score))
        self.label.append(label)
        debug1.append(self.origin_sentence(xml_a))
        debug2.append(self.origin_sentence(xml_q))
        #print self.origin_sentence(xml_a),self.origin_sentence(xml_q),label

    def extract_ner_feature(self,ner_q,ner_a):
        ner_rpt, ner_unq = 0, 0
        for item in ner_q:
            if item in ner_a:
                ner_rpt += 1.0
        ner_rpt = ner_rpt / len(ner_q) if len(ner_q) else 1
        for item in ner_a:
            if item not in ner_q:
                ner_unq += 1.0
        ner_unq = ner_unq / len(ner_a) if len(ner_a) else 0
        return ner_rpt,ner_unq

    def question_pattern_feature(self,sentence_root):
        pos_root = next(sentence_root.iterfind('tokens'))
        dep_root = next(sentence_root.iterfind('dependencies'))
        res = self.question_pattern(dep_root,pos_root)
        print 'type',res
        print self.origin_sentence(sentence_root)
        raw_input()

    def extract_features(self):
        question_roots = self.parsed_data.question_xml[0][0]
        answer_roots = self.parsed_data.answer_xml[0][0]
        ignore_cnt = 0
        for sentence in answer_roots:
            first_token_offset = int(sentence[0][0].findtext('CharacterOffsetBegin'))
            self.xml_ans_ref[first_token_offset] = sentence
        for sentence in question_roots:
            first_token_offset = int(sentence[0][0].findtext('CharacterOffsetBegin'))
            self.xml_ques_ref[first_token_offset] = sentence
        for (answer_offset, question_offset) in self.parsed_data.offset_aq_dict.items():
            label = self.parsed_data.offset_al_dict[answer_offset]
            try:
                xml_ans = self.xml_ans_ref[answer_offset]
            except KeyError:
                print '[Ignore]Key-ans not exists %d' % answer_offset
                continue
            try:
                xml_ques = self.xml_ques_ref[question_offset]
            except KeyError:
                print '[Ignore]Key-ques not exists %d' %  question_offset
                continue
            self.extract_feature_single(xml_ques, xml_ans, label)

    def train_model(self):
        from sklearn import linear_model
        self.ml_model = linear_model.LogisticRegression()
        print len(self.repeat_cnt)
        print self.repeat_cnt
        X = np.array([self.repeat_cnt, self.unique_cnt, self.sobj_sim, self.dobj_sim]).transpose()
        y = np.array(self.label)
        self.ml_model.fit(X, y)
        # LRpredictionprob=logistic.predict_proba(X_test_std)
        y_ = self.ml_model.predict_proba(X)
        f = open('weights.txt', 'w')
        for i in range(len(y)):
            f.write('{0} {1}\n'.format(X[i], y[i]))
        return y, y_



simple_model = SentenceStructureModel()
simple_model.extract_features()
y, y_ = simple_model.train_model()
f = open('log.txt', 'w')
for i in range(len(y)):
    f.write(str(y[i]) + '\t' + str(y_[i]) + '\t' + debug1[i] + '\t' + debug2[i] + '\n')
f.close()
