# coding=utf-8
from file_preprocessor import XmlBatchReader, Reader
from xml.etree import cElementTree as ET
import cow
import numpy as np
import re
from sematic_tree import Sematic_tree
from math import log

debug1 = []
debug2 = []


class SentenceStructureModel:
    def __init__(self, path='data/BoP2017-DBQA.train.txt', fid=0, train=True):
        # self.raw_data = Reader(path)
        self.train = train
        self.parsed_data = XmlBatchReader(fid,test=not train)
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
        self.ie_feature = []
        # Features PART III: Word Overlap features
        self.word_overlap = []
        self.label = []
        self.y_ = []
        self.ml_model = None
        self.invalid_cnt = 0
        self.prev_xml_q = None

    def origin_sentence(self, sentence_root):
        origin = ''
        for token in sentence_root[0]:
            origin += token.findtext('word')
        # print origin
        return origin

    def get_sim_score(self, word, word_list):
        sim = [cow.get_sim(word, item) for item in word_list] + [0.0]
        return max(sim)

    def question_pattern(self, dependency_root, pos_root):
        patterns = [
            (0, [u'什么', u'哪[\u4e00-\u9fa5]*', u'谁'], None, ['det']),  # 什么地方？什么时候？ 定语
            (1, [u'多[\u4e00-\u9fa5]*', u'几[\u4e00-\u9fa5]*'], None, None),  # 有多少 距离，数量，程度
            (2, [u'什么', u'哪[\u4e00-\u9fa5]*', u'谁'], None, ['dobj']),  # 是什么？ 做了什么？ 宾语
            (3, [u'什么', u'哪[\u4e00-\u9fa5]*', u'谁'], None, ['nsubj']),  # 哪些是？ 什么是？ 主语
            (4, [u'怎[\u4e00-\u9fa5]*'], None, None)  # 询问方式
        ]

        def find(pattern):
            word_match = False
            for token in pos_root:
                # print token.tag,token.attrib
                word = token.findtext('word')
                if not word:
                    print 'not word'
                    return False
                for reg in pattern[1]:
                    if re.match(reg, word.decode('utf-8')):
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

    def repeat_word_feature(self,xml_q,xml_a,freq_dict):
        ques_word = []
        score = 0.0
        for token in xml_q[0]:
            ques_word.append(token.findtext('word'))

        s = 0.0
        for item in freq_dict.items():
            if item[0] in ques_word:
                s += item[1]
        #calculate the information entropy of each word
        #print freq_dict
        if s == 0.0:
            return 0.0
        for item in freq_dict.items():
            #print item
            freq_dict[item[0]] = -log(item[1]/s,2)

        for token in xml_a[0]:
            word = token.findtext('word')
            if word in ques_word:
                #print freq_dict[word]
                score += freq_dict[word]
        return score

    def key_query_word_feature(self, xml_q, xml_a):
        token_q = next(xml_q.iterfind('tokens'))
        dep_q = next(xml_q.iterfind('dependencies'))
        token_a = next(xml_a.iterfind('tokens'))
        dep_a = next(xml_a.iterfind('dependencies'))

        def find_dependency_pair(deps, type_list):
            for dep in deps:
                if dep.attrib['type'] in type_list:
                    return dep.findtext('governor'), dep.findtext('dependent')
            return None

        def ner(tokens):
            l = []
            for token in tokens:
                if token.findtext('NER') != 'O':
                    l.append(token.findtext('word'))
            return l

        # 记录上一个问题是什么，判断是否是重复的问题
        new_question = False
        q = self.origin_sentence(xml_q)
        if q != self.last_question:
            new_question = True
            self.last_question = q
        word_list = [token.findtext('word') for token in token_a]
        #pos_list = [token.findtext('POS') for token in token_a]
        # 答案的第一句话都是总结性话语，里面的NER是无用的。
        if new_question:
            self.useless_ner = [word for word in ner(token_a)]

        # 句子的主要成分是[时间，地点][主语][谓语][宾语]
        # 找到动宾关系
        pair = find_dependency_pair(dep_q, ['dobj'])
        if pair is None:
            dobj_score = 0.0
        else:
            #print pair[0], pair[1]
            if pair[0] in ['有', '是']:  # 存在性动词，只看宾语
                dobj_score = self.get_sim_score(pair[1], word_list)
            else:
                s1 = self.get_sim_score(pair[0], word_list)
                s2 = self.get_sim_score(pair[1], word_list)
                dobj_score = (s1 + s2) / 2
        # 找到主谓关系,NP(名词词组)，VP(动词词组)
        sobj_score = 0.0
        tree = Sematic_tree()
        tree.s = xml_q.findtext('parse').encode('utf-8')
        try:
            tree.build_tree_from_root()
        except Exception:
            print 'tree error'
            return 0.0,0.0
        VP_l = tree.find_tag('VP')
        #print [node.tag for node in tree.flatten]
        leaf_word = []
        for VP in VP_l:
            NP = tree.find_nearest_tag(VP, 'NP')
            if NP:
                t_leaf_word = []
                tree.find_all_leaf_word(NP, t_leaf_word)
                for item in t_leaf_word:
                    if item not in leaf_word:
                        leaf_word.append(item)
        for word in leaf_word:
            word = word.decode('utf-8')
            if word in self.useless_ner:
                continue
            #print word, 'leaf word'
            s = self.get_sim_score(word, word_list)
            sobj_score += s
        #print sobj_score, 'sobj_score'
        #print dobj_score, self.origin_sentence(xml_q), self.origin_sentence(xml_a)
        return sobj_score, dobj_score

    def extract_feature_single(self, xml_q, xml_a, label, xml_a_batch):
        if self.prev_xml_q != xml_q:
            self.prev_xml_q = xml_q
            self.words_a_batch = []
            #print xml_a_batch,self.prev_xml_q,xml_q
            for _xml_a in xml_a_batch:
                for token in _xml_a[0]:
                    self.words_a_batch.append(token.findtext('word'))

        #make freq dict
        freq_dict = {}
        for word in self.words_a_batch:
            time = freq_dict.get(word,0)
            if time == 0:
                freq_dict[word] = 1
            else:
                freq_dict[word] += 1
        #print freq_dict
        # TODO: Information Entropy should be taken into consideration; common words in answer set should be less significant.
        sobj_score, dobj_score = self.key_query_word_feature(xml_q, xml_a)
        ie_score = self.repeat_word_feature(xml_q,xml_a,freq_dict)
        self.add_to_weight(sobj_score,dobj_score, ie_score, label)
        debug1.append(self.origin_sentence(xml_a))
        debug2.append(self.origin_sentence(xml_q))
        # print self.origin_sentence(xml_a),self.origin_sentence(xml_q),label

    def add_to_weight(self, sobj_score,dobj_score, ie_score, label):
        self.sobj_sim.append(float(sobj_score))
        self.dobj_sim.append(float(dobj_score))
        self.ie_feature.append(float(ie_score))
        if self.train:
            self.label.append(label)

    def question_pattern_feature(self, sentence_root):
        pos_root = next(sentence_root.iterfind('tokens'))
        dep_root = next(sentence_root.iterfind('dependencies'))
        res = self.question_pattern(dep_root, pos_root)
        print 'type', res
        print self.origin_sentence(sentence_root)
        raw_input()

    def extract_features(self,to_Xy=True):
        question_roots = self.parsed_data.question_xml[0][0]
        answer_roots = self.parsed_data.answer_xml[0][0]

        for sentence in answer_roots:
            first_token_offset = int(sentence[0][0].findtext('CharacterOffsetBegin'))
            self.xml_ans_ref[first_token_offset] = sentence
        for sentence in question_roots:
            first_token_offset = int(sentence[0][0].findtext('CharacterOffsetBegin'))
            self.xml_ques_ref[first_token_offset] = sentence

        prev_question_offset = -1
        xml_a_batch = []
        xml_q_batch = []
        label_batch = []
        unicode_fix = 0
        unicode_fix_a = 0
        aq_dict_sorted = sorted(self.parsed_data.offset_aq_dict.items(),key = lambda x:x[1]) + [(-1,-1)]
        for (_answer_offset, _question_offset) in aq_dict_sorted:
            # update
            _answer_offset += unicode_fix
            _question_offset += unicode_fix_a
            if _question_offset != prev_question_offset:
                #print _question_offset,prev_question_offset
                for i in range(len(xml_a_batch)):
                    self.extract_feature_single(xml_q_batch[i], xml_a_batch[i], label_batch[i],xml_a_batch)
                xml_a_batch,xml_q_batch,label_batch = [],[],[]
                prev_question_offset = _question_offset
                if _question_offset == -1:
                    break

            if self.train:
                label = self.parsed_data.offset_al_dict[_answer_offset]
            else:
                label = None
            if self.xml_ans_ref.get(_answer_offset,-1)!=-1:
                xml_ans = self.xml_ans_ref[_answer_offset]
            else:
                print '[Ignore]Key-ans not exists %d' % _answer_offset

                if self.xml_ans_ref.get(_answer_offset-1,1) != 1:
                    print 'found(_answer_offset-1)'
                    xml_ans = self.xml_ans_ref[_answer_offset - 1]
                elif self.xml_ans_ref.get(_answer_offset+1,1) != 1:
                    print 'found(_answer_offset+1)'
                    xml_ans = self.xml_ans_ref[_answer_offset + 1]

                else:

                    self.add_to_weight(-1, -1, -1, label)
                    debug1.append('#')
                    debug2.append('#')
                    self.invalid_cnt += 1
                    continue

            try:
                xml_ques = self.xml_ques_ref[_question_offset]
            except KeyError:
                print '[Ignore]Key-ques not exists %d' % _question_offset
                '''
                if self.xml_ques_ref.get(_question_offset-1,1) != 1:
                    print 'found(_question_offset-1)'
                    xml_ques = self.xml_ans_ref[_question_offset - 1]
                    unicode_fix_a -= 1
                if self.xml_ans_ref.get(_question_offset+1,1) != 1:
                    print 'found(_question_offset+1)'
                    xml_ques = self.xml_ans_ref[_question_offset + 1]
                    unicode_fix_a += 1
                '''
                self.add_to_weight(-1, -1, -1, label)
                debug1.append('#')
                debug2.append('#')
                self.invalid_cnt += 1
                continue


            xml_a_batch.append(xml_ans)
            xml_q_batch.append(xml_ques)
            label_batch.append(label)

        self.replace_nan()
        if to_Xy:
            self.to_Xy()

    def to_Xy(self):
        self.X = np.array([self.sobj_sim, self.dobj_sim, self.ie_feature]).transpose()
        if self.train:
            self.y = np.array(self.label)
        print self.X.shape

    def replace_nan(self):
        for l in [self.ie_feature, self.sobj_sim, self.dobj_sim]:
            avg = sum([i for i in l if i != -1]) / (len(l) - self.invalid_cnt)
            for i in range(len(l)):
                if l[i] == -1:
                    l[i] = avg

    def scale_feature(self,X):
        from sklearn import preprocessing
        min_max_scaler = preprocessing.MinMaxScaler()
        return min_max_scaler.fit_transform(X)

    def train_model(self):
        from sklearn import linear_model
        self.X= self.scale_feature(self.X)
        self.ml_model = linear_model.LogisticRegression()
        self.ml_model.fit(self.X, self.y)
        return self.ml_model

    def predict(self, trained_model):
        #self.X = np.array([self.sobj_sim, self.dobj_sim]).transpose()
        #self.X = np.array([self.repeat_cnt, self.unique_cnt, self.sobj_sim, self.dobj_sim]).transpose()
        self.X = self.scale_feature(self.X)
        self.y_ = trained_model.predict_proba(self.X)

    def mrr_calc(self):
        qa_items = self.parsed_data.offset_aq_dict.items()
        items = self.parsed_data.offset_al_dict.items()
        print len(qa_items),len(items)
        truth_i = -1 if items[0][1] == 0 else 1
        buf = [(0, self.y_[0])]
        prev_q_offset = qa_items[0][1]
        i = 1
        s = 0.0
        q = 0
        exact_answer = 0
        while i < len(items):
            if prev_q_offset != qa_items[i][1]:
                #print i,prev_q_offset,qa_items[i][1]
                if truth_i != -1:
                    #print buf
                    buf.sort(key=lambda x: x[1][1],reverse=True)
                    rank = 0
                    for idx,item in enumerate(buf):
                        if item[0]==truth_i:
                            rank = idx+ 1
                            break
                    assert(rank>=1)
                    if rank == 1:
                        exact_answer += 1
                    s += 1.0 / rank
                    q += 1
                buf = []
            buf.append((i, self.y_[i]))
            prev_q_offset = qa_items[i][1]
            assert (items[i][1] in [0, 1])
            if items[i][1] == 1:
                truth_i = i
            i += 1
        mpp = s / q
        print "mRR:", mpp
        print "1/mRR", 1.0 / mpp
        print "exact answer:", exact_answer
        print "total question:", q

    def log(self,file_name='log.txt'):
        f = open(file_name, 'w')
        for i in range(len(self.y)):
            f.write(str(self.y[i]) + '\t' + str(self.y_[i]) + '\t' + debug1[i] + '\t' + debug2[i] + str(self.X[i])+'\n')
        f.close()

    def save_feature(self,file_name='feature.txt',label=False):
        f = open(file_name,'w')
        for i in range(len(self.y)):
            if label:
                f.write(str(self.X[i])+'\t'+str(self.y[i])+'\n')
            else:
                f.write(str(self.X[i])+'\n')
        f.close()

    def output_prediction(self,outfile = 'out.txt'):
        f = open(outfile,'w')
        for i in range(len(self.y_)):
            f.write(str(self.y_[i][1])+'\n')
        f.close()

class BatchModel:
    def batch_train(self,maxnum=230000):
        print 'training ', 0
        self.train_model = SentenceStructureModel(fid=0)
        self.train_model.extract_features(to_Xy=False)
        i = 10000
        while i <= maxnum:
            print 'training ', i
            t_model = SentenceStructureModel(fid = i)
            t_model.extract_features(to_Xy=False)
            self.train_model.sobj_sim += t_model.sobj_sim
            self.train_model.dobj_sim += t_model.dobj_sim
            self.train_model.ie_feature += t_model.ie_feature
            self.train_model.label += t_model.label
            i += 10000
        self.train_model.to_Xy()
        self.ml_model = self.train_model.train_model()
        self.train_model.save_feature(label=True)
        #save model
        try:
            from sklearn.externals import joblib
            joblib.dump(self.ml_model,'trained_model.m')
        except Exception:
            print 'error on dumping'

    def load_saved_model(self):
        from sklearn.externals import joblib
        self.ml_model = joblib.load('trained_model.m')

    def batch_test(self,maxnum=190000):
        print 'testing ', 0
        self.feature_container = SentenceStructureModel(fid = 0,train=False)
        self.feature_container.extract_features(to_Xy=False)
        i = 10000
        while i<= maxnum:
            print 'testing ', i
            t_container = SentenceStructureModel(fid=i,train=False)
            t_container.extract_features(to_Xy=False)
            self.feature_container.sobj_sim += t_container.sobj_sim
            self.feature_container.dobj_sim += t_container.dobj_sim
            self.feature_container.ie_feature += t_container.ie_feature
            self.feature_container.label += t_container.label
            i+=10000
        self.feature_container.to_Xy()
        self.feature_container.predict(trained_model=self.ml_model)
        self.feature_container.output_prediction()


def test_model():
    train_model = SentenceStructureModel(fid=0)
    train_model.extract_features()
    model = train_model.train_model()

    test_model = SentenceStructureModel(fid=110000)
    debug1 = []
    debug2 = []
    test_model.extract_features()
    test_model.predict(model)
    test_model.mrr_calc()
    test_model.log()

def run():
    b = BatchModel()
    b.load_saved_model()
    model = b.ml_model
    fid = 0
    while fid <= 230000:
        test_model = SentenceStructureModel(fid=fid)
        debug1 = []
        debug2 = []
        test_model.extract_features()
        test_model.predict(model)
        test_model.mrr_calc()
        test_model.log()
        fid += 10000


def run_test():
    b = BatchModel()
    b.load_saved_model()
    b.batch_test()
