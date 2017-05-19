# coding=utf-8
from file_preprocessor import XmlBatchReader, Reader
from xml.etree import cElementTree as ET
import cow
import numpy as np

debug1 = []
debug2 = []

class SentenceStructureModel:
    def __init__(self, path='data/BoP2017-DBQA.train.txt'):
        # self.raw_data = Reader(path)
        self.parsed_data = XmlBatchReader(id=0)
        self.n = self.parsed_data.answer_cnt
        self.xml_ans_ref = {}
        self.xml_ques_ref = {}
        # Features PART I: NER features:
        self.repeat_cnt = []
        self.unique_cnt = []  # unique in answer
        # Features PART II: Word level features
        self.n_sim = []  # compute average(max(similarity)) for each nn.
        self.v_sim = []  # compute average(max(similarity)) for each vv.
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

    def extract_feature_single(self, xml_q, xml_a, label):
        ner_q, ner_a = [], []
        words_q, words_a = [], []
        pos_q, pos_a = {}, {}
        ner_rpt, ner_unq = 0, 0
        for token in xml_q[0]:
            words_q.append(token.findtext('word'))
            ner_t = token.findtext('NER')
            if ner_t != 'O':
                ner_q.append(token.findtext('word'))
            pos_t = token.findtext('POS')
            if not pos_q.get(pos_t, None):
                pos_q[pos_t] = []
            pos_q[pos_t].append(token.findtext('word'))

        for token in xml_a[0]:
            words_a.append(token.findtext('word'))
            ner_t = token.findtext('NER')
            if ner_t != 'O':
                ner_a.append(token.findtext('word'))
            pos_t = token.findtext('POS')
            if not pos_a.get(pos_t, None):
                pos_a[pos_t] = []
            pos_a[pos_t].append(token.findtext('word'))
        # ner feature
        for item in ner_q:
            if item in words_a:
                ner_rpt += 1.0
        ner_rpt = ner_rpt / len(ner_q) if len(ner_q) else 1

        for item in ner_a:
            if item not in ner_q:
                ner_unq += 1.0
        ner_unq = ner_unq / len(ner_a) if len(ner_a) else 0
        # pos/word similarity feature
        # TODO: Information Entropy should be taken into consideration; common words in answer set should be less significant.
        NN_cnt, VV_cnt = 0, 0
        NN_score, VV_score = 0.0, 0.0
        for word in pos_q.get('NN', []):
            if self.get_sim_score(word, pos_a.get('VV', [])) != -1:
                NN_score += self.get_sim_score(word, pos_a.get('NN', []))
                NN_cnt += 1
        for word in pos_q.get('VV', []):
            if self.get_sim_score(word, pos_a.get('VV', [])) != -1:
                VV_score += self.get_sim_score(word, pos_a.get('VV', []))
                VV_cnt += 1
        NN_score = NN_score / NN_cnt if NN_cnt else 0
        VV_score = VV_score / VV_cnt if VV_cnt else 0

        self.repeat_cnt.append(float(ner_rpt))
        self.unique_cnt.append(float(ner_unq))
        self.n_sim.append(float(NN_score))
        self.v_sim.append(float(VV_score))
        self.label.append(label)
        debug1.append(self.origin_sentence(xml_a))
        debug2.append(self.origin_sentence(xml_q))
        print self.origin_sentence(xml_a),self.origin_sentence(xml_q),label

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
        X = np.array([self.repeat_cnt, self.unique_cnt, self.n_sim, self.v_sim]).transpose()
        y = np.array(self.label)
        self.ml_model.fit(X, y)
        # LRpredictionprob=logistic.predict_proba(X_test_std)
        y_ = self.ml_model.predict_proba(X)
        f = open('weights.txt', 'w')
        for i in range(len(y)):
            f.write('{0} {1}\n'.format(X[i], y[i]))
        return y, y_


def test():
    simple_model = SentenceStructureModel()
    simple_model.extract_features()
    y, y_ = simple_model.train_model()
    f = open('log.txt', 'w')
    for i in range(len(y)):
        f.write(str(y[i]) + '\t' + str(y_[i]) + '\t' + debug1[i] + '\t' + debug2[i] + '\n')
    f.close()


test()