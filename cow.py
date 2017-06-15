# coding=utf-8
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
import codecs
import sys
import pickle

reload(sys)
sys.setdefaultencoding('utf-8')


# http://blog.csdn.net/xieyan0811/article/details/62056558#
# 还是要吐槽下这博主代码好难看
def loadWordNet():
    global trans_dict
    trans_dict = pickle.load(open('data/dictionary.pkl', 'rb'))
    f = codecs.open("cow/cow-not-full.txt", "rb", "utf-8")
    known = set()
    for l in f:
        if l.startswith('#') or not l.strip():
            continue
        row = l.strip().split("\t")
        if len(row) == 3:
            (synset, lemma, status) = row
        elif len(row) == 2:
            (synset, lemma) = row
            status = 'Y'
        else:
            print "illformed line: ", l.strip()
        if status in ['Y', 'O']:
            if not (synset.strip(), lemma.strip()) in known:
                known.add((synset.strip(), lemma.strip()))
    known_ref = {}
    i = 0
    print len(known)
    for item in known:
        l = known_ref.get(item[1], [])
        if not len(l):
            known_ref[item[1]] = []
            known_ref[item[1]].append(item[0])
        i += 1
    return known_ref


known = loadWordNet()
# 神他妈一个set还要你去遍历找，人家python辛辛苦苦给你哈希意义何在，吃屎去吧
'''
def findWordNet(known, key):
    ll = []
    for kk in known:
        if (kk[1] == key):
            ll.append(kk[0])
    return ll
'''


def id2ss(ID):
    return wn._synset_from_pos_and_offset(str(ID[-1:]), int(ID[:8]))


def find_word_net_id(known_ref, key):
    found_list = known_ref.get(key, []) + known_ref.get(key + u'+的', []) + known_ref.get(key + u'+地', [])
    return [id2ss(i) for i in found_list]


def get_sim(word_a, word_b):
    if word_a == word_b:
        return 1.0
    else:
    #try:
        #a = find_word_net_id(known, word_a)[0]
        #b = find_word_net_id(known, word_b)[0]
        #sim = a.path_similarity(b)
        #return sim

    #except IndexError:  # 没有找到这个词汇
        # 通过英文翻译去查找
        score = 0.0
        for char in word_a:
            if char in word_b:
                score += 1.0
        score /= len(word_a)
        return score
        '''
        try:
            word_a = word_a.encode('utf-8')
            word_b = word_b.encode('utf-8')
            en_word_a_l = trans_dict[word_a].split()
            en_word_b_l = trans_dict[word_b].split()
            sim_sum = 0.0

            max_cnt = 5
            cnt = 0

            for word_a in en_word_a_l:
                word_synset_a_first = wn.synsets(word_a)
                for word_b in en_word_b_l:
                    word_synset_b_first = wn.synsets(word_b)
                    max_sim = 0.0
                    break_flag = 0
                    for syn_a in word_synset_a_first:
                        for syn_b in word_synset_b_first:
                            if syn_a._pos == syn_b._pos:
                                sim = syn_a.path_similarity(syn_b)
                                max_sim = sim if sim is not None and sim > max_sim else max_sim
                                cnt += 1
                                if cnt >= max_cnt:
                                    break_flag = 1
                                    break
                        if break_flag:
                            break

                    sim_sum += max_sim
            sim_avg = sim_sum / (len(en_word_a_l) + len(en_word_b_l))
            return sim_avg
        except KeyError:
            return 0.0
        except IndexError:
            return 0.0
        '''
    #except ValueError:
    #    return 0.0


get_sim('美誉', '美称')