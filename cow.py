# coding=utf-8
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
import codecs
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


# http://blog.csdn.net/xieyan0811/article/details/62056558#
# 还是要吐槽下这博主代码好难看
def loadWordNet():
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
        l = known_ref.get(item[1],[])
        if not len(l):
            known_ref[item[1]] = []
            known_ref[item[1]].append(item[0])
        i+=1
    return known_ref

known = loadWordNet()
#神他妈一个set还要你去遍历找，人家python辛辛苦苦给你哈希意义何在，吃屎去吧
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

def find_word_net_id(known_ref,key):
    found_list = known_ref.get(key,[]) + known_ref.get(key+u'+的',[]) + known_ref.get(key+u'+地',[])
    return [id2ss(i) for i in found_list]

def get_sim(word_a,word_b):
    if word_a == word_b:
        return 1.0
    try:
        a = find_word_net_id(known,word_a)[0]
        b = find_word_net_id(known,word_b)[0]
        sim = a.path_similarity(b)
        return sim
    except IndexError: # 没有找到这个词汇
        #一个尝试性的方法：中文中，两个同义词通常有相同的字，因此……
        score = 0.0
        for char in word_a:
            if char in word_b:
                score += 1.0
        score /= len(word_a)
        return score
    except ValueError:
        return 0.0