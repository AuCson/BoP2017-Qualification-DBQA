class Sematic_tree:
    def __init__(self):
        self.root = self.Node()
        self.s = ""
        self.quote_ref = {}
        self.flatten = []

    class Node:
        def __init__(self):
            self.tag = ''
            self.word = ''
            self.par = 0
            self.child = []

    def preprocess_quote(self):
        stack = []
        i = 0
        while i < len(self.s):
            if self.s[i] == '(':
                stack.append(i)
            elif self.s[i] == ')':
                l = stack[-1]
                stack.pop()
                self.quote_ref[l] = i
            i += 1

    def build_tree(self, start_pos):
        n = self.Node()
        i = start_pos + 1
        # ignore blanks
        while str.isspace(self.s[i]):
            i += 1
        # get the tag of the node
        c = i
        while not (str.isspace(self.s[i]) or self.s[i] in ('(',')')):
            i += 1
        n.tag = self.s[c:i]
        # recurrently build the tree
        try:
            r = self.quote_ref[start_pos]
        except KeyError:
            print self.quote_ref
            print self.s
            r = 0
            raw_input()
        while i < r:
            while str.isspace(self.s[i]):
                i += 1
            if self.s[i] != '(':
                # it is leaf
                n.word = self.s[i:r]
                break
            new_n = self.build_tree(i)
            n.child.append(new_n)
            i = self.quote_ref[i] + 1
        return n

    def build_tree_from_root(self):
        self.preprocess_quote()
        self.root = self.build_tree(min(self.quote_ref.keys()))

    def do_flatten(self,node,res):
        res.append(node)
        for child in node.child:
            self.do_flatten(child,res)
        self.flatten = res

    def find_tag(self,tag):
        if not self.flatten:
            self.do_flatten(self.root,self.flatten)
        res = []
        for node in self.flatten:
            if node.tag == tag:
                res.append(node)
        return res

    def find_nearest_tag(self,node,tag):
        if not self.flatten:
            self.do_flatten(self.root,self.flatten)
        idx = -1
        for i,n in enumerate(self.flatten):
            if node == n:
                idx = i
                break
        if idx == -1:
            return None
        while idx >= 0:
            if self.flatten[idx].tag == tag:
                return self.flatten[idx]
            idx -= 1
        return None

    def find_all_leaf_word(self,node,res):
        if node.word != '':
            res.append(node.word)
        for child in node.child:
            self.find_all_leaf_word(child,res)


def test():
    with open('debug.txt','r') as f:
        tree = Sematic_tree()
        tree.s = f.read()
        tree.build_tree_from_root()
        print 1