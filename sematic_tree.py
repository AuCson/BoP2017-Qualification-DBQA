class Sematic_tree:
    def __init__(self):
        self.root = self.Node()
        self.s = ""
        self.quote_ref = {}

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
        r = self.quote_ref[start_pos]
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
        self.root = self.build_tree(0)


if __name__=='main':
    with open('debug.txt','r') as f:
        tree = Sematic_tree()
        tree.s = f.read()
        tree.build_tree_from_root()
        print 1