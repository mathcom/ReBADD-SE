from abc import *


class Voca(metaclass=ABCMeta):
    tok_pad = "<pad>"
    tok_unk = "<unk>"
    tok_sos = "<sos>"
    tok_eos = "<eos>"
    index2word = [tok_pad, tok_unk, tok_sos, tok_eos]
    word2index = {v:k for k,v in enumerate(index2word)}
    word2count = {}
    num_words = len(index2word)
    filepath = None
        
    def make_dict(self, lines):
        for line in lines:
            self.addWord(line)
        
    def addWord(self, sentence):
        for w in set(self.tokenized(sentence)):
            ## w2i, i2w
            if w not in self.word2index:
                self.index2word.append(w)
                self.word2index[w] = self.num_words
                self.num_words += 1
            ## counts
            self.word2count[w] = self.word2count.get(w, 0) + 1

    def indexesFromSentence(self, sentence):
        return [self.tok_sos] + [word for word in self.tokenzied(sentence)] + [self.tok_eos]
        
    def load(self):
        '''
        < example >
        <pad>	0
        <unk>	0
        <sos>	0
        <eos>	0
        MG	69072
        GF	142975
        FV	133118
        '''
        with open(self.filepath) as fin:
            lines = fin.readlines()
            lines = [line.rstrip().split('\t') for line in lines]
        ## parse
        wc = [(w, int(c)) for w, c in lines]
        ## setup
        self.index2word = [w for w, c in wc]
        self.word2count = {w:c for w, c in wc}
        self.word2index = {w:k for k, w in enumerate(self.index2word)}
        self.num_words = len(self.index2word)
        self.tok_pad = self.index2word[0]
        self.tok_unk = self.index2word[1]
        self.tok_sos = self.index2word[2]
        self.tok_eos = self.index2word[3]
        
    @abstractmethod
    def tokenzied(self):
        pass


class ProteinVoca(Voca):
    def __init__(self, filepath=None):
        self.filepath = filepath
        if self.filepath:
            self.load()
        
    def tokenzied(self, line):
        return [line[i:i+2] for i in range(len(line) - 2)]


class LigandVoca(Voca):
    def __init__(self, filepath=None):
        self.filepath = filepath
        if self.filepath:
            self.load()
    
    def tokenzied(self, input):
        return input
