class FeatureExtractor:
    def __init__(self, train_vocab):
        self.train_vocab = train_vocab

    def word2features(self, sent, i):
        #this function creates a features of the current word.
        word = sent[i]  #current word

        # Base features for the current word
        features = {
            'bias': 1.0,
            'word': word,
            'word.lower()': word.lower(),
            'word[-4:]': word[-4:],  # Last 4 letters for suffixes
            'word[-3:]': word[-3:],  # Last 3 letters for suffixes
            'word[-2:]': word[-2:],  # Last 2 letters for suffixes
            'word[-1:]': word[-1:],  # Last 1 letter for suffixes
            'word[:4]': word[:4],    # First 4 letters for prefixes
            'word[:3]': word[:3],    # First 3 letters for prefixes
            'word[:2]': word[:2],    # First 2 letters for prefixes
            'word[:1]': word[:1],    # First 1 letter for prefixes
            'word.length()': len(word), #length of the word
            'word.contains_hyphen()': '-' in word, #check if the word contains hyphen
            'word.isupper()': word.isupper(), #check if all the letters are capitalised
            'word.islower()': word.islower(),
            'word.istitle()': word.istitle(), #check if the first letter is capatilised
            'word.isdigit()': word.isdigit(), #check if the word is a number
            'is_unknown': word.lower() not in self.train_vocab,
        }

        # Features from the previous word
        if i > 0:
            word_prev = sent[i - 1]
            features.update({
                '-1:word.lower()': word_prev.lower(),
                '-1:word_prev()': word_prev,
                '-1:word.istitle()': word_prev.istitle(),
                '-1:word.isupper()': word_prev.isupper(),
                '-1:word.islower()': word_prev.islower(),
                '-1:word.isdigit()': word.isdigit(),
            })
        else:
            features['BOS'] = True  # Beginning of sentence

        # Features from two words before
        if i > 1:
            word_prev2 = sent[i - 2]
            features.update({
                '-2:word.lower()': word_prev2.lower(),
                '-2:word_prev2()': word_prev2,
                '-2:word.istitle()': word_prev2.istitle(),
                '-2:word.isupper()': word_prev2.isupper(),
                '-2:word.islower()': word_prev2.islower(),
                '-2:word.isdigit()': word.isdigit(),
            })
        #else:
        #    features['BOS2'] = True  # Two words from the beginning of sentence

        # Features from the next word
        if i < len(sent) - 1:
            word_fwd = sent[i + 1]
            features.update({
                '+1:word.lower()': word_fwd.lower(),
                '+1:word_fwd()': word_fwd,
                '+1:word.istitle()': word_fwd.istitle(),
                '+1:word.isupper()': word_fwd.isupper(),
                '+1:word.islower()': word_fwd.islower(),
                '+1:word.isdigit()': word.isdigit(),
            })
        else:
            features['EOS'] = True  # End of sentence

        return features

    def sent2features(self, sent):
        #this function creates a features of the sentence.
        feat = []
        for i in range(len(sent)):
            feat.append(self.word2features(sent, i))
        return feat