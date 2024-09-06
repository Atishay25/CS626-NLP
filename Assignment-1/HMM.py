from collections import defaultdict, Counter
import pickle


class Viterbi:
    
    # tags contain the name of all possible tags 
    # initial_probs is basically the probab of going from ^ to the word
    # emission_probs are the P(word|tags)
    def __init__(self, tags, initial_probs, transition_probs, emission_probs):

        self.tags = tags
        self.initial_probs = initial_probs
        self.transition_probs = transition_probs
        self.emission_probs = emission_probs
    
    def viterbi(self, sentence):
        prob_table = [{}]  # This is like a table which stores all the probablities
        path = {} # This dictionary will store good paths only.

        #This loop is only for the first word of the sentence.
        #here a path is created for all possible initial tags
        for tag in self.tags:
            prob_table[0][tag] = self.initial_probs.get(tag, 1e-7) * self.emission_probs.get(tag, {}).get(sentence[0], 1e-7)
            path[tag] = [tag]  # Initialize the path with the starting tag

        #Now we will calculate the paths for each word and will prune the tree suitably. 
        #basically we will calculate that for a current tag what sequence of previous tag is the most probable.
        for t in range(1, len(sentence)):
            prob_table.append({})
            new_path = {}

            for curr_tag in self.tags:
                best_prob = -1
                best_prev_tag = None
                
                # Find the best previous tag
                for prev_tag in self.tags:
                    prob = (prob_table[t-1].get(prev_tag, 1e-7) *
                            self.transition_probs.get(prev_tag, {}).get(curr_tag, 1e-7) *
                            self.emission_probs.get(curr_tag, {}).get(sentence[t], 1e-7))
                    
                    if prob > best_prob:
                        best_prob = prob
                        best_prev_tag = prev_tag

                # Update the probability table and new paths
                prob_table[t][curr_tag] = best_prob
                new_path[curr_tag] = path[best_prev_tag] + [curr_tag]

            path = new_path

        #finding the best final tag
        best_prob = -1
        best_last_tag = None

        for tag in self.tags:
            prob = prob_table[len(sentence) - 1].get(tag, 1e-7)
            if prob > best_prob:
                best_prob = prob
                best_last_tag = tag

        return path[best_last_tag]
    

class HiddenMarkovModel:

    def train(self, train_data):

        transition_probs = defaultdict(create_float_defaultdict)
        emission_probs = defaultdict(create_float_defaultdict)
        initial_probs = defaultdict(float)
        tags = set()
        vocabulary = set()

        tag_bigrams = Counter()
        tag_unigrams = Counter()
        word_tag_pairs = Counter()
        tag_starts = Counter()

        for sentence in train_data:

            prev_tag = None

            for word, tag in sentence:
                
                vocabulary.add(word)
                tags.add(tag)
                tag_unigrams[tag] += 1
                word_tag_pairs[(word, tag)] += 1

                if prev_tag is None:
                    tag_starts[tag] += 1
                else:
                    tag_bigrams[(prev_tag, tag)] += 1

                prev_tag = tag

        total_sentences = len(train_data)

        for tag in tags:
            initial_probs[tag] = tag_starts[tag] / total_sentences

        for (prev_tag, curr_tag), count in tag_bigrams.items():
            transition_probs[prev_tag][curr_tag] = count / tag_unigrams[prev_tag]

        for (word, tag), count in word_tag_pairs.items():
            emission_probs[tag][word] = count / tag_unigrams[tag]

        self.viterbi = Viterbi(tags, initial_probs, transition_probs, emission_probs)

    def predict(self, sentence):
        
        return self.viterbi.viterbi(sentence)
    
    def save(self, filename = 'model.pkl'):

        with open(filename, 'wb') as out:
            pickle.dump(self.viterbi, out, pickle.HIGHEST_PROTOCOL)

    def load(self, filename = 'model.pkl'):

        with open(filename, 'rb') as inp:
            self.viterbi = pickle.load(inp)


def create_float_defaultdict():
    return defaultdict(float)