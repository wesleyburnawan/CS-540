import os
import math

#These first two functions require os operations and so are completed for you
#Completed for you
def load_training_data(vocab, directory):
    """ Create the list of dictionaries """
    top_level = os.listdir(directory)
    dataset = []
    for d in top_level:
        if d[-1] == '/':
            label = d[:-1]
            subdir = d
        else:
            label = d
            subdir = d+"/"
        files = os.listdir(directory+subdir)
        for f in files:
            bow = create_bow(vocab, directory+subdir+f)
            dataset.append({'label': label, 'bow': bow})
    return dataset

#Completed for you
def create_vocabulary(directory, cutoff):
    """ Create a vocabulary from the training directory
        return a sorted vocabulary list
    """

    top_level = os.listdir(directory)
    vocab = {}
    for d in top_level:
        subdir = d if d[-1] == '/' else d+'/'
        files = os.listdir(directory+subdir)
        for f in files:
            with open(directory+subdir+f,'r', encoding = 'utf-8') as doc:
                for word in doc:
                    word = word.strip()
                    if not word in vocab and len(word) > 0:
                        vocab[word] = 1
                    elif len(word) > 0:
                        vocab[word] += 1
    return sorted([word for word in vocab if vocab[word] >= cutoff])

#The rest of the functions need modifications ------------------------------
#Needs modifications
def create_bow(vocab, filepath):
    """ Create a single dictionary for the data
        Note: label may be None
    """
    bow = {}
    # TODO: add your code here
    with open(filepath, 'r', encoding = 'utf-8') as f:
        for word in f:
            word = word.strip()
            if not word in bow and (word in vocab):
                bow[word] = 1
            elif not word in bow and (word not in vocab) and (None not in bow):
                bow[None] = 1
            elif word in bow:
                bow[word] += 1
            else:
                bow[None] += 1
    return bow

#Needs modifications
def prior(training_data, label_list):
    """ return the prior probability of the label in the training set
        => frequency of DOCUMENTS
    """

    smooth = 1 # smoothing factor
    logprob = {}
    # TODO: add your code here
    for l in label_list:
        logprob[l] = 0
        number = 0
        for d in training_data:
            if(d['label'] == l):
                number += 1
        logprob[l] = math.log(number+smooth) - math.log((len(training_data) + 2))
    return logprob

#Needs modifications
def p_word_given_label(vocab, training_data, label):
    """ return the class conditional probability of label over all words, with smoothing """

    smooth = 1 # smoothing factor
    word_prob = {}
    # TODO: add your code here
    word_prob[None] = 0
    wc = 0
    v = len(vocab)
    for d in training_data:
        for f in d['bow']:
            if(d['label'] == label):
                wc += d['bow'][f]
    for l in training_data:  
        for word in l['bow']:
            if (l['label'] == label):
                if word in vocab and word not in word_prob:
                    word_prob[word] = l['bow'][word]
                elif word in vocab and word in word_prob:
                    word_prob[word] += l['bow'][word]
                else:
                    word_prob[None] += l['bow'][word]
            else:
                if word in vocab and word not in word_prob:
                    word_prob[word] = 0
    for e in word_prob:
        word_prob[e] = math.log(word_prob[e] + smooth) - math.log(wc + v + smooth)
    return word_prob


##################################################################################
#Needs modifications
def train(training_directory, cutoff):
    """ return a dictionary formatted as follows:
            {
             'vocabulary': <the training set vocabulary>,
             'log prior': <the output of prior()>,
             'log p(w|y=2016)': <the output of p_word_given_label() for 2016>,
             'log p(w|y=2020)': <the output of p_word_given_label() for 2020>
            }
    """
    retval = {}
    label_list = os.listdir(training_directory)
    # TODO: add your code here
    vocab = create_vocabulary(training_directory, cutoff)
    retval['vocabulary'] = vocab
    training = load_training_data(vocab, training_directory)
    retval['log prior'] = prior(training, ['2020', '2016'])
    retval['log p(w|y=2016)'] = p_word_given_label(vocab, training, '2016')
    retval['log p(w|y=2020)'] = p_word_given_label(vocab, training, '2020')
    return retval

#Needs modifications
def classify(model, filepath):
    """ return a dictionary formatted as follows:
            {
             'predicted y': <'2016' or '2020'>,
             'log p(y=2016|x)': <log probability of 2016 label for the document>,
             'log p(y=2020|x)': <log probability of 2020 label for the document>
            }
    """
    retval = {}
    # TODO: add your code here
    bow = create_bow(model['vocabulary'], filepath)
    
    label2016 = 0
    for d in model['log p(w|y=2016)']:
        if d in bow:
            label2016 += model['log p(w|y=2016)'][d] * bow[d]
    retval['log p(y=2016|x)'] = label2016 + model['log prior']['2016']

    label2020 = 0
    for d in model['log p(w|y=2020)']:
        if d in bow:
            label2020 += model['log p(w|y=2020)'][d] * bow[d]
    retval['log p(y=2020|x)'] = label2020 + model['log prior']['2020']

    if retval['log p(y=2020|x)'] > retval['log p(y=2016|x)']:
        retval['predicted y'] = '2020'
    else:
        retval['predicted y'] = '2016'
    return retval

