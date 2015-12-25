""" 
Created : 4/12/2015
Updated : 13/12/2015
@author: Mikhail Molotkov 
"""
import numpy as np
from autocorrect import spell
import scipy.misc
import string

def correct_word(x):
    """ Function returns the word, which is most similar to the input word
       x - input word for which function finds similar word """
    if x in word_dict: # Check if the word are correct and in the set
        return x
    else:              # Otherwise find closest match
        return spell(x)

def extract_data(npy_file, dat_file):
    """ Extracts pixels from .npy file according to information in .dat file.
    Returns an array where each row represents a pixels of letter
    and array of correcponding labels
    npy_file - path to .npy file
    dat_file - path to .dat file """   
    page = np.load(npy_file)
    letters = np.genfromtxt(dat_file)
    
    page_height = page.shape[0] # Define a height of a page
    n_rows = int(letters.shape[0])
    
    width, height = 39,51# Width and height of a box, where I place each letter
    n_columns = width*height
    pixels = np.zeros((n_rows, n_columns))
    labels = np.ndarray((2, n_rows))
    
    # Detect a boundary box around letter; move corresponding pixels into array
    for rows in xrange(n_rows):
        temp_pixels = np.zeros((width, height))
        x1 = int(letters[rows, 1])
        y1 = page_height - int(letters[rows, 4])
        x2 = int(letters[rows, 3])
        y2 = page_height - int(letters[rows, 2])
        let_pix=page[y1:y2, x1:x2];
        temp_pixels[1:1+let_pix.shape[0], 1:1+let_pix.shape[1]] = let_pix
        pixels[rows, :] = temp_pixels.flatten()
    
    # Extract labels and letter's state
    with open(dat_file) as f:
        for index, line in enumerate(f):
            x = line.split()           
            let = x[0]
            labels[0,index] = dictionary[let]
            labels[1,index] = x[5]
            
    return pixels, labels

def performPCA(data, n_feat):
    """ PCA algorithm
        data - data set
        n_feat - number of features to be returned """
    covx = np.cov(data, rowvar=0)
    n_data = covx.shape[0]
    w, v = scipy.linalg.eigh(covx, eigvals=(n_data-n_feat, n_data-1))
    v = np.fliplr(v)    
    return v

def classify(train, train_labels, test, test_labels):
    """ Nearest neighbour classification
       train - train data set
       train_labels - corresponding labels for train data set
       test - test data set
       test_labels - corresponding labels for test data set """
    # Reduce amount of features to desired one
    features=np.arange(0, train.shape[1])
    train = train[:, features]
    test = test[:, features]
    
    # Nearest neighbour implementation
    x= np.dot(test, train.transpose())
    modtest=np.sqrt(np.sum(test*test,axis=1))
    modtrain=np.sqrt(np.sum(train*train,axis=1))
    dist = x/np.outer(modtest, modtrain.transpose()) # cosine distance
    nearest=np.argmax(dist, axis=1)
    label = train_labels[0,nearest]
    
    score = (100.0 * sum(test_labels[0,:]==label))/label.shape[0]
    
    # Construct classifier output
    output = ""
    word = ""
    for index, letter in enumerate(label):
        if test_labels[1,index]==0:
            word += sorted(dictionary.keys())[int(letter)-1]
        else:
            word += sorted(dictionary.keys())[int(letter)-1]
            #print word
            word = correct_word(word.lower())
            output = output + word + " "
            word = ""

    return score, output

# Create a dictionary of letters
dictionary =dict(zip(string.letters,[ord(c)%32 for c in string.letters]))

# Create a set of all English words
with open("data/wordsEn.txt") as line:
    word_dict = set(word.strip() for word in line)