"""
Created : 4/12/2015
Updated : 12/12/2015
@author: Mikhail Molotkov
"""
import numpy as np
import Assignment2 as ocr

def main():
    train1, train_labels1 = ocr.extract_data('data/train1.npy','data/train1.dat') 
    train2, train_labels2 = ocr.extract_data('data/train2.npy','data/train2.dat')
    train3, train_labels3 = ocr.extract_data('data/train3.npy','data/train3.dat')
    train4, train_labels4 = ocr.extract_data('data/train4.npy','data/train4.dat')    
    trains = np.vstack((train1,train2,train3,train4))
    train_labels = np.hstack((train_labels1, train_labels2, train_labels3, train_labels4))
    
    print "Trial1: Standard evaluation in process..."
    for i in xrange(2):
        print "Evaluation of test"+str(i+1)+".npy"
        test, test_labels = ocr.extract_data('data/test'+str(i+1)+'.npy', 'data/test'+str(i+1)+'.dat')
        score,text = ocr.classify(trains, train_labels, test, test_labels)
        print "The score of the classifier for this test page is:\n",score
        print "The output text is:\n",text
        print "----------------------------------"
    print "Trial1 is finished.\n"
    
    print "Trial2: Noise robustness in process..."
    for i in xrange(2):
        for j in xrange(4):
            print "Evaluation of test"+str(i+1)+"."+str(j+1)+".npy"
            test, test_labels = ocr.extract_data('data/test'+str(i+1)+"."+str(j+1)+'.npy', 'data/test'+str(i+1)+'.dat')
            score,text = ocr.classify(trains, train_labels, test, test_labels)
            print "The score of the classifier for this state of test page is:\n",score
            print "----------------------------------"    
        print "**********************************"
    print "Trial2 is finished.\n"
    
    print "Trial 3: Dimensionality reduction in process..."
    pca_data = ocr.performPCA(trains,10)
    pca_train = np.dot((trains - np.mean(trains)), pca_data)
    print "Clean data(without noise)..."
    for i in xrange(2):
        print "Evaluation of test"+str(i+1)+".npy"
        test, test_labels = ocr.extract_data('data/test'+str(i+1)+'.npy', 'data/test'+str(i+1)+'.dat')
        pca_test = np.dot((test - np.mean(test)), pca_data)
        score,text = ocr.classify(pca_train, train_labels, pca_test, test_labels)
        print "The score of the classifier for this test page is:\n",score
        print "----------------------------------"
    print "Noisy data..." 
    for i in xrange(2):
        for j in xrange(4):
            print "Evaluation of test"+str(i+1)+"."+str(j+1)+".npy"
            test, test_labels = ocr.extract_data('data/test'+str(i+1)+"."+str(j+1)+'.npy', 'data/test'+str(i+1)+'.dat')
            pca_test = np.dot((test - np.mean(test)), pca_data)
            score,text = ocr.classify(pca_train, train_labels, pca_test, test_labels)
            print "The score of the classifier for this state of test page is:\n",score
            print "----------------------------------"    
        print "**********************************"   
main()