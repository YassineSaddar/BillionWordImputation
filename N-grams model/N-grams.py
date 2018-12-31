########################################################################################
# Author: Tianlong Song
# Name: LocationAndFillingTestData.py
# Description: Locating and filling the missing word, working on test data
# Date created: 04/20/2015
########################################################################################
import numpy
import re

#---------------------------------------------------------------------------------------
# Function: Main function
#---------------------------------------------------------------------------------------
def main():
    # Global settings
    trainSize = 1500000
    testSize = 306681
    M = 1       # Maximum distance considered
    gamma = 0.01
    
    # Initialization
    featureTable = {}   # Bigram with particular distance
    triGramTable = {}   # Trigram table
    
    # Statistics collection
    f = open('data/train-cl.txt','r')
    cnt = 0
    while cnt<trainSize:
        line = f.readline()
        cnt = cnt + 1
        words = line.split()
        for i in range(0,len(words)):
            for m in range(0,M+1):
                if i+m+1>=len(words):
                    break
                key = words[i] + ' ' + words[i+m+1] + ' ' + str(m)
                if key in featureTable:
                    featureTable[key] = featureTable[key] + 1
                else:
                    featureTable[key] = 1
            if i<=len(words)-3:
                key = words[i] + ' ' + words[i+1] + ' ' + words[i+2]
                if key in triGramTable:
                    triGramTable[key] = triGramTable[key] + 1
                else:
                    triGramTable[key] = 1
    print ('Feature table size: ' + str(len(featureTable)))
    print ('Trigram table size: ' + str(len(triGramTable)))
    f.close()
    
    # Missing word location and filling
    f = open('data/test-cl.txt','r',encoding="UTF-8")
    fR = open('data/result.txt','w',encoding="UTF-8")
    line = f.readline()
    fR.write('id,"sentence"\n')
    cnt = 1
    while cnt<=testSize:
        line = f.readline() 
        
        # Sentence preprocessing
        words = line.split()
        splitted = re.split(',',words[0])
        words[0] = splitted[1]
        words[len(words)-1] = words[len(words)-1][:-1]
        wordsOriginal = words[:]
        wordsOriginal[0] = wordsOriginal[0].replace('"','')
        
        # Missing word location
        score = numpy.zeros(len(words)-1)
        print(score)
        for k in range(1,len(words)):
            key = words[k-1] + ' ' + words[k] + ' ' + str(0)
            if key in featureTable:
                numNeg = featureTable[key]
            else:
                numNeg = 0
            key = words[k-1] + ' ' + words[k] + ' ' + str(1)
            if key in featureTable:
                numPos = featureTable[key]
            else:
                numPos = 0
            if numNeg+numPos!=0:
                if words[k-1]=='UNKA' or words[k]=='UNKA':
                    score[k-1] = 1.0*numPos/(numNeg+numPos) - 1.0*numNeg/(numNeg+numPos)
                else:
                    score[k-1] = 1.0*numPos**(1+gamma)/(numNeg+numPos) - 1.0*numNeg**(1+gamma)/(numNeg+numPos)
        location = numpy.argmax(score) + 1
        
        # Missing word filling
        maxWord = '___'
        wordsOriginal.insert(location,maxWord)
                
        # Write the completed sentence to file
        fR.write(str(cnt)+',"')
        for k in range(0,len(wordsOriginal)-1):
            fR.write(wordsOriginal[k]+' ')
        fR.write(wordsOriginal[len(wordsOriginal)-1]+'"\n')
        cnt = cnt + 1
        
    fR.close()
    f.close()
    print ('Missing word location and filling done') 
    
if __name__ == '__main__':
    main()
