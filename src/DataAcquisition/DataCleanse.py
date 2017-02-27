import os
import re

# stringwithhashtag = "this is a string with a#hashtag"
# stringwithhashtag1 = "this is a string#this wi#hashtagth a#hashtag"
# **Cleaning Ashwin twitter dataset**
def removeHashtagsFromLine(words):
    i=0
    while(i<len(words)):
        if(words[i][0]=='#'):
            words.remove(words[i])
            i-=1
        elif('#' in words[i]):
            positionofhashtag= words[i].index('#')
            amounttoremove = len(words[i])-positionofhashtag
            print(amounttoremove)
            words[i]= words[i][:-amounttoremove]
        i+=1
    # print(words)
    return words

# newstring=removeHashtagsFromLine(stringwithhashtag1.split())

def removeHashtagsFromFile(inputFile, outputFile):
    with open(inputFile, 'r') as f:
        header = next(f)
        with open(outputFile, 'a') as newappend:
            if (os.path.getsize(outputFile) == 0):
                newappend.write("Tweet\t\tSarcasm")
                newappend.write("\n")
            for line in f:
                words = line.split()
                sizeOfstring = len(words)
                if (words[sizeOfstring - 1] == '0' or words[sizeOfstring - 1] == '1'):
                    label = words[sizeOfstring - 1]
                    words.remove(label)
                    nohashtags = removeHashtagsFromLine(words)
                    tweet = ' '.join(nohashtags)
                    tweetAndLabel = tweet + "\t\t" + label
                    newappend.write(tweetAndLabel)
                    newappend.write("\n")
                else:
                    print(line)
        newappend.close()
    f.close()

def removeNewLinesFromFile(inputFile, outputFile):
    # if the file has any unnecessary new lines, we remove them
    # example:
    # @ii_Loveyou_x @CSRytonNufc
    # Wow, you are nice.
    # Did you study at Cambridge or Harvard?
    ##sarcasm		1
    # this is one tweet but it has been separated into multiple lines because of instances of new lines within the tweet
    # using this function when the file containing these is read, it should take make a line with this tweet in only one line
    # @ii_Loveyou_x @CSRytonNufc Wow, you are nice. Did you study at Cambridge or Harvard? #sarcasm		1
    with open(inputFile, 'r') as f:
        header = next(f)
        with open(outputFile, 'a') as newappend:
            if (os.path.getsize(outputFile) == 0):
                newappend.write("Tweet\t\tSarcasm")
                newappend.write("\n")
            for line in f:
                words = line.split()
                sizeOfstring = len(words)
                if (sizeOfstring == 0 and '\n' in line):
                    line.replace('\n', '')
                elif (words[sizeOfstring - 1] == '0' or words[sizeOfstring - 1] == '1'):
                    newappend.write(line)
                else:
                    stringofline = line
                    stringofline = stringofline.replace('\n', ' ')
                    newappend.write(stringofline)
        newappend.close()
    f.close()


def removeHttpLinks(inputFile, outputFile):
    with open(inputFile, 'r') as f:
        header = next(f)
        with open(outputFile, 'a') as newappend:
            if (os.path.getsize(outputFile) == 0):
                newappend.write("Tweet\t\tSarcasm")
                newappend.write("\n")
            for line in f:
                if 'http' in line:
                    print("**********Before Removal**********\n", line, "\n")
                    httpremoved = re.sub(r"http\S+", "", line)
                    newappend.write(httpremoved)
                    print(httpremoved)
                else:
                    newappend.write(line)
        newappend.close()
    f.close()



DataToClean='/Users/FelixDSantos/LeCode/DeepLearning/fyp/Data/TweetOnly.txt'

def main():
    # removeNewLinesFromFile(DataToClean, "Data/Cleaned/sarcasmdataset_nonewlines.txt")
    removeHashtagsFromFile("/Users/FelixDSantos/LeCode/DeepLearning/fyp/Data/Cleaned/SarcasmDataset_Final.txt", "/Users/FelixDSantos/LeCode/DeepLearning/fyp/Data/Cleaned/SarcasmDataset_Final.txt1")
    # removeHttpLinks("Data/Cleaned/sarcasmdataset_removedhashtags.txt", "Data/Cleaned/SarcasmDataset_Final.txt")
if __name__ == "__main__": main()
# testString= "@test Hello this is a tweet, i guess you could say it's!        #sarcastic #sarcasm        0"
# testString1= "@jamietoomeylive when we were playing I was thinkin that xD		0"
# words=testString1.split()
# sizeOfstring = len(words)
# label = words[sizeOfstring-1]
# words.remove(label)
# nohashtags = removeHashtags(words)
# nohashtags =' '.join(nohashtags)
# nohashtags = nohashtags +"\t\t"+label
# print(nohashtags)

# newstring= ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",testString).split())
# newstring= ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)","",testString).split())
# print(newstring)
