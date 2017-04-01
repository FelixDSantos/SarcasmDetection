from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import time
import tweepy
import os
import itertools

#consumer key, consumer secret, access token, access secret.
ckey = 'zc7f3iKjDkeJYCdbEhfKQJ7bU'
csecret = 'pQKhuzZkRJ0sJ1bHevnkR42qh4UGW4dxLw3FGzgoVSSPXUzmGQ'
atoken ='333587045-PRmu0YPeMFoEBYCQi9gk4OGRGr9MkLx4aLs45rHj'
asecret ='vmmuJ0KjEQ6nsARCm8zjcfNCbRN9YKRr9at2edD8OWKBB'
#

def getTweet(id):
    try:
        tweet = api.get_status(id)
        return tweet.text
    except tweepy.TweepError as e:
        print('Failed to retrieve tweet with ID: ',id,' ' ,e.reason)
        if(e.reason.__contains__('Rate limit exceeded')):
            return 'Sleep'


Sarcasmset='/Users/FelixDSantos/LeCode/DeepLearning/fyp/Data/sarcasm_tweets.txt'
# TweetOnly='/Users/FelixDSantos/LeCode/DeepLearning/fyp/Data/Cleaned/TweetOnly.txt'
TweetOnly='/Users/FelixDSantos/LeCode/DeepLearning/fyp/Data/Cleaned/test.txt'

auth = tweepy.AppAuthHandler(ckey, csecret)
# auth.set_access_token(atoken, asecret)
api = tweepy.API(auth,wait_on_rate_limit=True,
				   wait_on_rate_limit_notify=True)

def tweetIDsToTweettxt(idtext,tweetoutputtext):
    with open(Sarcasmset, 'r') as f:
        # header= next(f)
        with open(TweetOnly, 'a') as newappend:
            if (os.path.getsize(TweetOnly) == 0):
                newappend.write("Tweet\t\tSarcasm")
                newappend.write("\n")
            for line in f:
                words = line.split(",")
                tweetid=words[1].replace("\n","")
                result = getTweet(tweetid)

                while(result=='Sleep'):
                    time.sleep(60)
                    result=getTweet(tweetid)
                if(result!= None):
                    tweet = result
                    label= words[0]
                    newappend.write(tweet+'\t\t'+label)
                    newappend.write("\n")
        newappend.close()
    f.close()

def streamHashtag(hashtag,label,amount):
    Tweets = tweepy.Cursor(api.search, q=hashtag,languages=["en"]).items(amount)
    # listoftweets=[]
    for tweet in Tweets:
        if (not tweet.retweeted) and ('RT @' not in tweet.text) and ('@' not in tweet.text) and ('http' not in tweet.text) and(tweet.lang=='en'):
            yield([tweet.text,label])

    # print("{} {}tweets retrieved.".format(hashtag,len(listoftweets)))
    # return listoftweets
sarcasmtweets=streamHashtag("#sarcasm",1,1000000)
# sarcasmtweets+=streamHashtag("#not",1,200000)
# lensarcasmtweets=sum(1 for x in sarcasmtweets)
print("Successfully retrieved {} tweets".format('#sarcasm'))
sarcasmtweets=itertools.chain(sarcasmtweets,streamHashtag("#not",1,200000))
# lensarcasmtweets=sum(1 for x in sarcasmtweets)
print("Successfully retrieved {} tweets".format('#sarcasm and #not'))
nonsarcasm=streamHashtag("a",0,0)
alltweets=itertools.chain(sarcasmtweets,nonsarcasm)
tweetstream='/Users/FelixDSantos/LeCode/DeepLearning/fyp/Data/stream/streamedtweets3.txt'
print("Writing to file {}".format(tweetstream))
with open(tweetstream, 'a') as newappend:
    if (os.path.getsize(tweetstream) == 0):
        newappend.write("Tweet\t\tSarcasm")
        newappend.write("\n")
    for tweet in alltweets:
        newappend.write(tweet[0] + '\t\t' + str(tweet[1]))
        newappend.write("\n")
    print("Tweets writting to file {}".format(tweetstream))
# new_tweets = api.user_timeline(screen_name = screen_name,count=200)
# tweetIDsToTweettxt(Sarcasmset,TweetOnly)
