from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import time
import tweepy
import os

#consumer key, consumer secret, access token, access secret.
# ckey = 'zc7f3iKjDkeJYCdbEhfKQJ7bU'
# csecret = 'pQKhuzZkRJ0sJ1bHevnkR42qh4UGW4dxLw3FGzgoVSSPXUzmGQ'
# atoken ='333587045-PRmu0YPeMFoEBYCQi9gk4OGRGr9MkLx4aLs45rHj'
# asecret ='vmmuJ0KjEQ6nsARCm8zjcfNCbRN9YKRr9at2edD8OWKBB'
#
# # Andrews keys
# ckey2 = 'yl46dC10WaatQB42hRiEJCqP5'
# csecret2 = 'WojDaaxJ6KSDMj1st8HafS1fUpAIqbjPx5z0Vi8gmxI3wwPj7y'
# atoken2 = '2723619545-zWtCxzy2KU7GtgSAM1XAQzIXYxYl18u1AaMcTmE'
# asecret2 = 'O9hzrh8QagP5Lyw5SEIWeaLgtAOQGKMvnSTaG6uxTRI5z'
#
# #aoifes keys
# ckey3 = 'nyIehCLVFTDdaHj46xvjt3tbW'
# csecret3 = 'RLpJzEpdz6faOLk0eVn6HThIEb7V7Pr7mnajmpVKQobtvUXRL5'
# atoken3 = '378217041-wT8dPKMD1puQEBCMOSQ5ITmBPweOK2cCmIkXnNUg'
# asecret3 = '53no1fgZlTuuip7pCW1Sak3YAnrHykL3jpA28yUauPMJ0'

ckeys= ['zc7f3iKjDkeJYCdbEhfKQJ7bU','yl46dC10WaatQB42hRiEJCqP5','nyIehCLVFTDdaHj46xvjt3tbW']
csecrets = ['pQKhuzZkRJ0sJ1bHevnkR42qh4UGW4dxLw3FGzgoVSSPXUzmGQ','WojDaaxJ6KSDMj1st8HafS1fUpAIqbjPx5z0Vi8gmxI3wwPj7y','RLpJzEpdz6faOLk0eVn6HThIEb7V7Pr7mnajmpVKQobtvUXRL5']
atokens = ['333587045-PRmu0YPeMFoEBYCQi9gk4OGRGr9MkLx4aLs45rHj','2723619545-zWtCxzy2KU7GtgSAM1XAQzIXYxYl18u1AaMcTmE','378217041-wT8dPKMD1puQEBCMOSQ5ITmBPweOK2cCmIkXnNUg']
asecrets = ['vmmuJ0KjEQ6nsARCm8zjcfNCbRN9YKRr9at2edD8OWKBB','O9hzrh8QagP5Lyw5SEIWeaLgtAOQGKMvnSTaG6uxTRI5z','53no1fgZlTuuip7pCW1Sak3YAnrHykL3jpA28yUauPMJ0']

def getTweet(id):
    try:
        tweet = api.get_status(id)
        return tweet.text
    except tweepy.TweepError as e:
        print('Failed to retrieve tweet with ID: ',id,' ' ,e.reason)
        if(e.reason.__contains__('Rate limit exceeded')):
            return 'Re-Authenticate'

    # while()
#

hm_lines=18201

def authenticate(account):
    # print "RE_AUTHENTICATE_REAUTHENTICATE"
    # if(getTweet(id)!=None):
    if(account>2):
        account = 0
        time.sleep(60)
        authenticate(account)
    auth = tweepy.OAuthHandler(ckeys[account], csecrets[account])
    auth.set_access_token(atokens[account], asecrets[account])
    api = tweepy.API(auth)
    return api

#TODO MAKE VARIABLE FOR GETTWEET CHECK
Sarcasmset='/Users/FelixDSantos/LeCode/DeepLearning/fyp/Data/Cleaned/sarcasm_tweets.txt'
TweetOnly='/Users/FelixDSantos/LeCode/DeepLearning/fyp/Data/Cleaned/TweetOnly.txt'

auth = tweepy.OAuthHandler(ckeys[0], csecrets[0])
auth.set_access_token(atokens[0], asecrets[0])
api = tweepy.API(auth)
with open(Sarcasmset, 'r') as f:
    # header= next(f)
    with open(TweetOnly, 'a') as newappend:
        if (os.path.getsize(TweetOnly) == 0):
            newappend.write("Tweet\t\tSarcasm")
            newappend.write("\n")
        account=1
        # authenticate(account)
        for line in f:
            words = line.split(",")
            # result= 'Re-Authenticate'
            tweetid=words[1].replace("\n","")
            result = getTweet(tweetid)
            while(result=='Re-Authenticate'):
                account=account+1
                # account=authenticate(account)
                # print '***********Re-Authenticating to: ', ckeys[account], ' '
                api = authenticate(account)
                print('***********Re-Authenticating*********** ',api.auth.consumer_key)
                result=getTweet(tweetid)
            if(result!= None):
                tweet = result
                label= words[0]
                newappend.write(tweet+'\t\t'+label)
                newappend.write("\n")
    newappend.close()
f.close()
