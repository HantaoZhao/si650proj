from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.callbacks import EarlyStopping
import datetime as dt
import urllib.request, json
import os
import numpy as np
import tensorflow as tf # This code has been tested with TensorFlow 1.6
from sklearn.preprocessing import MinMaxScaler
import re
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))


def clean_text(text):

    text = text.lower()  # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ',
                                   text)  # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('',
                              text)  # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing.
    text = text.replace('x', '')
    #    text = re.sub(r'\W+', '', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # remove stopwors from text
    return text


df = pd.read_csv("Gungor_2018_VictorianAuthorAttribution_data-train.csv",engine ="python")
# print(clean_text(df["text"].iloc[0]))
df["text"] = df["text"].apply(clean_text)
MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 100
# This is fixed.
EMBEDDING_DIM = 50
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['text'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


X = tokenizer.texts_to_sequences(df['text'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)


Y = pd.get_dummies(df['author']).values
print('Shape of label tensor:', Y.shape)


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.1, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)




model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(MAX_SEQUENCE_LENGTH, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(30, activation='relu'))
model.add(Dense(45, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


epochs = 10
batch_size = 100

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])


accr = model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
# import os.path
# from gensim import corpora
# from gensim.models import LsiModel
# from nltk.tokenize import RegexpTokenizer
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# from gensim.models.coherencemodel import CoherenceModel
# import matplotlib.pyplot as plt
#
#
# def load_data():
#     documents_list = []
#     docs = ["nt it seems te me how much money is he worth and that within a reasonable margin i can tell you as for i private character most likely it is like that of other men which means that the less you investigate it th happier you will be private character indeed what have the men of to day to do with things miss made a pretty picture as she sat there in mrs s sitting room she was a pro of perhaps one and twenty i that s the of good height and with one of those well rounded figures which would better please did they not arouse fears of too great in the distant future at present she was as nearly perfect a as it it possible to imagine her mass of straw tinted hair was arranged in a manner that would have become a queen her head was poised on her she had a bright color thai was all her own and her excessive vivacity became her well mrs was a widow of about forty years of age who might once have been beautiful but who now seemed to feel the cares of life too heavily to mind the deep which time had hastened to place in the lines of her pale countenance she had a certain air of dignity as one who had seen better days and could rise in thought at least above the misfortunes of recent years the house which she occupied was situated within pistol shot of college in the city of cambridge it was a comfortable structure set back from the street and almost hidden from the gaze of by the shade trees and very tall hedge which bordered the her support came almost wholly from the letting of rooms to students from which source she managed to support in a modest manner herself and two daughters she listened with much interest to miss s and then said he is rich then you may set that down as assured when i have promised to marry him smiled the young lady his grandfather was one of those mill owners who made such a pile of money fifty years or so ago he left it all to when he was a baby that ia to his of and it has kept oc grow w t ay and growing as such fortunes do oh yes do is rich enough but as for his character that s quite thing i haven t got as far as that yet the speaker paused and looked at her companion if the troubles of this world had mad much impression on her young life there was no outward evidence of it a big yellow cat which sat on the window seat a few yards away seemed quite as much worried by either past or present as she yet this young lady had a history which would make the foundation of a romance an air of mystery pervaded her life which no one seemed able to penetrate she had her moods too and the of one of them seemed very little like the of another mrs had known her when a child her father captain arthur was a cambridge man and when her mother died mrs saw much of the pretty little orphan girl until the captain took her out to south america whence he never returned one of the first things did after coming back a grown young lady was to seek out mrs s house the old friendship was renewed and as is often the case it gained strength from the fact that the two ladies were so totally as well as from other causes which will appear later on i am glad indeed that you are to be so happy said the widow money is not a thing to be despised and those who pretend to so consider it art only striving for effect i have often felt the need of t since my husband died partly for myself but much for and money will do a great deal to make life pleasant but the first requisite in a mat partner should be private character art not cannot comprehend this fully you are fa young you remember henry there were thing about him which i could have wished different but ae one ever his private character when th bank officials found that he had fled with the books in such bad shape and thirty thousand dollars gone it was a blow to me the news which followed so soon of his death abroad was very hard to bear my only consolation in my distress was the reflection that whatever he might have done his private character was there was a momentary suspicion of amusement in miss s deep blue eyes but she mastered it before it attracted her friend s attention and said i am sorry to say i can t agree with you if try husband should ever run away he might take all the women in america with him for all of me but if he forgot to leave me a handsome pile of cash i d never forgive him never i have all right on that score in advance the day we are married he is to give me fifty thousand dollars that ll make a sure foundation for me in case anything should happen ah mrs if every husband would do as well how much more married happiness there would be miss laughed at her own and the elder lady s features relaxed a little and you have been engaged for mere than a year said mrs yes and i would make it ten if i is if dared for he might have heart disease and go off suddenly i could get him to give me the money now but that would have a disagreeable look i think would wait for me a century if i compelled ms i mob that s tub them sum",
# "to talk about why you heard of such a case as his in your life tell me all about it said mrs her chair an inch closer tell me everything i want the whole truth miss colored a little and waited some seconds before replying then she said well i will for i know what i tell you will go no further and i really should like to tell some one you won t judge me harshly i m sure for you ve known me ever since i was smaller than and i ve always regarded you just like a mother but really i fear my story will shock even you she paused m the telling of it can do no harm said the other lady evidently feeling the curiosity for which her sex is noted becoming stronger if you have done anything i may be able to advise you and prevent its repetition the young lady laughed silently and gave her companion a sly look well here it is she began at last summer was introduced to him for the first time there was a party and i had on the suit imaginable a costume just me off and i knew the minute i glanced around the boat that i had no there i noticed standing with the who seemed to know him and as he was by far the best looking man i an elderly gentleman in a quiet way who he was why that s he replied of the best fellows as well as richest here this season let me present you it was an is hi s that i mrs that led me to decline that old s offer mrs looked much puzzled you declined to be introduced right the first time said miss m tt i hadn t declined i should never have been to day the blushing that you see before you mrs shook her head your are too hard for me did you ever catch from a boat under full sail asked the young lady yes you cast out your line with a spoon hook and as fast as you can through the waves the see the shining thing they for it aad swallow it bait hook spoon and all well i was the spoon hook and the mrs caught a short breath and said u ah i thought you would saw i did not want an introduction and that made him all the more anxious to get one he got the ear of the old gentleman to whom i had spoken of him i couldn t hear them but i knew exactly what they were saying though i had my eyes on a ship a mile away all tht time who is that pretty girl asked mr i m not vain but i m not a fool either and i ve looked in the glass i know he said pretty them the old man answered her name is aad she is staying with the at their cottage then they waited a minute then mr said introduce me and the old man bless his heart ml that thi replied i can t she has declined it already at this mr opened his big eyes and was disposed to get angry then for a little while he affected not to care and then then he made up his contrary masculine mind that he would make my acquaintance whether i liked it or no and see what sort of young woman it was who had given him the first of his dainty life miss stopped to take breath and looked triumphantly at her interested listener shall i go on she asked or shall i continue this narrative the next time i come over i shall have to stop and think up some of it before i can make a connected story it s more than a year since it happened you know go on by all means said mrs i am quite entertained well that very evening after dinner i strolled out totally unconscious of any other person s existence you understand and took my way idly along the sand there was a brisk breeze blowing and i could only keep my hat on by holding to the ribbons my front hair blew around my face and my dress was nearly there came a terrible moment i felt my skirts and i had to choose quickly between two evils so i let go of my hat and it sailed through the air a gentleman strolling some rods behind caught it as as the best league could have caught a ball and hastened to bring it to me you never could guess who not began mrs but odd as it seems it was i had secured control of my skirts by the time he came up and was able to take my hat place it on my head and thank him in m n i that i th a cold formal manner i tied the strings tightly under my chin this time as i had no further use for a flying hat and started on my way but another gust came along at that exact moment you never saw such a gust thought i should go up in the air like an it was terrible my face was enveloped in my dress skirt in a second so i never can testify from actual knowledge the extent of the ruin but it was enough and that terrible man caught my dress and pulled it down just as if it was any of his business and the first thing i saw when my burning face was restored to the light was his royal bowing profoundly and hoping i was now quite able to proceed on my way can you conceive of anything more it was indeed admitted mrs what did you do what did i do i turned on that man with all the i could muster stamped",
# "my foot on the ground and said i believe you did that on purpose you re quite right he replied i brought the wind along here just at the moment i knew you were coming i did it i admit but if you ll forgive me i ll never do it again now what could one say to a man like that r the face of a beautiful child of ten years peeped la at the doorway at this juncture she had heavy dark hair hanging loosely about her head in masses i only want mamma she said going to toe window and taking the big cat in her arms we are playing school and there aren t scholars enough we want to sit in one of the chain m that s it the last explanation was made for miss i benefit and that young lady signified her approval at the scheme by a smile and nod you re going to stay to dinner continued child as she held the door cause would be very much disappointed if you didn t yes said mrs speaking for her visitor will surely be here to dinner she is going to stay several days run along now we are very busy with our conversation the child hesitated a moment longer something little girls shouldn t hear i s pose she said wisely it s always the way i shall be s glad when i am big enough to hear everything she off with the cat and miss proceeded what a lovely child is let me see where was i oh yes well i drew myself up with what dignity i could command for i still had a wholesome fear of what the next gust might do and i said in withering tones only they didn t seem to him at all you are very impertinent sir i wish you a good evening then i marched off home without looking around and as luck would have it without further accident mrs waited for her guest to proceed seeming qui e absorbed in the story she was hearing if you can conceive of anything more than that in the way of making an acquaintance continued miss i would like to know it i hid in the house for three days overcome with the occurrence and then only ventured out but mr was not the sort of youth to be discouraged by that s t little things he happened as it turned out to know colonel and it was easy enough to get asked up to the cottage one day i walked out on the and came upon him and the colonel where there was no escape this is mr mr miss and it was done bore himself remarkably well at that time as for me i must have looked like a red red rose the good colonel saw nothing these ancient military men never do and the usual followed in a few minutes a messenger came post haste for him and he excused himself in the manner mr said he miss will entertain you i m sure till my return i sha n t be over an hour and off he flew i looked rather helplessly after him and then as there seemed no remedy took a chair and sat looking at the sea for the next ten minutes without a word goodness exclaimed mrs how could you how could i do otherwise he had no right to e there at all he should have relieved my embarrassment by taking himself out of the way i gave him ten minutes to do it and that ten minutes cost him his freedom for life i didn t as if i was thinking of anything in particular as i sat there with my eyes on the ocean but i was a plan and i carried it out to the letter a plan yes a plan to make that fellow come to mo oa his knees i knew he was laughing at every minute and i object of all things to being n f that i the finally i turned my chair about and looked him full in the eyes yes he was laughing just as i suspected colonel spoke the truth he said in response to my look he said you would entertain me and you do immensely indeed that is much more than you do lot me r i answered at that he burst out laughing and i felt the wrinkles disappearing in spite of all i could do come he said holding out his hand let s be friends why we ve got to there are reasons i could have swept into the house and left him but that would not have you were on the the other day he pursued and you avoided me we walked on the shore and the wind my cause i came to this and that lucky message has aided me i wish to talk to you miss you would do me the greatest favor by leaving me i said you ought to see that your presence me your conduct this statement which might otherwise seem rude but you heard me promise colonel that i would wait till his return he said his eyebrows then will go i answered half rising of course i had no intention of going no he replied making a movement to me he left you here to entertain his guest and you mil never let me tell him you disregarded his wish then mrs i made a heroic effort i laughed as heartily as i could and professed to have the whole affair as a joke hi a few moments we were like old friends th if that s returned we were talking of a hundred things and knew i had my fish all ready whenever i chose to jerk the line opened the door to inquire whether they would take dinner at the usual"
# ]
#     for doc in docs:
#         text = doc.strip()
#         documents_list.append(text)
#     return documents_list
#
# def preprocess_data(doc_set):
#
#     tokenizer = RegexpTokenizer(r'\w+')
#     # create English stop words list
#     en_stop = set(stopwords.words('english'))
#     # Create p_stemmer of class PorterStemmer
#     p_stemmer = PorterStemmer()
#     # list for tokenized documents in loop
#     texts = []
#     # loop through document list
#     for i in doc_set:
#         # clean and tokenize document string
#         raw = i.lower()
#         tokens = tokenizer.tokenize(raw)
#         # remove stop words from tokens
#         stopped_tokens = [i for i in tokens if not i in en_stop]
#         # stem tokens
#         stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
#         # add tokens to list
#         texts.append(stemmed_tokens)
#     return texts
#
#
#
#
# doc_clean = preprocess_data(load_data())
# dictionary = corpora.Dictionary(doc_clean)
# doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
# lsamodel = LsiModel(doc_term_matrix, num_topics=2, id2word = dictionary)  # train model
# print(lsamodel.print_topics(num_topics=2))