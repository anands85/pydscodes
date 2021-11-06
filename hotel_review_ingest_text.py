import numpy as np
import pandas as pd
import datetime
import os
import json
import string
import sklearn
import nltk
import re
import fuzzywuzzy
nltk.download('punkt')
from transformers import pipeline

review_text = '...We stayed a week at the hotel in a front-line room. ' \
              'The "all inclusive" arrangement was a nonsense - a couple of drinks with lunch and a couple of drinks with dinner, ' \
              'all served in the dining room from a limited selection of house wines plus beer; ' \
              'there was a bar where other alcohol could be purchased, but no alcohol was served after 9.30pm. ' \
              '(The hotel claimed this time limit applied in bars, restaurants and other hotels in Mallorca, but this was not the case.)' \
              ' The \'front-line\' room was ok, but nothing special. ' \
              'The dining room was a large cafeteria plus some outside seating. ' \
              'Guests had to collect food from a buffet area where there was no serious attempt to enforce social distancing, nor the one-way system nor the maximum number of guests at any one time in the buffet area. ' \
              'Salads, fruits and cheeses were truly excellent, and there was a wide choice of other foods available too, but meats and fish portions were often overcooked (or in one case meat which was seriously undercooked). ' \
              'There is no luxury feel at any point in the hotel, but it is close to the beach and some lovely bars and cafes on the promenade. ' \
              'It is also close to the airport, but without being able to hear the planes. ' \
              'Staff generally friendly. ' \
              'Reception tried to charge me for the room safe (21 Euros) both on arrival and again on departure...'
print(review_text)
replace_str_lst = [['\.\.\.',' '],
                   ['\"',''],
                   [';','.'],
                   ['\(',''],
                   ['\)','.'],
                   ['\.\.','\.'],
                   ['\'','\"'],
                   ['-',''],
                   ['pre ','pre-']]

#convert to utf8 text - standardize
review_text_utf8 = str(review_text.encode('utf-8','replace'))
print(review_text_utf8)
#convert to lower text
review_text_lower = review_text.lower()
print(review_text_lower)
#remove review type text into standard english text with sentences ans shorter phrases or sub-sentences
review_text_clean = review_text_lower
for pattern,replace_text in replace_str_lst:
    print(pattern)
    review_text_clean = re.sub(pattern,replace_text,review_text_clean)
print(review_text_clean)
#standardize punctuation
punc_list = []
ignore_list = [',','?','.',':']
replace_punc_str = ''
for character in string.punctuation:
    if character not in ignore_list:
        punc_list.append(str(character))
review_text_std= str("".join([i if i not in punc_list else replace_punc_str for i in review_text_clean]))
print(review_text_std)
print('===============================================================')
print('=================== PRE-PROCESS OUTPUT ========================')
print('===============================================================')
#identify sentences in standard form in lower text format
review_text_std_sent_phr = nltk.sent_tokenize(review_text_std,'english')
context_q_n_a = "".join(review_text_std_sent_phr)

nlp_sentiment = pipeline("sentiment-analysis")
nlp_q_n_a = pipeline("question-answering")
count_sent_phr = 1
for sent_phr in review_text_std_sent_phr:
    print(str(count_sent_phr) + '.',sent_phr)
    print('Sentiment',nlp_sentiment(sent_phr))
    count_sent_phr += 1
print("What are the services?",
      nlp_q_n_a(question="What are the services?", context=context_q_n_a))
print("What help are provided at the reception?",
      nlp_q_n_a(question="What help are provided at the reception?", context=context_q_n_a))
print("How are the staff?",
      nlp_q_n_a(question="How are the staff?", context=context_q_n_a))
print("How is the accomodation?",
      nlp_q_n_a(question="How is the accomodation?", context=context_q_n_a))
print("What is the schedule?",
      nlp_q_n_a(question="What is the schedule?", context=context_q_n_a))
print("What are the charges?",
      nlp_q_n_a(question="What are the charges?", context=context_q_n_a))

# 1.  we stayed a week at the hotel in a frontline room.
# Sentiment [{'label': 'NEGATIVE', 'score': 0.5536131262779236}]
# 2. the all inclusive arrangement was a nonsense  a couple of drinks with lunch and a couple of drinks with dinner, all served in the dining room from a limited selection of house wines plus beer.
# Sentiment [{'label': 'NEGATIVE', 'score': 0.995652437210083}]
# 3. there was a bar where other alcohol could be purchased, but no alcohol was served after 9.30pm.
# Sentiment [{'label': 'NEGATIVE', 'score': 0.9976382255554199}]
# 4. the hotel claimed this time limit applied in bars, restaurants and other hotels in mallorca, but this was not the case.
# Sentiment [{'label': 'NEGATIVE', 'score': 0.9891620874404907}]
# 5. the frontline room was ok, but nothing special.
# Sentiment [{'label': 'NEGATIVE', 'score': 0.9915756583213806}]
# 6. the dining room was a large cafeteria plus some outside seating.
# Sentiment [{'label': 'POSITIVE', 'score': 0.9608898758888245}]
# 7. guests had to collect food from a buffet area where there was no serious attempt to enforce social distancing, nor the oneway system nor the maximum number of guests at any one time in the buffet area.
# Sentiment [{'label': 'NEGATIVE', 'score': 0.9984182119369507}]
# 8. salads, fruits and cheeses were truly excellent, and there was a wide choice of other foods available too, but meats and fish portions were often overcooked or in one case meat which was seriously undercooked.
# Sentiment [{'label': 'NEGATIVE', 'score': 0.9947566986083984}]
# 9. there is no luxury feel at any point in the hotel, but it is close to the beach and some lovely bars and cafes on the promenade.
# Sentiment [{'label': 'POSITIVE', 'score': 0.9992615580558777}]
# 10. it is also close to the airport, but without being able to hear the planes.
# Sentiment [{'label': 'NEGATIVE', 'score': 0.9917235970497131}]
# 11. staff generally friendly.
# Sentiment [{'label': 'POSITIVE', 'score': 0.9998335838317871}]
# 12. reception tried to charge me for the room safe 21 euros.
# Sentiment [{'label': 'NEGATIVE', 'score': 0.9954631924629211}]
# 13. both on arrival and again on departure
# Sentiment [{'label': 'POSITIVE', 'score': 0.6849472522735596}]
# What are the services? {'score': 0.2711167335510254, 'start': 121, 'end': 161, 'answer': 'lunch and a couple of drinks with dinner'}
# What help are provided at the reception? {'score': 0.00031074718572199345, 'start': 568, 'end': 613, 'answer': 'guests had to collect food from a buffet area'}
# How are the staff? {'score': 0.5508769154548645, 'start': 1187, 'end': 1205, 'answer': 'generally friendly'}
# How is the accomodation? {'score': 0.003145982977002859, 'start': 457, 'end': 503, 'answer': 'the frontline room was ok, but nothing special'}
# What is the schedule? {'score': 0.031125349923968315, 'start': 11, 'end': 17, 'answer': 'a week'}