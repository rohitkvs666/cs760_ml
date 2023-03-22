import math
import numpy as np
import itertools
from pathlib import Path
import glob

directory = "."

alphabets = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',' ']
alpha = 0.5
k_l = 3
k_char = 27

def freq_count(file_name):
    file = open(file_name,'r')
    text = file.read()
    dic = {}
    for char in alphabets:
        if dic.get(char) != None:
            dic[char] += text.count(char)
        else:
            dic[char] = text.count(char)
    file.close()
    return dic

def train():
    train_eng_list = glob.glob(directory+'/e?.txt')
    train_sp_list = glob.glob(directory+'/s?.txt')
    train_jp_list = glob.glob(directory+'/j?.txt')
    total_no_of_training_files = len(glob.glob(directory+'/e?.txt')) + len(glob.glob(directory+'/s?.txt')) + len(glob.glob(directory+'/j?.txt'))
    dict_prior = {}
    dict_prior['e'] = (len(glob.glob(directory+'/e?.txt'))+alpha)/(total_no_of_training_files + k_l*alpha)
    dict_prior['s'] = (len(glob.glob(directory+'/s?.txt'))+alpha)/(total_no_of_training_files + k_l*alpha)
    dict_prior['j'] = (len(glob.glob(directory+'/j?.txt'))+alpha)/(total_no_of_training_files + k_l*alpha)
    dict_eng = {}
    dict_sp = {}
    dict_jap = {}
    for (f_e,f_s,f_j) in itertools.zip_longest(train_eng_list,train_sp_list,train_jp_list):
        dic_temp_eng = {}
        dic_temp_sp = {}
        dic_temp_jap = {}
        dic_temp_eng = freq_count(f_e)
        dic_temp_sp = freq_count(f_s)
        dic_temp_jap = freq_count(f_j)

        for key in dic_temp_eng:
            if dict_eng.get(key) != None:
                dict_eng[key] = dict_eng[key] + dic_temp_eng[key]
            else:
                dict_eng[key] = dic_temp_eng[key]

        for key in dic_temp_sp:
            if dict_sp.get(key) != None:
                dict_sp[key] = dict_sp[key] + dic_temp_sp[key]
            else:
                dict_sp[key] = dic_temp_sp[key]


        for key in dic_temp_jap:
            if dict_jap.get(key) != None:
                dict_jap[key] = dict_jap[key] + dic_temp_jap[key]
            else:
                dict_jap[key] = dic_temp_jap[key]

    total_char_count_sp = sum(dict_sp.values())
    total_char_count_eng = sum(dict_eng.values())
    total_char_count_jap = sum(dict_jap.values())
    
    for (k_e,k_s,k_j) in zip(dict_eng,dict_sp,dict_jap):
        count_eng = dict_eng[k_e]
        count_sp = dict_sp[k_s]
        count_jap = dict_jap[k_j]
        prob_count_eng = float((count_eng+alpha)/(total_char_count_eng + k_char*alpha))
        prob_count_sp = float((count_sp+alpha)/(total_char_count_sp + k_char*alpha))
        prob_count_jap = float((count_jap+alpha)/(total_char_count_jap+k_char*alpha))
        dict_eng[k_e] = [count_eng,prob_count_eng,math.log(prob_count_eng)]
        dict_sp[k_s] = [count_sp,prob_count_sp,math.log(prob_count_sp)]
        dict_jap[k_j] = [count_jap,prob_count_jap,math.log(prob_count_jap)]
    
    return dict_eng,dict_sp,dict_jap,dict_prior

#def predict(file_name):
#print(glob.glob(directory+'/???.txt'))

d_eng = {}
d_sp = {}
d_jap = {}
d_prior = {}

d_eng , d_sp , d_jap , d_prior = train()


predictions = {}

test_set = glob.glob(directory+'/???.txt')

for f in test_set:
    dict_test = freq_count(f)

    prob_test_eng = 0
    prob_test_sp = 0
    prob_test_jap = 0

    for (k_test,k_e,k_s,k_j) in zip(dict_test, d_eng,d_sp,d_jap):
        prob_test_eng += dict_test[k_test]*d_eng[k_e][2]
        prob_test_sp += dict_test[k_test]*d_sp[k_s][2]
        prob_test_jap += dict_test[k_test]*d_jap[k_j][2]
        
    prob_eng_test = prob_test_eng + math.log(d_prior['e'])
    prob_sp_test = prob_test_sp + math.log(d_prior['s'])
    prob_jap_test = prob_test_jap + math.log(d_prior['j'])
    #print(f)
    #print("ln( p_hat(x | y = e) ) : " + str(prob_test_eng))
    #print("ln( p_hat(x | y = s) ) : " + str(prob_test_sp))
    #print("ln( p_hat(x | y = j) ) : "  + str(prob_test_jap))
    #print("\n\n")
    #print(f)
    #print("ln( p_hat(y = e | x) ) : " + str(prob_eng_test))
    #print("ln( p_hat(y = s | x) ) : " + str(prob_sp_test))
    #print("ln( p_hat(y = j | x) ) : "  + str(prob_jap_test))

    max_value = max(prob_eng_test, prob_sp_test, prob_jap_test)
    
    #print("the max value of prob of test are : " + str(max_value))
    #print("\n")
    
    if max_value == prob_eng_test: 
        predictions[f] = 'e'
    elif max_value == prob_sp_test :
        predictions[f] = 's'
    elif max_value == prob_jap_test:
        predictions[f] = 'j'

c = 1
for k, v in predictions.items():
  print("FILE " + str(c) + " : " + k + " Prediction : " + v)
  c = c + 1