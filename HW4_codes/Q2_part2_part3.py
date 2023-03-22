import math
import itertools
alphabets = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',' ']
alpha = 0.5
k_l = 3
k_char = 27
directory = "."
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

#print(freq_count('C:\\Users\\yashw\\Documents\\CS760\\P4\\hw4-1\\languageID\\e10.txt'))

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
#print(dict_eng)
#counta = dict_sp['a']
#dict_sp['a'] = [counta, counta/total_char_count_sp]
#dict_sp['a'].append([counta, counta/total_char_count_sp])
#print(dict_sp['a'])
#print(total_char_count_sp)
#print(dict_eng)
#print(dict_jap)
#print(dict_sp)
theta_e = []
theta_s = []
theta_j = []
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
    theta_e.append(round(prob_count_eng,5))
    theta_s.append(round(prob_count_sp,5))
    theta_j.append(round(prob_count_jap,5))

    
print("theta_e is :")
print(theta_e)
#for k in range(8):
#  print(theta_e[k], end=" ")
#  print("")
#for k in range(8, 16):
#  print(theta_e[k], end=" ")
#  print("")

print("\n")
print("theta_s : ")
print(theta_s)
print("\n")
print("theta_j : ")
print(theta_j)
print("\n")