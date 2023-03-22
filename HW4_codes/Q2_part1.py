from pathlib import Path
import glob
#directory = "C:\\Users\\yashw\\Documents\\CS760\\P4\\hw4-1\\languageID\\"
directory = "."
alpha = 0.5
k_l = 3
k_s = 27
L = ['e','j','s']
prob_prior_lst = []
#print(glob.glob(directory+'/e?.txt'))
total_no_of_training_files = len(glob.glob(directory+'/e?.txt')) + len(glob.glob(directory+'/s?.txt')) + len(glob.glob(directory+'/j?.txt'))
prob_prior_eng_file= (len(glob.glob(directory+'/e?.txt'))+alpha)/(total_no_of_training_files + k_l*alpha)
prob_prior_lst.append(prob_prior_eng_file)
prob_prior_sp_file=(len(glob.glob(directory+'/s?.txt'))+alpha)/(total_no_of_training_files + k_l*alpha)
prob_prior_lst.append(prob_prior_sp_file)
prob_prior_jap_file=(len(glob.glob(directory+'/j?.txt'))+alpha)/(total_no_of_training_files + k_l*alpha)
prob_prior_lst.append(prob_prior_jap_file)
print(prob_prior_lst)