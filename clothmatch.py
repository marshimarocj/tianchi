import cPickle
import os
import time
import random
import math
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.linear_model import LogisticRegression

#global params
listdir = "/home/apexgpu/data/tianchi/lists/"
imgdir = '/home/apexgpu/data/tianchi/imgs/tianchi_fm_img1_'
# CJ: global variable is a good place to place config information such as dir name

#init in load_items
items = {}
category = {}
describe = {}
# CJ: Making these variables global is dangerous,
# are you sure you won't change them in your code?
# event if they are commonly used variables, you can
# access them by call a load function, which return these variables.

#init in prepare_item_txt_feature
key_desc = []
#init in load_matches
match = {}
match_cat = {}
match_items = []
#init in load_bought
bought = {}
#init in init_key_cat
key_cat = []
key_cat_matrix = []
#init in load vali
test = {}
test_match = {}
# CJ: for similar reasons, these variables shouldn't be global variable.
# global variable is a very bad coding habit. Even for acmers, they
# have to change this bad behavior to be a SE. And I don't see any reason
# for a non-acmers to have this habit.
# in python, only the basic types (int, float, ...) are passed by copy,
# all other objects are passed by reference, so there is no reason to use
# global variables. It is very dangerous since you may modify them somewhere
# in your code unexpectedly. Then with global variable, you can never know where
# you changed it incorrectly.

#func
def load_items():
	starttime = time.time()
	itemfile = open(listdir+"dim_items.txt",'r')
	lines = itemfile.readlines()
	itemfile.close()
	# CJ: readlines is a little bit brute force
	# since you may not need all the information in the file.
	# This also leads to unnecessary memory occupation if file is not small
	# a simple way to read file line by line could be:
	# with open(itemfile) as f:
	# 	for line in f:
	#		process(line)
	# such code also saves the line f.close()
	# for the usage and meaning of "with", you can google it
	print 'Num of items:',len(lines)
	for line in lines:
		item_id,rest = line.split(' ',1)
		rest = rest.split()
		cat_id = rest[0]
		if len(rest)==2:
			terms = rest[1]
		else:
			print 'No name!'
			terms = ''
		items[item_id] = [cat_id,terms.split(',')]
		# CJ: you are starting a new section of logic after this line,
		# so it's better to leave one blank line after this line
		# so that logic sections can be separated by blank lines.
		# It is a good habit to make code clear and more readable
		# since in doing experiments, you are likely to change your past code in the near future,
		# which means that you are likely to read your own code.
		# blank lines do lead to more lines of code, but when we are talking
		# about succinct code, we are talking about logic.

		#save to category
		if cat_id in category:
			category[cat_id].append(item_id)
		else:
			category[cat_id] = [item_id]
		# CJ:  a more uniform way to handle key not in map issue:
		# if cat_id not in category:
		# 	category[cat_id] = []
		# category[cat_id].append(cat_id)

		#the above blank line is added by CJ for the same reason
		#count term
		for term in terms.split(','):
			if term in describe:
				describe[term] += 1
			else:
				describe[term] = 1
		# CJ: you can use the above uniform way to handle this section of code as well
		# CJ: usually after preprocessing, your program just copes with consecutive int IDs,
		# which is fast to process in array/map and save memory. Just store the string to int map
		# in another file when you need interpret/case study the result.

	# CJ: the above blank line is added by CJ for the same reason.
	# I won't add blank line in other functions and just leave them for exercise
	#count
	print 'Num of category:',len(category)
	print 'Num of describe:',len(describe)
	#check describe
	count = 0;
	for term in describe:
		if describe[term] >=20:
			count += 1
		# CJ:  it is better encode 20 by a variable, say "xxThreshold", since it is likely that
		# we will tune this threshold in the future, if you mention 20 in more than one place in your code
		# it is likely to cause bug when you need to change the threshold 20 in the future
	print 'describe>=20:',count
	print '**load items done in %.1f s**\n'%(time.time() - starttime)

def prepare_item_txt_feature(thresh = 20):
	starttime = time.time()
	for term in describe:
		if describe[term]>=thresh:
			key_desc.append(term)
	# CJ: this could be written by one line using filter function
	print 'desc vector:',len(key_desc)
	#desc_ref
	desc_ref = {}
	for i in range(len(key_desc)):
		desc_ref[key_desc[i]] = i
	#feature all
	for item_id in items:
		'''#dense feature
		item_feature = np.zeros(len(key_desc), np.bool)
		for term in items[item_id][1]:
			if term in desc_ref:
				item_feature[desc_ref[term]] = True
		'''

		#sparse feature
		item_feature = []
		for term in items[item_id][1]:
			if term in desc_ref:
				item_feature.append(desc_ref[term])
		items[item_id].append(item_feature)
	print 'calc item text feature done in %.1f s\n'%(time.time()-starttime)

def load_matches():
	starttime = time.time()
	matchfile = open(listdir+"dim_fashion_matchsets.txt",'r')
	lines = matchfile.readlines()
	matchfile.close()
	print 'Num of match groups:',len(lines)
	matchcount = 0
	for item_id in items:
		match[item_id] = []
	for line in lines:
		coll_id,item_list = line.split()
		#store each item match
		onematch = item_list.split(';')
		for i in range(len(onematch)):
			onematch[i] = onematch[i].split(',')
		for i in range(len(onematch)):
			matchlist = []
			for j in range(i+1,len(onematch)):
				matchlist += onematch[j]
			for item_id in onematch[i]:
				for match_id in matchlist:
					match[item_id] += [match_id]
					match[match_id] += [item_id]
					matchcount += 1
					# CJ: It is good to use A += B if B has more than one element for code concise
					# But for one element, just call add(), list has other memory overhead
					# don't waste any unnecessary memory
	print 'Num of match pairs:',matchcount
	#remove dup
	count = 0
	for item_id in items:
		# CJ: for large block or embedded if block
		# one trick to avoid unnecessary indention in code
		# is to write as:
		# if len(match[item_id]) <= 0
		# 	continue/return
		# code to do the work
		# this is especially useful when you just want to process one case
		# but have several cases to skip and the judgment of these other cases
		# have to be processed by several if conditions
		if len(match[item_id])>0:
			match_items.append(item_id)
			nodup = []
			for match_id in match[item_id]:
				if match_id in nodup:
					matchcount-=1
				else:
					nodup.append(match_id)
			match[item_id] = nodup
			item_cat = items[item_id][0]
			if item_cat in match_cat:
				match_cat[item_cat] += 1
			else:
				match_cat[item_cat] = 1
	print 'Num of match pairs(No dup):',matchcount
	print 'Num of match items:',len(match_items)
	print 'Num of match category:',len(match_cat)
	#count match_cat
	count = 0
	for cat in match_cat:
		count += len(category[cat])
	print 'match category sum:',count
	print '\t',count*1000/len(items)/10.0,'%'
	print '**load match done in %.1f s**\n'%(time.time()-starttime)

def load_bought():
	starttime = time.time()
	userfile = open(listdir+"user_bought_history.txt",'r')
	lines = userfile.readlines()
	print 'Num of bought:',len(lines)
	for item_id in items:
		bought[item_id] = 0
	for line in lines:
		user_id,item_id,create_at = line.split()
		bought[item_id] += 1
	count = 0
	for item_id in items:
		if bought[item_id] > 0:
			count += 1
	print 'Num of items bought:',count
	print '**load bought done in %.1fs**\n'%(time.time()-starttime)

def init_key_cat():
	starttime = time.time()
	for cat_id in match_cat:
		key_cat.append(cat_id)
	#print key_cat
	print 'Num of key cat:',len(key_cat)
	'''
	for cat_id in match_cat:
		match_cat[cat_id] = 0
	for item_id in items:
		if len(match[item_id])>0:
			match_cat[items[item_id][0]] += 1
	for cat_id in match_cat:
		print cat_id,match_cat[cat_id]
	'''
	ref = {}
	for i in range(len(key_cat)):
		ref[key_cat[i]] = i
	for i in range(len(key_cat)):
		key_cat_matrix.append([0]*len(key_cat))
	for item_id in match:
		for match_id in match[item_id]:
			key_cat_matrix[ref[items[item_id][0]]][ref[items[match_id][0]]] += 1
	print '**init key cat & matrix done in %.1f s**\n'%(time.time()-starttime)

def check_matrix():
	ref = {}
	for i in range(len(key_cat)):
		ref[key_cat[i]] = i
	new_cat_matrix = []
	for i in range(len(key_cat)):
		new_cat_matrix.append([0]*len(key_cat))
	for item_id in match:
		for match_id in match[item_id]:
			new_cat_matrix[ref[items[item_id][0]]][ref[items[match_id][0]]] += 1
	#check
	for i in range(len(key_cat)):
		for j in range(i,len(key_cat)):
			if key_cat_matrix[i][j]>0:
				if new_cat_matrix[i][j]>0:
					pass
				else:
					print 'err1 at',i,j,key_cat_matrix[i][j]
			elif new_cat_matrix[i][j]>0:
				print 'err2 at',i,j
	print 'check matrix done\n'

def divide_gt_rand(target = 5000):
	vali = {}
	vali['bought'] = []
	vali['unbought'] = []
	for item_id in match_items:
		if bought[item_id]>0:
			vali['bought'].append(item_id)
		else:
			vali['unbought'].append(item_id)
	random.shuffle(vali['bought'])
	random.shuffle(vali['unbought'])
	vali['bought'] = vali['bought'][:target/2]
	vali['unbought'] = vali['unbought'][:target/2]
	print 'Vali:',len(vali['bought']),'(b)+',len(vali['unbought'])
	print '**divide done**'
	outfile = open('divide_vali.lst','w')
	cPickle.dump(vali,outfile)
	outfile.close()
	print '**save done**'
	# CJ: when you use cPickle, you usually want to save more space
	# rather than its readbility, so you could use 'wb' to save more space on disk
	# if you want readability, you can use json to save file

def divide_gt(target = 5000):
	vali = {}
	vali['bought'] = []
	vali['unbought'] = []
	candi = {}
	candi['bought'] = []
	candi['unbought'] = []
	for item_id in match_items:
		if bought[item_id]>0:
			candi['bought'].append(item_id)
		else:
			candi['unbought'].append(item_id)
	random.shuffle(candi['bought'])
	random.shuffle(candi['unbought'])
	count = 0
	while(count<(target/2)):
		candi_id = candi['bought'][0]
		if try_delete(candi_id):
			vali['bought'].append(candi_id)
			count += 1
		candi['bought'].pop(0)
	count = 0
	while(count<(target/2)):
		candi_id = candi['unbought'][0]
		if try_delete(candi_id):
			vali['unbought'].append(candi_id)
			count += 1
		candi['unbought'].pop(0)
	print 'vali:',len(vali['bought']),len(vali['unbought'])
	print '**divide done**'
	outfile = open('divide_vali.lst','w')
	cPickle.dump(vali,outfile)
	outfile.close()
	print '**save done**'

	#check


def try_delete(candi_id):
	#check match_cat
	candi_cat = items[candi_id][0]
	if match_cat[candi_cat]<=1:
		return False
	#check matrix
	ref = {}
	for i in range(len(key_cat)):
		ref[key_cat[i]] = i
	for match_id in match[candi_id]:
		cat_id = items[match_id][0]
		if key_cat_matrix[ref[candi_cat]][ref[cat_id]] <=1:
			return False
	#check passed and update
	for match_id in match[candi_id]:
		match[match_id].remove(candi_id)
		cat_id = items[match_id][0]
		if key_cat_matrix[ref[candi_cat]][ref[cat_id]]!=key_cat_matrix[ref[cat_id]][ref[candi_cat]]:
			raw_input('error')
		key_cat_matrix[ref[candi_cat]][ref[cat_id]] -= 1
		key_cat_matrix[ref[cat_id]][ref[candi_cat]] -= 1
	match_cat[candi_cat] -= 1
	match_items.remove(candi_id)
	match[candi_id] = []
	return True

def check_divide():
	divide_file = open('divide_vali.lst','r')
	vali = cPickle.load(divide_file)
	divide_file.close()
	test = vali['bought']+vali['unbought']
	print 'test:',len(test)
	test_cat = []
	for test_id in test:
		if items[test_id][0] in test_cat:
			pass
		else:
			test_cat.append(items[test_id][0])
		# CJ: could be abbriviated by
		# if items[test_id][0] not in testcat:
		# 	test_cat.append(items[test_id][0])
		# you should know the not keyword, it's in the tutorial
	print 'test_cat:',len(test_cat)
	count = 0
	for cat_id in test_cat:
		count += len(category[cat_id])
	print 'count:',count
	print 'perc:',float(count)/len(items)

def check_test():
	test_cat = {}
	testfile = open(listdir+"test_items.txt",'r')
	lines = testfile.readlines()
	testfile.close()
	print 'Num of test:',len(lines)
	for line in lines:
		test_id = line.split()[0]
		if test_id in test:
			print 'dup test:',line
		else:
			test[test_id] = items[test_id]
			if test[test_id][0] in test_cat:
				test_cat[test[test_id][0]].append(test_id)
			else:
				test_cat[test[test_id][0]] = [test_id]
	print 'test_cat:',len(test_cat)
	for cat_id in test_cat:
		print cat_id,match_cat[cat_id]

	all_in = True
	for cat in test_cat:
		if cat in match_cat:
			pass
		else:
			all_in = False
	print 'test_cat all in match_cat:',all_in
	#print 'No|test Cat|num of test|num of cat|num of bought|num of unbought'
	total = 0
	for i,cat_id in enumerate(test_cat):
		count = 0
		for test_id in test_cat[cat_id]:
			if bought[test_id]>0:
				count += 1
		#print i,cat_id,len(test_cat[cat_id]),len(category[cat_id]),count,len(test_cat[cat_id])-count
		total += count
	print 'test bought:',total
	for test_id in test:
		if len(match[test_id])>0:
			print test_id,'have match:',len(match[test_id])
	print '**load test done.**'

def load_vali():
	valifile = open('divide_vali.lst','r')
	vali = cPickle.load(valifile)
	valifile.close()
	test.clear()
	for test_id in vali['bought']+vali['unbought']:
		test[test_id] = items[test_id]
		test_match[test_id] = match[test_id][:]
	print 'Num of vali:',len(test)

def update_matches():
	starttime = time.time()
	for test_id in test:
		match_items.remove(test_id)
		for match_id in match[test_id]:
			match[match_id].remove(test_id)
		del match[test_id][:]
	print 'update match done'
	match_cat.clear()
	for item_id in items:
		if len(match[item_id])>0:
			cat_id = items[item_id][0]
			if cat_id in match_cat:
				match_cat[cat_id].append(item_id)
			else:
				match_cat[cat_id] = [item_id]
	print 'Num of match cat:',len(match_cat)
	ref = {}
	for i in range(len(key_cat)):
		ref[key_cat[i]] = i
	del key_cat_matrix[:]
	for i in range(len(key_cat)):
		key_cat_matrix.append([0]*len(key_cat))
	for item_id in match:
		for match_id in match[item_id]:
			key_cat_matrix[ref[items[item_id][0]]][ref[items[match_id][0]]] += 1
	print 'update matrix done'
	'''
	#check 1
	if len(match_cat)!=71:
		for cat_id in key_cat:
			if not (cat_id in match_cat):
				print 'Miss cat:',cat_id
				for test_id in test:
					if items[test_id][0] == cat_id:
						print '\t',test_id
	#check 2
	for item_id in items:
		if item_id in test:
			if len(match[item_id])>0:
				print 'err1:',item_id
		else:
			if len(match[item_id])>0:
				for match_id in match[item_id]:
					if match_id in test:
						print 'err2:',item_id
	'''
	print '**update match done in %.1f s**\n'%(time.time()-starttime)

def reduce_candi():
	item_list = []
	candi_list = []
	for item_id in items:
		if item_id in match_items:
			candi_list.append(item_id)
		else:
			item_list.append(item_id)
	print 'origin(neg items):',len(item_list)
	random.shuffle(item_list)
	candi_list += item_list[:150000-len(match_items)]
	print 'candi now:',len(candi_list)
	#store
	outfile = open('test_candi.lst','w')
	cPickle.dump(candi_list,outfile)
	outfile.close()

def update_category():
	infile = open('test_candi.lst','r')
	candi_list = cPickle.load(infile)
	infile.close()
	for cat_id in key_cat:
		del category[cat_id][:]
	for item_id in candi_list:
		category[items[item_id][0]].append(item_id)
	print 'update category done'

######################################################################################
###############                      INIT DONE                  #################################################
######################################################################################

def get_match_cat(cat_id):
	match_cat_result = []
	match_cat_num = []
	for i in range(len(key_cat)):
		if key_cat[i] == cat_id:
			for j in range(len(key_cat)):
				if key_cat_matrix[i][j]>0:
					match_cat_result.append(key_cat[j])
					match_cat_num.append(key_cat_matrix[i][j])
			return (match_cat_result,match_cat_num)

def divide_data():
	starttime = time.time()
	train_pos = []
	train_neg = []
	#form pos
	for item_id in match:
		if len(match[item_id])>0:
			for match_id in match[item_id]:
				if item_id<match_id:
					train_pos.append((item_id,match_id))
	print 'train_pos:',len(train_pos)
	#form neg
	# CJ: don't write too much codes in one line
	# usually one line consists less than 80 characters
	# you could know how to break one line of code in to several lines
	# by google it
	# I consider writing such code as too lazy to know how to break one
	# line of code in to several lines in python
	for i,pos_pair in enumerate(train_pos):
		neg_pair = (category[items[pos_pair[0]][0]][random.randint(0,len(category[items[pos_pair[0]][0]])-1)], category[items[pos_pair[1]][0]][random.randint(0,len(category[items[pos_pair[1]][0]])-1)])
		while(neg_pair[1] in match[neg_pair[0]]):
			neg_pair = (category[items[pos_pair[0]][0]][random.randint(0,len(category[items[pos_pair[0]][0]])-1)], category[items[pos_pair[1]][0]][random.randint(0,len(category[items[pos_pair[1]][0]])-1)])
		train_neg.append(neg_pair)
		if i%10000==0:
			print i
	print 'train_neg:',len(train_neg)
	print '**divide data done in %.1f s**'%(time.time()-starttime)
	train = {'pos':train_pos,'neg':train_neg}
	outfile = open('train_pairs.lst','w')
	cPickle.dump(train,outfile)
	outfile.close()
	print '**save done**\n'

def train_LR():
	print 'load train pairs'
	infile = open('train_pairs.lst','r')
	divide_list = cPickle.load(infile)
	infile.close()
	print 'prepare train feature...'
	#form feature vector
	train_X = []
	train_Y = []
	#sparse feature
	row = []
	col = []
	starttime = time.time()
	count = 0
	for pair in (divide_list['pos']+divide_list['neg']):
		#text feature
		for i in items[pair[0]][2]:
			if i in items[pair[1]][2]:
				row.append(count)
				col.append(i)
			else:
				row.append(count)
				col.append(i+len(key_desc))
		for i in items[pair[1]][2]:
			if not(i in items[pair[0]][2]):
				row.append(count)
				col.append(i+2*len(key_desc))
		#count
		count += 1
	print 'count:',count
	train_Y = [1]*(len(divide_list['pos']))+[0]*(len(divide_list['neg']))
	data = np.ones(len(row),np.bool)

	train_X = coo_matrix((data,(row,col)),shape=(len(train_Y),3*len(key_desc)))

	Cs = [1.,10.,100.]
	Penaltys = ['l1','l2']
	for penalty in Penaltys:
		for C in Cs:
			print 'start fitting LR(%s,%.1f)'%(penalty,C)
			starttime = time.time()
			model = LogisticRegression(penalty = penalty, C=C)
			model.fit(train_X,train_Y)
			print '\tfitting done in {0:.1f} s'.format(time.time()-starttime)
			print '\ttrain score',model.score(train_X,train_Y)
			#store
			modelname = 'LR_3vec_%s_%.1f.model'%(penalty,C)
			modelfile = open(modelname,'w')
			cPickle.dump(model,modelfile)
			modelfile.close()
			print '\tsave done\n'



def pred_test_LR(modelname,resultname):
	mf = open(modelname,'r')
	model= cPickle.load(mf)
	print model
	mf.close()
	resultfile = open(resultname,'w')
	#test feature
	alltime = time.time()
	cnt = 1
	for test_id in test:
		stime = time.time()
		print 'test',cnt,':',test_id
		candi_list = []
		row = []
		col = []
		data = []
		count = 0
		starttime = time.time()
		match_cat_list,match_cat_num = get_match_cat(items[test_id][0])
		for cat_id in match_cat_list:
			candi_list += category[cat_id]
			test_item_feature = items[test_id][2]
			for item_id in category[cat_id]:
				candi_item_feature = items[item_id][2]
				#text feature
				for i in test_item_feature:
					if i in candi_item_feature:
						row.append(count)
						col.append(i)
					else:
						row.append(count)
						col.append(i+len(key_desc))
				for i in candi_item_feature:
					if not(i in test_item_feature):
						row.append(count)
						col.append(i+2*len(key_desc))
				#count
				count += 1
		print '\tcalc feature done in %0.2f s' % (time.time()-starttime)
		starttime = time.time()
		data = np.ones(len(row),np.bool)
		test_feature = coo_matrix((data,(row,col)),shape = (count,3*len(key_desc)))
		print '\tcoo matrix done in %0.2f s' % (time.time()-starttime)
		starttime = time.time()
		resultfile.write('%s:'%(test_id))
		test_pred = model.decision_function(test_feature)
		#test_pred = model.predict(test_feature)
		sort_idx = np.argsort(test_pred)[::-1]
		gt = test_match[test_id]
		for i,idx in enumerate(sort_idx):
			if candi_list[idx] in gt:
				resultfile.write('(%s,%d)'%(candi_list[idx],i+1))
		resultfile.write('\n')
		resultfile.flush()
		print '\tpred and sort done in %0.2f s' % (time.time() - starttime)
		cnt+=1
		print '\tall done in {0:.1f}s'.format(time.time()-stime)
	resultfile.close()
	print 'pred test done in {0:.1f}min'.format((time.time()-alltime)/60)

def mAP(filename):
	mAP = 0.
	resultfile = open(filename,'r')
	lines = resultfile.readlines()
	#print 'Num of tests:',len(lines)
	resultfile.close()
	count = 0
	for line in lines:
		count += 1
		test_id,result = line.split(':')
		#print 'Test',count,test_id
		result = line.split(')')
		result.remove('\n')
		ap = 0.
		for i,tup in enumerate(result):
			pos = int(tup.split(',')[1])
			p = float(i+1)/pos
			ap += 1./(1-math.log(p,math.e))
		mAP += (ap/len(result))
	return mAP/len(lines)

def mAP_200(filename):
	mAP = 0.
	resultfile = open(filename,'r')
	lines = resultfile.readlines()
	#print 'Num of tests:',len(lines)
	resultfile.close()
	count = 0
	for line in lines:
		count += 1
		test_id,result = line.split(':')
		#print 'Test',count,test_id
		result = line.split(')')
		result.remove('\n')
		ap = 0.
		for i,tup in enumerate(result):
			pos = int(tup.split(',')[1])
			if pos<=200:
				p = float(i+1)/pos
				ap += 1./(1-math.log(p,math.e))
		mAP += (ap/len(result))
	return mAP/len(lines)

############
if __name__ == '__main__':

	#divide gt
	load_items()
	prepare_item_txt_feature()
	load_matches()
	#load_bought()
	init_key_cat()

	##divide_gt()
	##check_divide()
	##check_test()

	load_vali()
	update_matches()
	##check_matrix()

	#reduce_candi()
	update_category()
	print '****INIT DONE****\n'

	modelname = 'LR_3vec_l2_10.0.model'
	#divide_data()
	#train_LR()
	#print '****TRAIN DONE****\n'

	resultname = modelname+'.reduced.result'
	pred_test_LR(modelname,resultname)
	print '****PRED DONE****\n'

	print 'mAP:',mAP(resultname)
	print 'mAP@200:',mAP_200(resultname)
	print '****EVAL DONE****\n'
