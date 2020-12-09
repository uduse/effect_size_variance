#!/usr/bin/env python
# coding: utf-8

# In[77]:


import numpy as np
import pandas as pd
import statistics
# from bokeh.plotting import figure
# from bokeh.io import show,output_notebook
from statsmodels.stats.proportion import proportion_confint


# In[89]:


def corr_info(x, y):
    corr = np.corrcoef(x, y)[0][1]
    print(f"corr: {corr}")


# In[100]:


def summarize(arr):
    return str(round(arr.mean(), 2)) + ' ± ' + str(round(arr.std(), 2))


# In[101]:


def read_ret(fname):
    with open(fname, 'r') as f: 
        lines = f.read().splitlines() 
        for line in reversed(lines): 
            if 'AverageEpRet' in line: 
                val = float(line.split('|')[2].strip())
                break
    return val


# In[119]:


# def parse_run(id_, num_runs):
id_ = 14031034
base_dir = "log/" + str(id_) + '/'
with open(base_dir + 'job.sh') as f:
    for line in f.read().splitlines():
        if '#SBATCH --array' in line:
            rest = line.replace('#SBATCH --array=1-', '')
            num_runs = int(rest)
            break
print('num_runs', num_runs)


# In[120]:


vpg = []
ppo_c = []
ppo_uc = []

for i in range(1, num_runs + 1):
    vpg_i_fname = str(i) + '_vpg.out'
    vpg_i = read_ret(base_dir + vpg_i_fname)
    vpg.append(vpg_i)
    
    ppo_i_c_fname = str(i) + '_ppo_c.out'
    ppo_i_c = read_ret(base_dir + ppo_i_c_fname)
    ppo_c.append(ppo_i_c)
    
    ppo_i_uc_fname = str(i) + '_ppo_uc.out'
    ppo_i_uc = read_ret(base_dir + ppo_i_uc_fname)
    ppo_uc.append(ppo_i_uc)

vpg = np.array(vpg)
ppo_c = np.array(ppo_c)
ppo_uc = np.array(ppo_uc)


# In[121]:


print('vpg', summarize(vpg))
print('ppo_c', summarize(ppo_c))
print('ppo_uc', summarize(ppo_uc))


# In[122]:


delta_c = vpg - ppo_c
delta_uc = vpg - ppo_uc


# In[123]:


print('delta_c', summarize(delta_c))
print('delta_uc', summarize(delta_uc))


# In[124]:


var_c = np.var(delta_c)
var_uc = np.var(delta_uc)
print(f'var_c: {var_c}')
print(f'var_uc: {var_uc}')
print(f'var_c < var_uc ? {var_c < var_uc}')


# In[125]:


# print(np.abs(delta_c))
# print(np.abs(delta_uc))
# print(np.abs(delta_c) < np.abs(delta_uc))
print(sum(np.abs(delta_c) == np.abs(delta_uc)))
num_success = np.sum((np.abs(delta_c) < np.abs(delta_uc)))


# In[126]:


print('num_success:', num_success)
print('num_runs:', num_runs)
interval = proportion_confint(num_success, num_runs)
print('interval:', interval)


# In[127]:


if interval[0] < 0.5 < interval[1]:
    print('Interval doesn\'t tell us anything.')
elif interval[1] < 0.5:
	print('Controlled case has more variance')
else:
	print('Uncontrolled case has more variance')
