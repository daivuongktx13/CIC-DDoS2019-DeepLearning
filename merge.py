import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn import preprocessing

mult = 5

def load_file(path):
    data = pd.read_csv(path, sep=',')

    is_benign = data[' Label']=='BENIGN'
    flows_ok = data[is_benign]
    flows_ddos_full = data[~is_benign]
    
    sizeDownSample = len(flows_ok)*mult # tamanho do set final de dados anomalos
    
    # downsample majority
    if (len(flows_ok)*mult) < (len(flows_ddos_full)): 
        flows_ddos_reduced = resample(flows_ddos_full,
                                         replace = False, # sample without replacement
                                         n_samples = sizeDownSample, # match minority n
                                         random_state = 27) # reproducible results
    else:
        flows_ddos_reduced = flows_ddos_full
    
    return flows_ok, flows_ddos_reduced

 
def load_huge_file(path):
    df_chunk = pd.read_csv(path, chunksize=500000)
    
    chunk_list_ok = []  # append each chunk df here 
    chunk_list_ddos = [] 

    # Each chunk is in df format
    for chunk in df_chunk:  
        # perform data filtering 
        is_benign = chunk[' Label']=='BENIGN'
        flows_ok = chunk[is_benign]
        flows_ddos_full = chunk[~is_benign]
        
        if (len(flows_ok)*mult) < (len(flows_ddos_full)): 
            sizeDownSample = len(flows_ok)*mult # tamanho do set final de dados anomalos
            
            # downsample majority
            flows_ddos_reduced = resample(flows_ddos_full,
                                             replace = False, # sample without replacement
                                             n_samples = sizeDownSample, # match minority n
                                             random_state = 27) # reproducible results 
        else:
            flows_ddos_reduced = flows_ddos_full
            
        # Once the data filtering is done, append the chunk to list
        chunk_list_ok.append(flows_ok)
        chunk_list_ddos.append(flows_ddos_reduced)
        
    # concat the list into dataframe 
    flows_ok = pd.concat(chunk_list_ok)
    flows_ddos = pd.concat(chunk_list_ddos)
    
    return flows_ok, flows_ddos

# # file 1
# flows_ok, flows_ddos = load_huge_file('cicddos2019/01-12/TFTP.csv')
# print('file 1 loaded')

# # file 2
# a,b = load_file('cicddos2019/01-12/DrDoS_LDAP.csv')
# flows_ok = flows_ok.append(a,ignore_index=True)
# flows_ddos = flows_ddos.append(b,ignore_index=True)
# print('file 2 loaded')

# # file 3
# a,b = load_file('cicddos2019/01-12/DrDoS_MSSQL.csv')
# flows_ok = flows_ok.append(a,ignore_index=True)
# flows_ddos = flows_ddos.append(b,ignore_index=True)
# print('file 3 loaded')

# # file 4
# a,b = load_file('cicddos2019/01-12/DrDoS_NetBIOS.csv')
# flows_ok = flows_ok.append(a,ignore_index=True)
# flows_ddos = flows_ddos.append(b,ignore_index=True)
# print('file 4 loaded')

# # file 5
# a,b = load_file('cicddos2019/01-12/DrDoS_NTP.csv')
# flows_ok = flows_ok.append(a,ignore_index=True)
# flows_ddos = flows_ddos.append(b,ignore_index=True)
# print('file 5 loaded')

# # file 6
# a,b = load_file('cicddos2019/01-12/DrDoS_SNMP.csv')
# flows_ok = flows_ok.append(a,ignore_index=True)
# flows_ddos = flows_ddos.append(b,ignore_index=True)
# print('file 6 loaded')

# # file 7
# a,b = load_file('cicddos2019/01-12/DrDoS_SSDP.csv')
# flows_ok = flows_ok.append(a,ignore_index=True)
# flows_ddos = flows_ddos.append(b,ignore_index=True)
# print('file 7 loaded')

# # file 8
# a,b = load_file('cicddos2019/01-12/DrDoS_UDP.csv')
# flows_ok = flows_ok.append(a,ignore_index=True)
# flows_ddos = flows_ddos.append(b,ignore_index=True)
# print('file 8 loaded')

# # file 9
# a,b = load_file('cicddos2019/01-12/Syn.csv')
# flows_ok = flows_ok.append(a,ignore_index=True)
# flows_ddos = flows_ddos.append(b,ignore_index=True)
# print('file 9 loaded')

# # file 10
# a,b = load_file('cicddos2019/01-12/DrDoS_DNS.csv')
# flows_ok = flows_ok.append(a,ignore_index=True)
# flows_ddos = flows_ddos.append(b,ignore_index=True)
# print('file 10 loaded')

# # file 11
# a,b = load_file('cicddos2019/01-12/UDPLag.csv')
# flows_ok = flows_ok.append(a,ignore_index=True)
# flows_ddos = flows_ddos.append(b,ignore_index=True)
# print('file 11 loaded')

# del a,b

# samples = flows_ok.append(flows_ddos,ignore_index=True)
# samples.to_csv(r'cicddos2019/01-12/export_dataframe.csv', index = None, header=True) 

# del flows_ddos, flows_ok

# file 1
flows_ok, flows_ddos = load_file('cicddos2019/03-11/LDAP.csv')
print('file 1 loaded')

# file 2
a,b = load_file('cicddos2019/03-11/MSSQL.csv')
flows_ok = flows_ok.append(a,ignore_index=True)
flows_ddos = flows_ddos.append(b,ignore_index=True)
print('file 2 loaded')

# file 3
a,b = load_file('cicddos2019/03-11/NetBIOS.csv')
flows_ok = flows_ok.append(a,ignore_index=True)
flows_ddos = flows_ddos.append(b,ignore_index=True)
print('file 3 loaded')

# file 4
a,b = load_file('cicddos2019/03-11/Portmap.csv')
flows_ok = flows_ok.append(a,ignore_index=True)
flows_ddos = flows_ddos.append(b,ignore_index=True)
print('file 4 loaded')

# file 5
a,b = load_file('cicddos2019/03-11/Syn.csv')
flows_ok = flows_ok.append(a,ignore_index=True)
flows_ddos = flows_ddos.append(b,ignore_index=True)
print('file 5 loaded')
'''
# following files won't load**
# file 6

a,b = load_file('cicddos2019/03-11/UDP.csv')
flows_ok = flows_ok.append(a,ignore_index=True)
flows_ddos = flows_ddos.append(b,ignore_index=True)
print('file 6 loaded')

# file 7
a,b = load_file('cicddos2019/03-11/UDPLag.csv')
flows_ok = flows_ok.append(a,ignore_index=True)
flows_ddos = flows_ddos.append(b,ignore_index=True)
print('file 7 loaded')
'''
tests = flows_ok.append(flows_ddos,ignore_index=True)
tests.to_csv(r'cicddos2019/01-12/export_tests.csv', index = None, header=True) 

del flows_ddos, flows_ok, a, b