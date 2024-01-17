import pandas as pd
from utils import get_data, GRU_model, compile_train, format_3d, format_2d, save_model, LR, save_Sklearn, kNN, LSTM_model, RandomForest
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from joblib import dump, load

# UPSAMPLE OF NORMAL FLOWS
    
samples = pd.read_csv('cicddos2019/01-12/export_dataframe_proc.csv', sep=',')

X_train, y_train = get_data(samples)

#junta novamente pra aumentar o numero de normais
X = pd.concat([X_train, y_train], axis=1)

# separate minority and majority classes
is_benign = X[' Label']==0 #base de dados toda junta

normal = X[is_benign]
ddos = X[~is_benign]

# upsample minority
normal_upsampled = resample(normal,
                          replace=True, # sample with replacement
                          n_samples=len(ddos), # match number in majority class
                          random_state=27) # reproducible results

# combine majority and upsampled minority
upsampled = pd.concat([normal_upsampled, ddos])

# Specify the data 
X_train=upsampled.iloc[:,0:(upsampled.shape[1]-1)]    #DDoS
y_train= upsampled.iloc[:,-1]  #DDoS

input_size = (X_train.shape[1], 1)

del X, normal_upsampled, ddos, upsampled, normal #, l1, l2

scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(X_train)

dump(scaler, 'Models/minmax_scaler.joblib')
X_train = scaler.transform(X_train)

## GRU
model_gru = GRU_model(82)
model_gru = compile_train(model_gru, format_3d(X_train), y_train)
save_model(model_gru, 'GRU20-32-b256')

model_lr = LR()
model_lr = compile_train(model_lr, format_2d(X_train) ,y_train,False)
save_Sklearn(model_lr, 'LR')

model_knn = kNN()
model_knn = compile_train(model_knn,format_2d(X_train),y_train,False)
save_Sklearn(model_knn, 'kNN-1viz')

model_rf = RandomForest()
model_rf = compile_train(model_rf,format_2d(X_train),y_train,False)
save_Sklearn(model_rf, 'RF')

model_lstm = LSTM_model(82)
model_lstm = compile_train(model_lstm,format_3d(X_train),y_train)
save_model(model_lstm, 'LSTM5-32-b256')
