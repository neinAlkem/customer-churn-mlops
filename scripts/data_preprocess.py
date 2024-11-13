import pandas as pd
import copy
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import yaml

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

data = pd.read_csv(config['data']['raw'])

data = data.drop(columns=['CLIENTNUM', 'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1', 
                           'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'])

data_train = copy.deepcopy(data)

categorical_cols = np.array(data.columns[data.dtypes == object])
numerical_cols = np.array(data.columns[data.dtypes != object])

d = defaultdict(LabelEncoder)
data_train[categorical_cols] = data_train[categorical_cols].apply(lambda x: d[x.name].fit_transform(x))

data_cor = data_train.corr()

corelation_matrix_target = data_cor['Attrition_Flag']
k = config['model']['top_k_features']  
top_k = corelation_matrix_target.abs().sort_values(ascending=False)[:min(k, len(corelation_matrix_target))].index

selected_features = data_train[top_k]

selected_features.to_csv(config['data']['processed'], index=False)
