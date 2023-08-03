import pandas as pd
import glob
import os
import numpy as np
import missingno as msno
from numpy.distutils.system_info import dfftw_info
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import linear_model
import seaborn as sns




# df_fake_account = pd.read_csv('out_final/fake_account_dataset.csv')
# print(df_fake_account.info())
# print(df_fake_account.head())

# drop the columns we don't need: userid, username, _merge
# df_fake_account = df_fake_account.drop(['user_id', 'username', '_merge'], axis=1)
# df_fake_account.to_csv('out_final/fake_account_IG_dataset.csv', index_label='id')

df_fake_account = pd.read_csv('out_final/fake_account_IG_dataset.csv')
print(df_fake_account.info())
print(df_fake_account.head())
print(df_fake_account.isnull().sum())
msno.bar(df_fake_account)


'''
# 1. MEAN IMPUTATION

df_fake_account_mean = df_fake_account.copy(deep=True)

# mean_imputer = SimpleImputer(missing_values=np.nan, strategy="mean") # applies for the numeric values only
mean_imputer = SimpleImputer(strategy="most_frequent")
mean_imputer.fit(df_fake_account_mean)
# Apply the imputer to fill the NaN values
df_fake_account_mean_filled = pd.DataFrame(mean_imputer.transform(df_fake_account_mean), columns=df_fake_account_mean.columns)
df_fake_account_mean_filled = df_fake_account_mean_filled*1
df_fake_account_mean_filled = df_fake_account_mean_filled.iloc[:, 1:]
df_fake_account_mean_filled.to_csv('dataset/fake_account_most_freq_imputer.csv', index_label='id')
print(df_fake_account_mean_filled.info())
print(df_fake_account_mean_filled.head())
print(df_fake_account_mean_filled.isnull().sum())
msno.bar(df_fake_account_mean_filled)

# plt.show()
'''

'''
# 2. MIX MOST FREQUENT IMPUTATION FOR BOOL VALUES AND MICE IMPUTATION FOR INT VALUES
# copy the dataframe with the columns that already have the most frequent values
df_fake_account_mfreq = pd.read_csv('dataset/fake_account_most_freq_imputer.csv')
df_fake_account_mice_complete = df_fake_account.copy(deep=True)
df_fake_account_mice_complete = df_fake_account_mice_complete*1
df_fake_account_mice_complete['is_private'] = df_fake_account_mfreq['is_private']
df_fake_account_mice_complete['is_business'] = df_fake_account_mfreq['is_business']
df_fake_account_mice_complete['is_recent_user'] = df_fake_account_mfreq['is_recent_user']


# take the columns to apply MICE from the original dataframe
df_fake_account_mice = df_fake_account.filter(['full_name_len', 'num_post', 'num_followers', 'num_followings', 'bio_len'], axis=1).copy(deep=True)

mice_imputer = IterativeImputer(estimator=linear_model.BayesianRidge(), missing_values=np.nan, n_nearest_features=None, imputation_order='ascending')
df_mice_imputed = pd.DataFrame(mice_imputer.fit_transform(df_fake_account_mice), columns=df_fake_account_mice.columns)


# copy the values obtained with MICE into the final dataframe where  previously were
# added the columns with the most frequent values

df_fake_account_mice_complete['full_name_len'] = df_mice_imputed['full_name_len'].astype(int)
df_fake_account_mice_complete['num_post'] = df_mice_imputed['num_post'].astype(int)
df_fake_account_mice_complete['num_followers'] = df_mice_imputed['num_followers'].astype(int)
df_fake_account_mice_complete['num_followings'] = df_mice_imputed['num_followings'].astype(int)
df_fake_account_mice_complete['bio_len'] = df_mice_imputed['bio_len'].astype(int)


df_fake_account_mice_complete = df_fake_account_mice_complete.iloc[:, 1:]
df_fake_account_mice_complete.to_csv('dataset/fake_account_MICE_imputer.csv', index_label='id')
print(df_fake_account_mice_complete.info())
print(df_fake_account_mice_complete.head())
print(df_fake_account_mice_complete.isnull().sum())
'''


'''
# merging the files
joined_files = os.path.join("out_final/", "*.csv")

# A list of all joined files is returned
joined_list = glob.glob(joined_files)

# Finally, the files are joined
df = pd.concat(map(pd.read_csv, joined_list), ignore_index=True)
df = df.set_index('user_id')
df.to_csv('out_final/fake_account_dataset.csv')
print(df)
'''




'''
# new indexing to adjust the first csv obtained during the first session
new_index = ['user_id','username','has_pic','username_len','username_digits',
             'is_private','full_name_len','num_post','num_followers','num_followings'
            ,'bio_len','is_business','is_recent_user','_merge']
csv1 = 'out/acc_1_data_scraped.csv'
df_1 = pd.read_csv(csv1)
df_1_formatted = df_1.reindex(columns=new_index)
df_1_formatted = df_1_formatted.set_index('user_id')
df_1_formatted.to_csv('out/acc_1_data_scraped_formatted.csv')
print(df_1_formatted)
'''

'''
# dataframe into existing CSV
df_1 = pd.read_csv('dataset/fake_account_most_freq_imputer.csv')
df_2 = pd.read_csv('dataset/fake_account_MICE_imputer.csv')

# deleting last two columns
df_1 = df_1.iloc[:, 1:-2]
df_2 = df_2.iloc[:, 1:-2]

print(df_1)
print(df_2)

# saving dataframe into a new csv
df_1.to_csv('dataset/fake_account_most_freq_imputer_original.csv', index_label='id')
df_2.to_csv('dataset/fake_account_MICE_imputer_original.csv', index_label='id')
'''

'''
# REAL DATASET
df_real_account = pd.read_csv('out/acc_real_IG_data_scraped.csv')
print(df_real_account.info())
print(df_real_account.head())

# drop the columns we don't need: userid, username, _merge
df_real_account = df_real_account.drop(['id', 'user_id', 'username', '_merge'], axis=1)
print(df_real_account)
df_real_account.to_csv('out_final/real_account_IG_dataset.csv', index_label='id')
'''


# ORIGINAL WITHOUT is_recet_user and is_business
df_real_account_original = pd.read_csv('out_final/real_account_IG_dataset.csv')
df_real_account_original = df_real_account_original.iloc[:, 1:-2]
df_real_account_original.to_csv('out_final/real_account_IG_dataset_original.csv', index_label='id')
print(df_real_account_original.info())
print(df_real_account_original.head())