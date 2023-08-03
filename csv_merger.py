from utils import merge_csv
import csv
import pandas as pd

intestazione_host = ['user_id', 'username', 'has_pic', 'username_len', 'username_digits',
                     'is_private', 'full_name_len']


intestazione_guest = ['user_id', 'num_post', 'num_followers', 'num_followings', 'bio_len',
                      'is_business', 'is_recent_user']


# File CSV originali senza intestazione
host_file = 'data/username_has_pic_real_1.csv'
guest_file = 'data/followers_info_real_1.csv'


# Lettura dei dati dal file_host CSV originale
with open(host_file, 'r') as f_originale:
    dati_originali = list(csv.reader(f_originale))

# Scrittura dei dati nel nuovo file CSV con l'intestazione
with open('data/formatted_host_file_real_1.csv', 'w', newline='') as f_con_intestazione:
    writer = csv.writer(f_con_intestazione)
    # Scrittura dell'intestazione
    writer.writerow(intestazione_host)
    # Scrittura dei dati originali
    writer.writerows(dati_originali)



# Lettura dei dati dal file_guest CSV originale
with open(guest_file, 'r') as f_originale:
    dati_originali = list(csv.reader(f_originale))

# Scrittura dei dati nel nuovo file CSV con l'intestazione
with open('data/formatted_guest_file_real_1.csv', 'w', newline='') as f_con_intestazione:
    writer = csv.writer(f_con_intestazione)
    # Scrittura dell'intestazione
    writer.writerow(intestazione_guest)
    # Scrittura dei dati originali
    writer.writerows(dati_originali)


csv_host = 'data/formatted_host_file_real_1.csv'
csv_guest = 'data/formatted_guest_file_real_1.csv'

df_merged = merge_csv(csv_host, csv_guest)
df_merged.to_csv('out/acc_real_data_scraped_new.csv', index_label='id')
df_merged = df_merged.drop_duplicates(subset='user_id')
print(df_merged)


