import re
import pandas as pd
import csv


def count_digits(stringa):
    pattern = r"\d"  # Pattern per cercare un singolo numero
    matches = re.findall(pattern, stringa)
    count = len(matches)
    return count


def save_usr_pic_into_csv(id_usr, usr, pc, usr_len, dgts, is_prvt, fnamelen):
    # saving info about followers
    with open('data/username_has_pic_real_1.csv', 'a', newline='', encoding="utf-8") as csvfile:          # with open('data/username_has_pic_acc8.csv', 'a', newline='', encoding="utf-8") as csvfile:   with open('data/username_has_pic_real.csv', 'a', newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow((id_usr, usr, pc, usr_len, dgts, is_prvt, fnamelen))


def save_followers_info_into_csv(user_ids, n_posts, n_flwr, n_flwing, bio_len, isbsnss, recent_user):
    # saving info about followers
    with open('data/followers_info_real_1.csv', 'a', newline='', encoding="utf-8") as csvfile:                # with open('data/followers_info_acc8.csv', 'a', newline='', encoding="utf-8") as csvfile:      with open('data/followers_info_real.csv', 'a', newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow((user_ids, n_posts, n_flwr, n_flwing, bio_len, isbsnss, recent_user))


def csv_to_df(csv_file):
    df_username = pd.read_csv(csv_file, header=None)
    return df_username


def df_to_list(df):
    list_usernames = df.iloc[:, 1].tolist()
    return list_usernames


def df_to_list_not_processed_usr(df):
    list_usr_not_processed = df.iloc[:, 0].tolist()
    return list_usr_not_processed


def merge_csv(csv1, csv2):
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)
    merged_df = pd.merge(df1, df2, on='user_id', how='outer', indicator=True)

    # Filtrare le righe che rimangono fuori dal dataframe
    non_merged_rows = merged_df[merged_df['_merge'] != 'both']

    # Stampa gli elementi che rimangono fuori
    print(non_merged_rows)
    return merged_df
