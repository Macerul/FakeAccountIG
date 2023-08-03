from ensta import Guest

from sel_guest_scraper import list_username_not_proc
from utils import df_to_list, csv_to_df, save_followers_info_into_csv, df_to_list_not_processed_usr
import random
import time
import pandas as pd

time_to_sleep = False

guest = Guest()

# run this to retrieve as much user as you can after running the host script
df_username = csv_to_df('data/username_has_pic_real_1.csv')    # df_username = csv_to_df('data/username_has_pic_acc8.csv')        df_username = csv_to_df('data/username_has_pic_real.csv')
print(df_username)
list_username = df_to_list(df_username)
print(list_username)

# run this when you want to retrieve the username you failed to process
# df_username_not_proc = csv_to_df('data/user_not_processed_real_2.csv')          # df_username_not_proc = csv_to_df('data/user_not_processed_acc8.csv')         df_username_not_proc = csv_to_df('data/user_not_processed_real5.csv')
# print(df_username_not_proc)
# list_username_not_proc = df_to_list_not_processed_usr(df_username_not_proc)
# print(list_username_not_proc)


user_not_fetched = []
for usr in list_username:         # list_username_not_proc when you want to iterate the username not processed yet
    profile = guest.profile(usr)  # Access full profile data
    if profile is None:
        print("Unable to fetch user data")
        user_not_fetched.append(usr)
    else:
        print("fetching info")
        user_id = profile.user_id
        num_posts = profile.total_post_count
        num_followers = profile.follower_count
        num_followings = profile.following_count
        bio_len = len(profile.biography)
        is_business = profile.is_business_account
        has_joined_recently = profile.is_joined_recently
        save_followers_info_into_csv(user_id, num_posts, num_followers, num_followings,
                                     bio_len, is_business, has_joined_recently)

# salvo in un dataframe la lista degli utenti che non sono riuscita a processare e li
# salvo in un csv in modo tale che posso rieseguire il codice partendo direttamente dagli
# utenti non ancora processati
df = pd.DataFrame(user_not_fetched)
df.to_csv('data/user_not_processed_real_test.csv', index=False)      # df.to_csv('data/user_not_processed_acc8_residual.csv', index=False)      df.to_csv('data/user_not_processed_real6.csv', index=False)
