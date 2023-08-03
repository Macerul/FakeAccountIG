from ensta import Host, NewSessionID
from ensta import Exceptions
from auth import USERNAME, PASSWORD
from utils import *

sessionFile = "sessiondir/session.txt"

# load the previous session if exists
with open("sessiondir/session.txt", "r") as file:
    session_id = file.read().strip()
    try:
        host = Host(session_id)
    except Exceptions.SessionError:
        session_id = NewSessionID(USERNAME, PASSWORD)

        with open(sessionFile, "w") as file:
            file.write(session_id)

        host = Host(session_id)

# you can pass either username or userid
# followers = host.followers(60361042819)  # To fetch the full list, don't specify 'count'
followings = host.followings(1950939923, count=100) # real account of dinfunisa 8227805616 - 100 unisalerno
'''
# run this if you want to retrieve list of fake accounts
if followers is None:
    print('unable to get followers')
else:
    for user in followers:
        if user is None:
            print("Unable to fetch user data")
        else:
            usr_id = user.user_id
            follower = user.username
            anon_pic = user.has_anonymous_profile_picture
            username_len = len(follower)
            username_digits = count_digits(follower)
            is_private = user.is_private
            full_name_len = len(user.full_name)
            save_usr_pic_into_csv(usr_id, follower, anon_pic, username_len, username_digits, is_private, full_name_len)

'''


# run this to retrieve the real followings of a user
if followings is None:
    print('unable to get followings')
else:
    for user in followings:
        if user is None:
            print("Unable to fetch user data")
        else:
            usr_id = user.user_id
            follower = user.username
            anon_pic = user.has_anonymous_profile_picture
            username_len = len(follower)
            username_digits = count_digits(follower)
            is_private = user.is_private
            full_name_len = len(user.full_name)
            save_usr_pic_into_csv(usr_id, follower, anon_pic, username_len, username_digits, is_private, full_name_len)


