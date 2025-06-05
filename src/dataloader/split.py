import numpy as np
from sklearn.model_selection import train_test_split
import pdb

def data_split(user2items, user2scores, user_count):
    all_user_ids = [i for i in range(1, (user_count + 1))] # this is all indices
    np.random.seed(0)
    # cold-start users split 20 for training
    train_user_ids, val_test_user_ids = train_test_split(all_user_ids, test_size=0.8, shuffle=True, random_state=42)
    val_user_ids, test_user_ids = train_test_split(sorted(val_test_user_ids), test_size=0.8, shuffle=True, random_state=42)

    train_data = {}
    for cur_user_id in sorted(train_user_ids):
        train_data[cur_user_id] = [user2items[cur_user_id], user2scores[cur_user_id]]

    val_data = {}
    for cur_user_id in sorted(val_user_ids):
        val_data[cur_user_id] = [user2items[cur_user_id], user2scores[cur_user_id]]
    
    test_data = {}
    for cur_user_id in sorted(test_user_ids):
        test_data[cur_user_id] = [user2items[cur_user_id], user2scores[cur_user_id]]
    
    return train_data, val_data, test_data