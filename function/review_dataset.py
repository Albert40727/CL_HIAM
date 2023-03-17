from torch.utils.data import Dataset
import numpy as np
import pandas as pd 
import torch
import os 

class ReviewDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.review_df = pd.read_pickle(args["data_dir"])
        # self.start = int(file_name.split(".")[0].split("_")[2].split("-")[0]) # ex: "filtered_reviews_500-599.h5" -> 500
      
    def __getitem__(self, idx):

        userId = self.review_df["UserID"][idx]
        itemId = self.review_df["AppID"][idx]
        y = self.review_df["Like"][idx]

        user_review_data = pd.read_pickle(os.path.join(self.args["user_data_dir"], str(userId)+".pkl"))
        item_review_data = pd.read_pickle(os.path.join(self.args["item_data_dir"], str(itemId)+".pkl"))

        user_review_emb = torch.from_numpy(np.array(user_review_data["SplitReview_emb"].tolist()))
        item_review_emb = torch.from_numpy(np.array(item_review_data["SplitReview_emb"].tolist()))
        pad_user_emb = torch.zeros(self.args["max_review_user"], user_review_emb.size(1), user_review_emb.size(2))
        pad_item_emb = torch.zeros(self.args["max_review_item"], item_review_emb.size(1), item_review_emb.size(2))

        user_lda_groups = torch.from_numpy(np.array(user_review_data["LDA_group"].tolist()))
        item_lda_groups = torch.from_numpy(np.array(item_review_data["LDA_group"].tolist()))
        pad_user_lda = torch.zeros(self.args["max_review_user"], user_lda_groups.size(1))
        pad_item_lda = torch.zeros(self.args["max_review_item"], item_lda_groups.size(1))

        if user_review_emb.size(0) > self.args["max_review_user"]:
            pad_user_emb = user_review_emb[:self.args["max_review_user"], :, :]
        else:
            pad_user_emb[:user_review_emb.size(0), :, :] = user_review_emb

        if item_review_emb.size(0) > self.args["max_review_item"]:
            pad_item_emb = item_review_emb[:self.args["max_review_item"], :, :]
        else:
            pad_item_emb[:item_review_emb.size(0), :, :] = item_review_emb

        if user_lda_groups.size(0) > self.args["max_review_user"]:
            pad_user_lda = user_lda_groups[:self.args["max_review_user"], :]
        else:
            pad_user_lda[:user_lda_groups.size(0), :] = user_lda_groups

        if item_lda_groups.size(0) > self.args["max_review_item"]:
            pad_item_lda = item_lda_groups[:self.args["max_review_item"], :]
        else:
            pad_item_lda[:item_lda_groups.size(0), :] = item_lda_groups

        return userId, itemId, pad_user_emb, pad_item_emb, pad_user_lda, pad_item_lda, y

    def __len__(self):
        return len(self.review_df)