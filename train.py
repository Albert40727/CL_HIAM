from tqdm import tqdm
import torch


def train_model(args, train_loader, user_network_model, item_network_model, co_attention_network):
    for epoch in range(args["epoch"]):
        user_network_model.train()
        item_network_model.train()
        # These are used to record information in training.
        train_loss = []
        train_accs = []
        review_emb_dict = []
        for batch in tqdm(train_loader):
            userId, itemId, user_review_emb, item_review_emb, user_lda_groups, item_lda_groups, y = batch
            user_logits = user_network_model(user_review_emb.to(args["device"]), user_lda_groups.to(args["device"]))
            item_logits = item_network_model(item_review_emb.to(args["device"]), item_lda_groups.to(args["device"]))
            print(user_logits.size(), item_logits.size())
            weighted_user_logits,  weighted_item_logits= co_attention_network(user_logits, item_logits)
            # print(weighted_user_logits.size(), weighted_user_logits.size())