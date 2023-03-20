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
            userId, itemId, user_review_emb, item_review_emb, user_lda_groups, item_lda_groups, user_mf_emb, item_mf_emb, y = batch
            user_logits = user_network_model(user_review_emb.to(args["device"]), user_lda_groups.to(args["device"]))
            item_logits = item_network_model(item_review_emb.to(args["device"]), item_lda_groups.to(args["device"]))
            weighted_user_logits,  weighted_item_logits = co_attention_network(user_logits.to(args["device"]), item_logits.to(args["device"]))
            user_feature = torch.cat((weighted_user_logits.to(args["device"]), user_mf_emb.to(args["device"])), dim=1)
            item_feature = torch.cat((weighted_item_logits.to(args["device"]), item_mf_emb.to(args["device"])), dim=1)
            print(user_feature.size(), item_feature.size())