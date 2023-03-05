from tqdm import tqdm
import torch


def train_model(dataset, train_loader, model, args):
    for epoch in range(args["epoch"]):
        model.train()
        # These are used to record information in training.
        train_loss = []
        train_accs = []
        for batch in tqdm(train_loader):
            appIds, userIds, review_embs, labels = batch
            review_embs = torch.permute(review_embs.to(args["device"]), (0, 2, 1))
            # review_embs = [batch_size, emb_dim, seq_length]
            logits = model(review_embs)