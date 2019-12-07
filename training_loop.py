import torch
from torch import nn
from torch.optim import Adam
from tqdm import trange

from loader import Loader_HeySnips
from model import KeyWordSpotter


def training_loop(model, train_loader, val_loader, num_epochs=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-4, betas=(.9, .999), eps=1e-8, weight_decay=1e-5, amsgrad=False)
    optimizer.zero_grad()
    L1_criterion = nn.L1Loss(reduction='none')

    train_batches_per_epoch = train_loader.num_batches
    val_batches_per_epoch = val_loader.num_batches

    for epoch in range(num_epochs):
        model.train()
        print("epoch: ", epoch)
        bar = trange(train_batches_per_epoch)
        avg_epoch_loss = 0.0
        avg_acc = 0.0
        for batch in bar:
            x_np, y_np = train_loader.get_batch()
            x = torch.from_numpy(x_np).float().to(device)
            y = torch.from_numpy(y_np).float().to(device)

            out = model(x)

            loss = weighted_L1_loss(L1_criterion, out, train_batches_per_epoch, y)
            loss.backward()
            avg_epoch_loss += loss.item() / train_batches_per_epoch
            acc = float((out[y == 0] < .5).sum() + (out[y == 1] > .5).sum()) / float(y.shape[0])
            avg_acc += acc / float(train_batches_per_epoch)
            curr_loss = loss.item()
            optimizer.step()
            optimizer.zero_grad()
            bar.set_description("epoch: %d, loss: %f, acc: %f " % (epoch, curr_loss, acc))
        bar.set_description("epoch: %d, loss: %f, acc: %f " % (epoch, avg_epoch_loss, avg_acc))

        model.eval()
        with torch.no_grad():
            avg_epoch_val_loss = 0.0
            avg_acc = 0.0
            bar = trange(val_batches_per_epoch)
            for batch in bar:
                x_np, y_np = val_loader.get_batch()
                x = torch.from_numpy(x_np).float().to(device)
                y = torch.from_numpy(y_np).float().to(device)

                out = model(x)

                loss = weighted_L1_loss(L1_criterion, out, val_batches_per_epoch, y)
                acc = float((out[y == 0] < .5).sum() + (out[y == 1] > .5).sum()) / float(y.shape[0])
                avg_acc += acc / float(val_batches_per_epoch)
                avg_epoch_val_loss += loss.item() / val_batches_per_epoch
                bar.set_description("epoch: %d, val_loss: %f, val_acc: %f " % (epoch, curr_loss, acc))
            bar.set_description("epoch: %d, val_loss: %f, val_acc: %f " % (epoch, avg_epoch_val_loss, avg_acc))

        print(f"epoch: {epoch}, loss: {avg_epoch_loss}, val_loss:{avg_epoch_val_loss}, val_acc: {avg_acc}")
    torch.save(model.state_dict(),
               "./weights/epoch_%d_%f_%f.weights" % (epoch, avg_epoch_val_loss, avg_acc))

    hi = 5


def weighted_L1_loss(L1_criterion, out, batches_per_epoch, y):
    eps = (.001 / batches_per_epoch)
    return (L1_criterion(out, y) * (y + y.mean() + eps)).mean()


def train():
    train_loader = Loader_HeySnips("/mnt/h/speech_detection/hey_snips_fl_5.0/hey_snips_fl_amt/train.json",
                                   "/mnt/h/speech_detection/hey_snips_fl_5.0/hey_snips_fl_amt/mfcc/", batch_size=100)
    val_loader = Loader_HeySnips("/mnt/h/speech_detection/hey_snips_fl_5.0/hey_snips_fl_amt/test.json",
                                 "/mnt/h/speech_detection/hey_snips_fl_5.0/hey_snips_fl_amt/mfcc/", batch_size=100)
    model = KeyWordSpotter(20)
    training_loop(model, train_loader, val_loader, num_epochs=5)


if __name__ == '__main__':
    train()
