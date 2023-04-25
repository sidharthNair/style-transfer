import numpy as np
import torch
import config
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from features import VGG16Features, gram_matrix
from model import Net
from data import CustomDataset, load_img, GENERAL_TRANSFORM
from test import test_net

torch.backends.cudnn.benchmark = True

features = VGG16Features()
mse_loss = nn.MSELoss()
style_img = GENERAL_TRANSFORM(load_img(config.STYLE_IMG)).repeat(config.BATCH_SIZE, 1, 1, 1).to(config.DEVICE)
style_features = features(style_img)
style_representation = [gram_matrix(x) for x in style_features]

def train_batch(net, optimizer, batch):
    batch = batch.to(config.DEVICE)
    stylized_batch = net(batch)

    content_batch_features = features(batch)
    stylized_batch_features = features(stylized_batch)
    stylized_batch_representation = [gram_matrix(x) for x in stylized_batch_features]

    # Content loss defined as the MSELoss between relu2_2 layer feature maps
    content_loss = config.CONTENT_WEIGHT * mse_loss(content_batch_features[1], stylized_batch_features[1])

    style_loss = 0.0
    for i in range(len(style_representation)):
        style_loss += mse_loss(style_representation[i], stylized_batch_representation[i])
    style_loss /= len(style_representation)
    style_loss *= config.STYLE_WEIGHT

    loss = content_loss + style_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.cpu().detach().numpy(), content_loss.cpu().detach().numpy(), style_loss.cpu().detach().numpy(), 0.0

def save(filename, model, optimizer, losses, content_losses, style_losses, tv_losses):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "losses": losses,
        "content_losses": content_losses,
        "style_losses": style_losses,
        "tv_losses": tv_losses
    }
    torch.save(checkpoint, filename)


def load(filename, model, optimizer, losses, content_losses, style_losses, tv_losses, lr):
    checkpoint = torch.load(filename, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    losses.extend(checkpoint["losses"])
    content_losses.extend(checkpoint["content_losses"])
    style_losses.extend(checkpoint["style_losses"])
    tv_losses.extend(checkpoint["tv_losses"])
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def main():
    dataset = CustomDataset("dataset")
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
    )

    net = Net().to(config.DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=config.LEARNING_RATE)

    losses = []
    content_losses = []
    style_losses = []
    tv_losses = []

    if config.LOAD:
        print('LOADING MODELS FROM CHECKPOINT FILES')
        load(config.CHECKPOINT_FILE, net, optimizer, losses, content_losses, style_losses, tv_losses, config.LEARNING_RATE)

    net.train()

    batch_count = 0
    base = len(losses) // config.BATCH_SIZE
    version = len(losses)
    for epoch in range(config.EPOCHS):
        tqdm_loader = tqdm(loader, leave=True)
        for batch in tqdm_loader:
            loss, content_loss, style_loss, tv_loss = train_batch(net, optimizer, batch)
            losses.append(loss)
            content_losses.append(content_loss)
            style_losses.append(style_loss)
            tv_losses.append(tv_loss)

            if config.SAVE and (batch_count % config.SAVE_FREQ) == 0:
                save(config.CHECKPOINT_FILE, net, optimizer, loss, content_loss, style_loss, tv_loss)
                np.savetxt(config.LOSSES_FILE, np.column_stack((losses, content_losses, style_losses, tv_losses)), delimiter=',', fmt='%s', header='loss, content_loss, style_loss, tv_loss')

            if config.TEST and (batch_count % config.TEST_FREQ) == 0:
                test_net(net, prefix=f'{version}_')
                version += 1
            batch_count += 1
            tqdm_loader.set_description(f'Epoch: {base + epoch}, Loss: {loss:.4f}, CL: {content_loss:.4f}, SL: {style_loss:.4f}, TVL: {tv_loss:.4f}', refresh=True)

if __name__ == "__main__":
    main()
