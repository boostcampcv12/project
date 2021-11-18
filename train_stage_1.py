from os import scandir
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random
from module import CEModule
import argparse
import os


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def save_model(model, saved_dir, file_name):
    check_point = {'net': model.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model, output_path)


def train(encoder, decoder, args):
    seed_everything(args.seed)
    data_dir = args.sketch_dir

    # -- settings
    use_cuda = torch.cuda.in_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset, data loader -> 각 part 에 맞는 sketch를 잘라서 받아온다.

    train_loader = DataLoader()
    val_loader = DataLoader()

    model = decoder(encoder)

    critetrion = nn.MSELoss()
    optimizer = torch.optim.AdamW
    shcheduler = torch.optim.lr_scheduler.CosineAnnealingLR

    for epoch in range(200):
        model.train()
        loss_value = 0

        for inputs in train_loader:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outs = model(inputs)

            loss = critetrion(outs, inputs)
            loss.backward()
            optimizer.step()

            loss_value += loss.item()

        shcheduler.step(float(loss))

        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            total_loss = 0.0
            cnt = 0

            for inputs in val_loader:
                inputs = inputs.to(device)
                outs - model(inputs)
                loss = critetrion(inputs, outs)
                total_loss += loss
                cnt += 1

            avrg_loss = total_loss / cnt
            print(
                f"Validation #{epoch} Average Loss : {round(avrg_loss.item(),4)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str, default='mouth',
                        help="Choose part name to encode")
    parser.add_argument('--seed', type=int, default=21, help="Fixing seed")
    parser.add_argument("--sketch_dir", type=str,
                        default="None", help="Loactaion of Sketch")

    part = parser.part
    args = parser.parse_args()

    part_encoder = CEModule.define_part_encoder(
        model=part, norm='instance', input_nc=1, latent_dim=512)
    part_decoder = CEModule.define_part_decoder(
        model=part, norm='instance', output_nc=1, latent_dim=512)

    train(part_encoder, part_decoder, args)
