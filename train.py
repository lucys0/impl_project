import argparse
import os
import torch
import torch.nn as nn
import gym
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import cv2
from general_utils import make_image_seq_strip
from sprites_datagen.rewards import *
from model import Model, Decoder
from sprites_datagen import moving_sprites 
from general_utils import AttrDict
from sprites_env.envs import sprites


def train(model, obs, reward_targets, optimizer):
    optimizer.zero_grad()
    reward_predicted = model(obs)
    loss = model.criterion(reward_predicted, reward_targets)
    loss.backward()
    optimizer.step()

    return loss.item()

# dataloader
def dataloader(image_resolution, time_steps, batch_size):
    spec = AttrDict(
        resolution=image_resolution,
        max_seq_len=time_steps, # such that there is a reward target for each time step
        max_speed=0.05,      # total image range [0, 1]
        obj_size=0.2,       # size of objects, full images is 1.0
        shapes_per_traj=1,      # number of shapes per trajectory
        rewards=[VertPosReward],
    )
    # gen = moving_sprites.DistractorTemplateMovingSpritesGenerator(spec)
    gen = moving_sprites.TemplateMovingSpritesGenerator(spec)
    traj = gen.gen_trajectory()
    img = make_image_seq_strip([traj.images[None, :, None].repeat(3, axis=2).astype(np.float32)], sep_val=255.0).astype(np.uint8)  
    cv2.imwrite("ground_truth.png", img[0].transpose(1, 2, 0))

    dataset = moving_sprites.MovingSpriteDataset(spec)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dl, torch.from_numpy(traj.images.astype(np.float32) / (255./2) - 1.0 )

# argument parser
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--image_resolution', type=int, default=64)
    parser.add_argument('--time_steps', type=int, default=5)
    parser.add_argument('--tasks', type=int, default=1)
    parser.add_argument('--conditioning_frames', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=1) #or?
    parser.add_argument('--env', type=str, default='Sprites-v0')
    args = parser.parse_args()
    return args

def main():
    # parse arguments
    args = parse_args()

    f = args.conditioning_frames
    t = args.time_steps
    # load data
    dl, traj_images = dataloader(args.image_resolution, t, args.batch_size)

    data = torch.zeros(len(dl), t-f, f+1, args.image_resolution, args.image_resolution)  
    for i, batch in enumerate(dl):
        traj = batch['images'].squeeze() # traj: (max_seq_len=t, 3, 64, 64) 
        for k in range(t-f):
            data[i][k] = traj[k:k+f+1, 0, :, :].squeeze()

    # initialize the environment
    env = gym.make(args.env)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    model = Model(t, f+1, args.tasks, args.image_resolution, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    train_loss = []

    for epoch in range(args.num_epochs):
        running_loss = 0.0
        for i, batch in enumerate(dl):
            # average_loss_traj = 0.0
            reward_targets = batch['rewards']['vertical_position'].squeeze() # change when needed
            for k in range(t-f):
                obs = data[i][k] # (f+1, 64, 64)               
                loss = train(model, obs, reward_targets, optimizer) # here it's assumed that there's only one task - how to fix it?
                running_loss += loss
            # average_loss_traj += loss
        
        # print or store data
        running_loss = running_loss / (len(dl)*(t-f))
        print('Epoch: {} \tLoss: {:.6f}'.format(epoch, running_loss))
        train_loss.append(running_loss)

    # visualize results: loss
    plt.figure()
    plt.plot(train_loss)
    plt.title('Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('train_loss.png')

    # decode and generate images with respect to reward functions
    # output = model.decode(traj_images).reshape((30, 64*64))
    # scaler = MinMaxScaler(feature_range=(0, 255))
    # output = scaler.fit_transform(output).reshape((30, 64, 64))
    output = model.decode(traj_images) * 255

    # print(output.min())
    # print("---------------------------------")
    img = make_image_seq_strip([output[None, :, None].repeat(3, axis=2).astype(np.float32)], sep_val=255.0).astype(np.uint8)   
    # print(img[0].transpose(1, 2, 0))
    cv2.imwrite("decode.png", img[0].transpose(1, 2, 0))


if __name__ == '__main__':
    main()
