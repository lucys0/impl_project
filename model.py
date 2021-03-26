import torch
import torch.nn as nn
import numpy as np

class Encoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, image_resolution=64):
        super(Encoder, self).__init__()

        # assume in_channels=1, out_channels=4, image_resolution=64
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, 4, kernel_size=3, stride=2)] # input: 64*64*1
        )

        self.convs.append(nn.Conv2d(4, 8, kernel_size=3, stride=2))  # 32*32*4
        self.convs.append(nn.Conv2d(8, 16, kernel_size=3, stride=2))  # 16*16*8
        self.convs.append(nn.Conv2d(16, 32, kernel_size=3, stride=2)) # 8*8*16
        self.convs.append(nn.Conv2d(32, 64, kernel_size=3, stride=2)) # 4*4*32
        self.convs.append(nn.Conv2d(64, 128, kernel_size=1, stride=2)) # 2*2*64

        # the final 1x1 feature vector gets mapped to a 64-dimensional observation space
        self.fc = nn.Linear(in_features=128, out_features=64)  # input: 1*1*128 output: 64

    # x is the observation at one time step
    def forward(self, x, detach=False):
        for i in range(6):
            x = torch.tanh(self.convs[i](x))
        x = self.fc(x.squeeze())

        # freeze the encoder
        if detach:
            x.detach()
        return x


# Build a 3-layer feedforward neural network
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_units=32):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_units)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_units, output_size)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        hidden_layer = self.relu(self.fc1(x))
        output_layer = self.tanh(self.fc2(hidden_layer))
        return output_layer


# Build a single-layer LSTM        
class LSTM(nn.Module):
    def __init__(self, sequence_length, input_dim=64, hidden_dim=64, layer_dim=1, output_dim=64):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.sequence_length = sequence_length

        # batch_first=True => input/output tensors of shape (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        # self.fc = nn.Linear(hidden_dim, output_dim) # if any prob later, maybe use it

    def forward(self, x): # x should be (64,)
        # modify the size of input cells, batch_dim=1
        input = x.unsqueeze(0).repeat(self.sequence_length, 1).unsqueeze(0) # shape: (1, T, 64)
        # initialize hidden and cell states with zeros
        h0 = torch.zeros(self.layer_dim, input.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, input.size(0), self.hidden_dim)

        out, _ = self.lstm(input, (h0.detach(), c0.detach()))
        return out.squeeze(0) # return (T, hidden)


class Model(nn.Module):
    def __init__(self, time_steps, frames, tasks, image_resolution, device):
        super(Model, self).__init__()
        self.encoder = Encoder().to(device)
        self.mlp = MLP(input_size=image_resolution*frames, output_size=image_resolution).to(device)
        self.lstm = LSTM(sequence_length=time_steps).to(device)
        self.T = time_steps
        self.N = frames
        self.K = tasks
        self.reward_heads = [MLP(input_size=time_steps*image_resolution, output_size=time_steps*1).to(device)] * tasks
        self.image_resolution = image_resolution
        self.device = device
        self.loss = nn.MSELoss()
        self.decoder = Decoder().to(device)

    # return the prediction of T future rewards given N conditioning frames
    def forward(self, obs):
        # first obtain the representation z_t at each time step using encoder
        z = []
        for frame in range(self.N):
            # obs corresponds to the observation at N conditioning frames: (N, 64, 64)
            # so obs[frame] is the observation at the current frame, represented as a (64, 64) tensor
            # the input to encoder should be in (1, 1, 64, 64)
            z_frame = self.encoder(obs[frame][None, None, :]) #tensor(64,)
            z.append(z_frame)
        z = torch.stack(z, dim=0) # (N, 64)

        z_mlp = self.mlp(z.flatten()) # (64,)
        h = self.lstm(z_mlp) # (T, 64)
        # z_mlp_copy = z_mlp.detach().clone()
                
        # then feed h to each reward head to predict the reward of all time steps for every task
        reward_predicted = []
        h = h.flatten()
        for task in range(self.K):
            # reward_heads is a list of K MLP's
            reward_head = self.reward_heads[task]        
            r_t = reward_head(h) # (T,) 
            reward_predicted.append(r_t)
        reward_predicted = torch.stack(reward_predicted, dim=0) # should be (K, T)
        return reward_predicted
                       
    def criterion(self, reward_predicted, reward_targets):
        reward_predicted = reward_predicted.squeeze()
        assert reward_predicted.shape == reward_targets.shape
        return self.loss(reward_predicted, reward_targets)

    def test_decode(self, traj_images):
        output = []
        for t in range(self.T):
            # the input to encoder should be in (1, 1, 64, 64)
            # print("\\\\", traj_images[t].max())
            z_t = self.encoder(traj_images[t][None, None, :], detach=True) #tensor(64,)
            decoded_img = self.decoder(z_t).squeeze()
            # print("////", decoded_img.max())
            output.append(decoded_img.detach().numpy())

        return np.array(output)

# add a detached decoder network and train the model on a single reward
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.tfc = nn.Linear(64, 128)
        self.tconvs = nn.ModuleList(
            [nn.ConvTranspose2d(128, 64, kernel_size=1, stride=2, output_padding=1)]
        )

        self.tconvs.append(nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.tconvs.append(nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.tconvs.append(nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.tconvs.append(nn.ConvTranspose2d(8, 4, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.tconvs.append(nn.ConvTranspose2d(4, 1, kernel_size=3, stride=2, padding=1, output_padding=1))

    def forward(self, x):
        x = self.tfc(x).view(1, 128, 1, 1)
        for i in range(6):
            x = torch.tanh(self.tconvs[i](x))

        # output: (1, 1, 64, 64)
        return x