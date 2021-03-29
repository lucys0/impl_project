import torch
from torch.utils.data import Dataset, DataLoader
from general_utils import make_image_seq_strip
from sprites_datagen.rewards import *
from sprites_datagen import moving_sprites 
from general_utils import AttrDict

class TrainingDataset(Dataset):
    def __init__(self, dataset, frames, time_steps):
        super(TrainingDataset, self).__init__()
        self.dataset = dataset
        self.frames = frames
        self.time_steps = time_steps
        preprocessed_dataset = []
        for i in range(len(dataset)):
            traj = dataset[i]
            for t in range(frames, time_steps):
                data = AttrDict()
                data.rewards = traj['rewards']['horizontal_position'] # change when needed
                data.obs = traj['images'][t-frames:t+1, 0, :, :].squeeze()
                assert len(data.obs) == frames+1
                data.states = traj['states'][t-frames:t+1, :].squeeze()
                preprocessed_dataset.append(data)
        self.preprocessed_dataset = preprocessed_dataset    

    def __getitem__(self, index):  
        return self.preprocessed_dataset[index]

    def __len__(self):
        return len(self.dataset)*(self.time_steps-self.frames)

# dataloader
def dataloader(image_resolution, time_steps, batch_size, frames):
    spec = AttrDict(
        resolution=image_resolution,
        max_seq_len=time_steps, # such that there is a reward target for each time step
        max_speed=0.1,      # total image range [0, 1]
        obj_size=0.2,       # size of objects, full images is 1.0
        shapes_per_traj=1,      # number of shapes per trajectory
        rewards=[HorPosReward],
    )
    # gen = moving_sprites.DistractorTemplateMovingSpritesGenerator(spec)
    gen = moving_sprites.TemplateMovingSpritesGenerator(spec)
    traj = gen.gen_trajectory()
    img = make_image_seq_strip([traj.images[None, :, None].repeat(3, axis=2).astype(np.float32)], sep_val=255.0).astype(np.uint8)  
    # cv2.imwrite("ground_truth.png", img[0].transpose(1, 2, 0))

    dataset = moving_sprites.MovingSpriteDataset(spec)
    preprocessed_dataset = TrainingDataset(dataset, frames, time_steps)

    dl = DataLoader(preprocessed_dataset, batch_size=batch_size, shuffle=True)
    return dl, torch.from_numpy(traj.images.astype(np.float32) / (255./2) - 1.0), img[0]
