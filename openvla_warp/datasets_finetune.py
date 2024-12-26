
from builtins import super
try:
    from Dataset_Lab.LabDataset import LabDataset
    from Dataset_Droid.DroidDataset_new import DroidDataset
    from Dataset_Sim.SimDataset import SimDatasetDumpy
except:
    pass
from Datasets.calvin_dataset import CalvinDataset_Policy


import pickle
import random
import numpy as np
import torch
import os

def convert_data_structure(trajs, batch_transform):
    output_dict = {}
            # import ipdb;ipdb.set_trace()
    output_dict['dataset_name'] = 'droid'
    import cv2
    img = trajs['observation']['image'].cpu().numpy() # 180 320 3
    # img = np.pad(img, ( (70, 70), (0, 0), (0, 0)), 'constant', constant_values=0)
    output_dict['observation'] = {}
    output_dict['observation']['image_primary'] = img
            # output_dict['pixel_values'] = cv2.resize(img, (224, 224))
            # input_size = [182, 322]
    from pytorch3d.transforms import quaternion_to_axis_angle, matrix_to_euler_angles, quaternion_to_matrix           
    # trajs['action']['rotation_delta'][0, 0][0] += 1.0

    low = torch.tensor([-0.432188, -0.545456,  0.293439, -3.141593, -0.811348, -3.141573, -1. ]).to(torch.float32)
    high = torch.tensor([0.42977 , 0.139396, 0.796262, 3.141592, 0.638583, 3.141551,        1. ]).to(torch.float32)
    #low = torch.tensor([-0.04222496, -0.04689935, -0.0431233 , -0.10854903, -0.13731207, -0.11201588, 0])
    #/mnt/petrelfs/share_data/zhangtianyi1/task_ABC_D/validation/statistics.yaml


    # import ipdb;ipdb.set_trace()
    output_dict['action'] = torch.cat([trajs['action']['world_vector'], 
                                trajs['action']['rotation_delta'], 
                                ( 1 - trajs['action']['gripper_closedness_action']).to(torch.float32)], dim=-1)
    output_dict['action'] = torch.clip(2 * (output_dict['action'] - low) / (high - low + 1e-8) -1, min=-1, max=1.).cpu().numpy()
    # tf.clip_by_value(2 * (x - low) / (high - low + 1e-8) - 1, -1, 1),
    # 32 16 3
    # output_dict['action'] = torch.cat([(trajs['action']['world_vector'][0, 0] / 0.1024), 
    #                             quaternion_to_axis_angle(trajs['action']['rotation_delta'][0, 0]) / 0.3232, 
    #                             ( 1 - trajs['action']['gripper_closedness_action'][0,0]).to(torch.float32)], dim=-1).cpu().numpy()[None,]
    output_dict['task'] = {}
    output_dict['task']['language_instruction'] = trajs['instruction']
    output_dict['action_past_goal'] = torch.sum(trajs['action']['loss_weight'], dim=-1) == 0
    return batch_transform(output_dict)



def convert_data_structure_calvin(trajs, batch_transform):
    output_dict = {}
            # import ipdb;ipdb.set_trace()
    output_dict['dataset_name'] = 'calvin'
    import cv2
    # trajs {'goal_rgb': tensor([[[]]]), 
    # 'rgb': tensor([[[[]]]], dtype=torch.uint8), 
    # 'hand_rgb': tensor([[[[ ]]]]), 
    # 'state': tensor([[0.0991, -0.2634,  0.3280,  3.1240, -0.0568,  1.4828,  0.0000]]), 
    # 'rel_state': tensor([[0., 0., 0., 0., 0., 0., 0.]]), 
    
    # 'action': {'action': tensor([[[-0.3476, -0.0011, -0.1365,  0.0298,  0.0386,  0.0151]]]), 
    # 'gripper_closedness_action': tensor([[[1.]]]), 
    # 'loss_weight': tensor([[[1]]]), 
    # 'terminate_episode': tensor([[[0., 0., 0.]]]), 
    # 'abs_tar_pose': tensor([[[0.0991, -0.2634,  0.3280,  3.1240, -0.0568,  1.4828,  0.0000]]])}, 
    
    # 'attention_mask': tensor([1]), 
    # 'action_mask': tensor([[1]]), 
    # 'text': ['push the button to turn on the led light'], 
    # 'progress': tensor([0.1111]), 
    
    # 'observation': {'image': tensor([[[[[]]]]], dtype=torch.uint8), 
    # 'wrist_image': tensor([[[[[ ]]]]]), 
    # 'camera_extrinsic_cv': tensor([[[[ 2.0507,  0.0892,  0.5913,  1.0058],
        #   [-0.6710, -1.3544,  0.1531, -1.0456],
        #   [ 1.1260,  0.2386,  0.1055,  0.5790],
        #   [-1.5899, -0.0336,  1.4321, -1.7794]]]]}, 
    
    # 'instruction': 'push the button to turn on the led light'}
    img = trajs['observation']['image'].cpu().numpy() # 180 320 3
    # print(img.min(), img.max())
    # img = np.pad(img, ( (70, 70), (0, 0), (0, 0)), 'constant', constant_values=0)
    output_dict['observation'] = {}
    output_dict['observation']['image_primary'] = img[...]
            # output_dict['pixel_values'] = cv2.resize(img, (224, 224))
            # input_size = [182, 322]
    # trajs['action']['rotation_delta'][0, 0][0] += 1.0
    
    low = torch.tensor([-0.5557471346855163, -0.6689224243164062, -0.5768502354621887, -0.4126838445663452, -0.40479663014411926, -1.2655779123306274, 0.0])
    high = torch.tensor([0.5577198266983032, 0.6465585827827454, 0.43592289090156555, 0.4223960340023041, 0.45459190011024475, 1.330322265625, 1.0])
        
    output_dict['action'] = torch.cat([trajs['action']['action'], 
                ( 1 - trajs['action']['gripper_closedness_action']).to(torch.float32)], dim=-1)
        
    # output_dict['state'] = torch.cat([trajs['action']['state_pose'], 
    #                         ( 1 - trajs['action']['gripper_closedness_action']).to(torch.float32)], dim=-1)
    output_dict['action'] = torch.clip(2 * (output_dict['action'] - low) / (high - low + 1e-8) -1, min=-1, max=1.)
    
    # no mean_std norm
    # mean = torch.tensor([0.013664860278367996, 0.0007325399201363325, 0.016608335077762604, -0.002270081778988242, -0.00021358212688937783, 0.006111276336014271, 0.40524646639823914])
    # std = torch.tensor([0.19084422290325165, 0.2262507677078247, 0.20552673935890198, 0.15208850800991058, 0.16257022321224213, 0.3827716112136841, 0.49034905433654785])
    # #print(type(output_dict['action'][...,:-1]),type(output_dict['action']))
    # #print(output_dict['action'][...,:-1])
    # output_dict['action'][...,:-1] = (output_dict['action'][...,:-1] - mean[:-1]) / (std[:-1] + 1e-8) 
    
    output_dict['action'] = output_dict['action'].cpu().numpy()
    # if 'DEBUG' in os.environ:
    #     print(low, output_dict['action'][0][0])
    #print(output_dict['action'][...,-1])

    output_dict['task'] = {}
    output_dict['task']['language_instruction'] = trajs['instruction']
        
    # print(trajs['action']['loss_weight'].shape)
    output_dict['action_past_goal'] = torch.sum(trajs['action']['loss_weight'], dim=-1) == 0
    output_dict['ep_path'] = trajs['ep_path'] if 'ep_path' in trajs else None
    #print("output_dict",output_dict)
    # output_dict {'dataset_name': 'calvin', 
    # 'observation': {'image_primary': array( , dtype=uint8)}, 
    # 'action': array([[[ 0.01494431,  0.22614431, -0.20797485, -0.15288794,
    #       0.29573035,  0.25493443,  1.        ]]], dtype=float32), 
    # 'task': {'language_instruction': 'go slide the blue block to the left'}, 
    # 'action_past_goal': tensor([[False]]), 
    # 'ep_path': None}
    return batch_transform(output_dict)


def convert_data_structure_lab(trajs, batch_transform, dataset_name=''):
    output_dict = {}
            # import ipdb;ipdb.set_trace()
    output_dict['dataset_name'] = dataset_name
    import cv2
    img = trajs['observation']['image'].cpu().numpy() # 180 320 3
    # print(img.min(), img.max())
    # img = np.pad(img, ( (70, 70), (0, 0), (0, 0)), 'constant', constant_values=0)
    output_dict['observation'] = {}
    output_dict['observation']['image_primary'] = img[...]
            # output_dict['pixel_values'] = cv2.resize(img, (224, 224))
            # input_size = [182, 322]
    # trajs['action']['rotation_delta'][0, 0][0] += 1.0
    if dataset_name == 'lab_907':
        # low = torch.tensor([-0.07335383, -0.07777873, -0.07212001, -0.10891825, -0.23829974, -0.19956847, 0]).to(torch.float32)
        # high = torch.tensor([0.08313077, 0.09487492, 0.08827358, 0.11910569, 0.18393938, 0.16685072, 1]).to(torch.float32)

        low = torch.tensor([-0.009082376956939697, -0.02768026292324066, -0.09064042553305625, -0.088255375623703, -0.07572497427463531, -0.10641985386610031, 0]).to(torch.float32)
        high = torch.tensor([0.049961209297180176, 0.029934369027614594, 0.06721316277980804, 0.06538952142000198, 0.03357397019863129, 0.17205530777573924, 1]).to(torch.float32)

    elif dataset_name == 'lab_907_1':
        low = torch.tensor([-0.07060414552688599, -0.21747050866484638, -0.2911102771759033, -0.18562862336635585, -0.1285559296607971, -0.4114302545785903, 0.0]).to(torch.float32)
        high = torch.tensor([0.11125240057706841, 0.1061392036080361, 0.12897171080112457, 0.1357136829197407, 0.10151379711925987, 0.4232045072317128, 1.0]).to(torch.float32)


# min [-1.7095022201538086, -2.0249087810516357, -2.528684377670288, -2.6583571434020996, -2.314984083175659, -2.6776225566864014, 0.0]
# max [1.4095149040222168, 2.55558705329895, 1.734158992767334, 1.8176867961883545, 1.9905537366867065, 4.227196216583252, 1.0]
# low [-0.5557471346855163, -0.6689224243164062, -0.5768502354621887, -0.4126838445663452, -0.40479663014411926, -1.2655779123306274, 0.0]
# high [0.5577198266983032, 0.6465585827827454, 0.43592289090156555, 0.4223960340023041, 0.45459190011024475, 1.330322265625, 1.0]
# mean [0.013664860278367996, 0.0007325399201363325, 0.016608335077762604, -0.002270081778988242, -0.00021358212688937783, 0.006111276336014271, 0.40524646639823914]
# std [0.19084422290325165, 0.2262507677078247, 0.20552673935890198, 0.15208850800991058, 0.16257022321224213, 0.3827716112136841, 0.49034905433654785]

    else:
        low = torch.tensor([-0.07335383, -0.07777873, -0.07212001, -0.10891825, -0.23829974, -0.19956847, 0]).to(torch.float32)
        high = torch.tensor([0.08313077, 0.09487492, 0.08827358, 0.11910569, 0.18393938, 0.16685072, 1]).to(torch.float32)

    
    
    output_dict['action'] = torch.cat([trajs['action']['world_vector'], trajs['action']['rotation_delta'], 
                                ( 1 - trajs['action']['gripper_closedness_action']).to(torch.float32)], dim=-1)

    output_dict['state'] = torch.cat([trajs['action']['state_pose'], 
                            ( 1 - trajs['action']['gripper_closedness_action']).to(torch.float32)], dim=-1)

    #print(output_dict['action'][...,-1])
#    output_dict['action'][...,:-1] = torch.clip(2 * (output_dict['action'][...,:-1] - low[:-1]) / (high[:-1] - low[:-1] + 1e-8) -1, min=-1, max=1.).cpu().numpy()
    if dataset_name not in ['no_norm', 'no_norm1']:
        output_dict['action'][...,:-1] = 2 * (output_dict['action'][...,:-1] - low[:-1]) / (high[:-1] - low[:-1] + 1e-8) -1
    else:
        
        if dataset_name == 'no_norm1':
            mean = torch.tensor([0.009494006633758545, -0.009273790754377842, -0.023042574524879456, -0.007257933262735605, -0.0023885052651166916, 0.00760976318269968, 0.45395463705062866])
            std = torch.tensor([0.03161812573671341, 0.051145896315574646, 0.06707415729761124, 0.05436154454946518, 0.03937571868300438, 0.12943556904792786, 0.4978216290473938])
        else:
            mean = torch.tensor([0.013636833988130093, -0.0017261075554415584, -0.03187509998679161, -0.0023941127583384514, 0.003163236426189542, 0.017856508493423462, 0.3645598292350769])
            std = torch.tensor([0.02514560893177986, 0.027360064908862114, 0.04232143610715866, 0.04644988849759102, 0.027758773416280746, 0.09074866026639938, 0.48129966855049133])
        output_dict['action'][...,:-1] = (output_dict['action'][...,:-1] - mean[:-1]) / (std[:-1] + 1e-8) 
        # (x - metadata[key]["mean"]) / (metadata[key]["std"] + 1e-8)
        # print('no_norm')
    output_dict['action'] = output_dict['action'].cpu().numpy()
    # if 'DEBUG' in os.environ:
    #     print(low, output_dict['action'][0][0])
    #print(output_dict['action'][...,-1])

    output_dict['task'] = {}
    output_dict['task']['language_instruction'] = trajs['instruction']
    
    # print(trajs['action']['loss_weight'].shape)
    output_dict['action_past_goal'] = torch.sum(trajs['action']['loss_weight'], dim=-1) == 0
    output_dict['ep_path'] = trajs['ep_path'] if 'ep_path' in trajs else None
     
    return batch_transform(output_dict)



class CalvinDataset_warp(CalvinDataset_Policy):

# data_dir,
#                  seq_len=10,
#                  act_len=5, 
#                  forward_n_max=25, 
#                  mode='train',
#                  subfolder='task_ABC_D',
#                  use_data_augmentation=True,
#                  task_num=10000,
#                  use_play=True,
#                  use_labeled=True,
#                  wrap_grmg_data=0

    def __init__(self, dataname='calvin', 
                 data_dir='vc_new:s3://houzhi/',
                 seq_len=10,
                 act_len=5, 
                 forward_n_max=25, 
                 mode='train',
                 subfolder='task_ABC_D',
                 use_data_augmentation=True,
                 task_num=10000,
                 use_play=True,
                 use_labeled=True,
                 wrap_grmg_data=0,
                 batch_transform=None
                 ):
        super().__init__(
                data_dir,
                 seq_len=seq_len,
                 act_len=act_len, 
                 forward_n_max=forward_n_max, 
                 mode=mode,
                 subfolder=subfolder,
                 use_data_augmentation=use_data_augmentation,
                 task_num=task_num,
                 use_play=use_play,
                 use_labeled=use_labeled,
                 wrap_grmg_data=wrap_grmg_data,
                 unorm=True
        )
        if batch_transform==None:
            self.batch_transform = None
        else:
            self.batch_transform = batch_transform
        self.dataname = dataname
        assert dataname!='lab'
        # import ipdb;ipdb.set_trace()
    
    def set_batch_transform(self, batch_transform):
        self.batch_transform = batch_transform
    
    @torch.no_grad()
    def __getitem__(self, index):
        data = super().__getitem__(index)
        return convert_data_structure_calvin(data, self.batch_transform)
    
    
