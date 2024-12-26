import os
import time
os.environ['LD_PRELOAD']='/mnt/petrelfs/houzhi/lib/libtcmalloc_minimal.so.4.5.8'
import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union
import sys
current_path = os.getcwd()
sys.path.append(os.path.join(current_path, "../"))
sys.path.append(os.path.join(current_path))
import dlimp as dl
import psutil
# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from PIL import Image
import tensorflow as tf

os.environ["AWS_ACCESS_KEY_ID"] = "H9HBOW256ACSZ0G62JGG"
os.environ["AWS_SECRET_ACCESS_KEY"] = "o3fiSkvVaNRsDiLMhqA1unUNYKzWfxnyGTErZLrW"
# os.environ["AWS_REGION"] = "us-east-1"
os.environ["S3_ENDPOINT"] = "http://p-ceph-norm-inside.pjlab.org.cn"
os.environ["S3_USE_HTTPS"] = "0"
os.environ["S3_VERIFY_SSL"] = "0"

import tensorflow_io as tfio
import tensorflow_datasets as tfds

data_dir = 's3://openx'
dataset_list = []
import sys
name_list = ['droid', 'bridge_orig', 'cmu_stretch', 'taco_play', 'jaco_play', 'berkeley_autolab_ur5', 'fractal20220817_data',  'berkeley_cable_routing', 
                'austin_buds_dataset_converted_externally_to_rlds', 'austin_sirius_dataset_converted_externally_to_rlds', 'berkeley_fanuc_manipulation',
                'dlr_edan_shared_control_converted_externally_to_rlds', 'furniture_bench_dataset_converted_externally_to_rlds', 'iamlab_cmu_pickup_insert_converted_externally_to_rlds',
                'jaco_play', 'nyu_franka_play_dataset_converted_externally_to_rlds', 'roboturk', 'stanford_hydra_dataset_converted_externally_to_rlds', 
                'taco_play', 'toto', 'ucsd_kitchen_dataset_converted_externally_to_rlds', 'utaustin_mutex', 'kuka', 'language_table', 'robo_set', 'dobbe']

if len(sys.argv) > 1:
    name_list = [sys.argv[1]]

for name in name_list:

# 'droid', 'robo_set', 
# for name in [d_name]:
    #if False:
    #  ("austin_buds_dataset_converted_externally_to_rlds", 1.0),
        #  ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
        #  ("berkeley_fanuc_manipulation", 2.0),
        #  ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),
        #  ("furniture_bench_dataset_converted_externally_to_rlds", 1.0),
        #  ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),
        #  ("jaco_play", 1.0),
        #  ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
        #  ("roboturk", 2.0),  # 2144,
        #  ("stanford_hydra_dataset_converted_externally_to_rlds", 2.0),
        #  ("taco_play", 2.0),
        #  ("toto", 1.0),  # 901
        #  ("ucsd_kitchen_dataset_converted_externally_to_rlds", 2.0),  # 150
        #  ("utaustin_mutex", 1.0),
        #  ("viola", 2.0),

        if name in ['droid', 'bridge_orig', 'vima_converted_externally_to_rlds']:
            builder = tfds.builder(name, data_dir=data_dir)
        else:
            # builder = tfds.builder(name, data_dir=data_dir.replace('s3://openx', 's3://openx/resized/'))
            builder = tfds.builder(name, data_dir=data_dir)
        
        
        dataset = dl.DLataset.from_rlds(builder, split="all", shuffle=True, num_parallel_reads=1)
        i =0 
        length_ = len(dataset)
        print('length', name, length_)
        # continue
        # import ipdb;ipdb.set_trace()
        # def rotate_if_scene(trajectory):
        #     import tensorflow as tf
        #     is_scene = tf.strings.regex_full_match(trajectory['traj_metadata']['episode_metadata']['file_path'][0], ".*scene.*")
        #     # import ipdb;ipdb.set_trace()
        #     trajectory['observation']['image_right'] = tf.image.decode_jpeg(trajectory['observation']['image_right'])
        #     trajectory['observation']['image_right'] = tf.cond(is_scene, lambda: trajectory['observation']['image_right'], lambda: tf.image.rot90(trajectory['observation']['image_right'], k=1))
        #     return trajectory
        # dataset = dataset.map(rotate_if_scene)


        empty_img = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x01,\x01,\x00\x00\xff\xdb\x00C\x00\x02\x01\x01\x01\x01\x01\x02\x01\x01\x01\x02\x02\x02\x02\x02\x04\x03\x02\x02\x02\x02\x05\x04\x04\x03\x04\x06\x05\x06\x06\x06\x05\x06\x06\x06\x07\t\x08\x06\x07\t\x07\x06\x06\x08\x0b\x08\t\n\n\n\n\n\x06\x08\x0b\x0c\x0b\n\x0c\t\n\n\n\xff\xdb\x00C\x01\x02\x02\x02\x02\x02\x02\x05\x03\x03\x05\n\x07\x06\x07\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\xff\xc0\x00\x11\x08\x01\x00\x01\x00\x03\x01"\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n\x16\x17\x18\x19\x1a%&\'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xc4\x00\x1f\x01\x00\x03\x01\x01\x01\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x11\x00\x02\x01\x02\x04\x04\x03\x04\x07\x05\x04\x04\x00\x01\x02w\x00\x01\x02\x03\x11\x04\x05!1\x06\x12AQ\x07aq\x13"2\x81\x08\x14B\x91\xa1\xb1\xc1\t#3R\xf0\x15br\xd1\n\x16$4\xe1%\xf1\x17\x18\x19\x1a&\'()*56789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00?\x00\xfe\x7f\xe8\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00(\xa2\x8a\x00\xff\xd9'        
        
        c = 0
        for item in dataset.as_numpy_iterator():
            
            i += 1
            import ipdb;ipdb.set_trace()
#            if i % 10 == 0:
#                for j in range(0, 20, 2):
#                    Image.fromarray(tf.image.decode_jpeg(item['observation']['image'][j], channels=3).numpy()).save('./temp_io/imagei_{}_{}_{}_{}.png'.format(name, i, j, item['language_instruction'][0].decode(), ))
#                    Image.fromarray(tf.image.decode_jpeg(item['observation']['image_right_side'][j], channels=3).numpy()).save('./temp_io/imageright_{}_{}_{}_{}.png'.format(name, i, j, item['language_instruction'][0].decode(), ))
            
#'language_instruction_3', 'language_instruction', 'is_terminal', 'action', 'language_instruction_2',
            if str(item['language_instruction'][0].decode()).__contains__('green block'):
                c+=1
                print(i, c, item['language_instruction'][0].decode())
                Image.fromarray(tf.image.decode_jpeg(item['observation']['exterior_image_1_left'][0], channels=3).numpy()).save('./temp_io/image_droid_{}_{}_{}_{}.png'.format(name, i, 0, item['language_instruction'][0].decode(), ))
            # if len(item['language_instruction'][0]) > 0 or len(item['language_instruction_3'][0]) > 0 or len(item['language_instruction_2'][0]) > 0:
            #     c += 1
            #     print(i, c, flush=True)
            
            continue
            if i % 50 == 0:
                try:
                    import numpy as np
                    
                    
                    # Image.fromarray(item['observation']['image_right'][0]).save('img_{}_{}_{}.png'.format(i, item['language_instruction'][0], item['observation']['segmentation_obj_info']['obj_name'][0][0]))
                    # Image.fromarray(item['observation']['image_right'][0]).save('img_{}_{}_{}.png'.format(i, item['language_instruction'][0], item['observation']['segmentation_obj_info']['obj_name'][0][0]))
                    # Image.fromarray(item['observation']['image_left'][0]).save('img_front_{}_{}_{}.png'.format(i, item['multimodal_instruction'][0][:20], item['observation']['segmentation_obj_info']['obj_name'][0][0]))
                    # Image.fromarray(item['observation']['state'][0]).save('state_{}_{}_{}.png'.format(i, item['multimodal_instruction'][[0]:20], item['observation']['segmentation_obj_info']['obj_name'][0][0]))
                    import tensorflow as tf
                    
                    if name in ['stanford_hydra_dataset_converted_externally_to_rlds']:
                        Image.fromarray(tf.image.decode_jpeg(item['observation']['image'][0], channels=3).numpy()).save('./temp/image{}_{}_{}.png'.format(name, i, item['language_instruction'][0].decode(), ))
                        pass
                    if name in ['nyu_franka_play_dataset_converted_externally_to_rlds']:
                        Image.fromarray(tf.image.decode_jpeg(item['observation']['image'][0], channels=3).numpy()).save('./temp/image{}_{}_{}.png'.format(name, i, item['language_instruction'][0].decode(), ))
                        Image.fromarray(tf.image.decode_jpeg(item['observation']['image_additional_view'][0], channels=3).numpy()).save('./temp/image_additional_view{}_{}_{}.png'.format(name, i, item['language_instruction'][0].decode(), ))
                        pass
                    if name in ['bridge_orig']:
                        for j in range(10):
                            Image.fromarray(tf.image.decode_jpeg(item['observation']['image_0'][j], channels=3).numpy()).save('./temp/img0_{}_{}_{}_{}.png'.format(name, i, item['language_instruction'][0].decode(), j))
                            Image.fromarray(tf.image.decode_jpeg(item['observation']['image_1'][j], channels=3).numpy()).save('./temp/img2_{}_{}_{}_{}.png'.format(name, i, item['language_instruction'][0].decode(), j))
                            Image.fromarray(tf.image.decode_jpeg(item['observation']['image_2'][j], channels=3).numpy()).save('./temp/img2_{}_{}_{}_{}.png'.format(name, i, item['language_instruction'][0].decode(), j))
                            Image.fromarray(tf.image.decode_jpeg(item['observation']['image_3'][j], channels=3).numpy()).save('./temp/img3_{}_{}_{}_{}.png'.format(name, i, item['language_instruction'][0].decode(), j))
                    if name in ['robo_set']:
                        # print(item['traj_metadata']['episode_metadata']['file_path'][0].decode().split('/'))
                        if not item['traj_metadata']['episode_metadata']['file_path'][0].decode().split('/')[-2].__contains__('scene') \
                            and not item['traj_metadata']['episode_metadata']['file_path'][0].decode().split('/')[-1].__contains__('scene'):
                            tag = item['traj_metadata']['episode_metadata']['file_path'][0].decode().split('/')[-2] + item['traj_metadata']['episode_metadata']['file_path'][0].decode().split('/')[-1]
                            Image.fromarray(tf.image.decode_jpeg(item['observation']['image_right'][0], channels=3).numpy()).save('./temp/img_{}_{}_{}_{}.png'.format(i, item['language_instruction'][0].decode(), item['traj_metadata']['episode_metadata']['trial_id'][0].decode(),tag))
                            Image.fromarray(tf.image.decode_jpeg(item['observation']['image_left'][0], channels=3).numpy()).save('./temp/imgleft_{}_{}_{}_{}.png'.format(i, item['language_instruction'][0].decode(), item['traj_metadata']['episode_metadata']['trial_id'][0].decode(),tag))
                            Image.fromarray(tf.image.decode_jpeg(item['observation']['image_top'][0], channels=3).numpy()).save('./temp/imgtop_{}_{}_{}_{}.png'.format(i, item['language_instruction'][0].decode(), item['traj_metadata']['episode_metadata']['trial_id'][0].decode(),tag))
                    
                    # import ipdb;ipdb.set_trace()    
                except Exception as e:
                    import ipdb;ipdb.set_trace()
                    import traceback
                    traceback.print_exc()
                    pass


        break
#             item['action']
# {'pose1_position': array([[ 0.4661258 , -0.07229367,  0.0479126 ]], dtype=float32), 'pose1_rotation': array([[ 0.        ,  0.        , -0.25881898,  0.9659258 ]],
#       dtype=float32), 'pose0_position': array([[ 0.46875  , -0.05625  ,  0.0479126]], dtype=float32), 'pose0_rotation': array([[0., 0., 0., 1.]], dtype=float32)}

        #     if i % 100 == 0:
        #         used_mem = psutil.virtual_memory().used
        #         print(i, length_, name, used_mem / 1024 / 1024, flush=True)


        dataset_list.append(dataset)

# 'robo_set', 1.),
        #  ("viola", 2.0),
        #  ("kuka", 0.8341046294),
        #  ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website

dataset: dl.DLataset = dl.DLataset.sample_from_datasets(dataset_list)
length_ = len(dataset)       

i = 0
for item in dataset.as_numpy_iterator():
    i += 1
    if i % 100 == 0:
        used_mem = psutil.virtual_memory().used
        print(i, length_, name, used_mem / 1024 / 1024, flush=True)

# length kuka 209880
# length droid 92233
# length bridge_orig 60064
# length fractal20220817_data 87212
# length cmu_stretch 135
# length taco_play 3603
# length jaco_play 1085
# length berkeley_autolab_ur5 1000
# length berkeley_cable_routing 1647
# length austin_buds_dataset_converted_externally_to_rlds 50
# length austin_sirius_dataset_converted_externally_to_rlds 559
# length berkeley_fanuc_manipulation 415
# length dlr_edan_shared_control_converted_externally_to_rlds 104
# length furniture_bench_dataset_converted_externally_to_rlds 5100
# length iamlab_cmu_pickup_insert_converted_externally_to_rlds 631
# length jaco_play 1085
# length nyu_franka_play_dataset_converted_externally_to_rlds 456
# length roboturk 1995
# length stanford_hydra_dataset_converted_externally_to_rlds 570
# length taco_play 3603
# length toto 1003
# length ucsd_kitchen_dataset_converted_externally_to_rlds 150
# length utaustin_mutex 1500
