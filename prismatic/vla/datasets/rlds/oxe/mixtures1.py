"""
mixtures.py

Defines a registry of dataset mixtures and weights for the Open-X Embodiment Datasets. Each dataset is associated with
a float "sampling weight"
"""

from typing import Dict, List, Tuple

# fmt: off
OXE_NAMED_MIXTURES: Dict[str, List[Tuple[str, float]]] = {
    # === Bridge V2 Dataset ===
    "bridge": [
        # ("bridge_oxe", 1.0),                                    # Version of Bridge V2 in Open-X GCP Bucket
        ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
    ],


    # === [Moderate-Scale] Bridge++ Mixtures ===
    "bridge_rt_1": [
        # ("bridge_oxe", 1.0)                                   # Version of Bridge V2 in Open-X GCP Bucket
        ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website

        ("fractal20220817_data", 1.0),                          # Google RT-1 Robot Data (Large-Scale)
    ],

    # === RT-X Mixtures ===
    "rtx": [
        ("fractal20220817_data", 0.54087122203),                # Google RT-1 Robot Data (Large-Scale)
        ("kuka", 0.8341046294),
        # ("bridge_oxe", 1.0)                                   # Version of Bridge V2 in Open-X GCP Bucket
        ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
        ("taco_play", 2.0),
        ("jaco_play", 2.0),
        ("berkeley_cable_routing", 3.0),
        ("roboturk", 1.0),
        # ("nyu_door_opening_surprising_effectiveness", 5.0),   # Note --> only contains wrist camera images (skip?)
        ("viola", 2.0),
        ("berkeley_autolab_ur5", 1.0),
        ("toto", 1.0),
    ],

    "rtx_franka": [
        ("fractal20220817_data", 0.54087122203),                # Google RT-1 Robot Data (Large-Scale)
        ("kuka", 0.8341046294),
        # ("bridge_oxe", 1.0)                                   # Version of Bridge V2 in Open-X GCP Bucket
        ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
        ("taco_play", 2.0),
        ("jaco_play", 2.0),
        ("berkeley_cable_routing", 3.0),
        ("roboturk", 1.0),
        # ("nyu_door_opening_surprising_effectiveness", 5.0),   # Note --> only contains wrist camera images (skip?)
        ("viola", 2.0),
        ("berkeley_autolab_ur5", 1.0),
        ("toto", 1.0),

        ("taco_play", 1.0),
        ("berkeley_cable_routing", 1.0),
        ("viola", 1.0),
        ("toto", 1.0),
        ("stanford_hydra_dataset_converted_externally_to_rlds", 1.0),
        ("austin_buds_dataset_converted_externally_to_rlds", 3.0),
        ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
        ("maniskill_dataset_converted_externally_to_rlds", 0.1),
        ("furniture_bench_dataset_converted_externally_to_rlds", 0.1),
        ("cmu_franka_exploration_dataset_converted_externally_to_rlds", 5.0),
        ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),
        ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
        ("berkeley_rpt_converted_externally_to_rlds", 1.0),
        ("kaist_nonprehensile_converted_externally_to_rlds", 3.0),
        ("stanford_robocook_converted_externally_to_rlds", 1.0),
        ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),
        ("utaustin_mutex", 1.0),
        ("cmu_play_fusion", 1.0),
    ],

    # === Open-X Magic Soup ===
    "oxe_magic_soup": [
        ("fractal20220817_data", 0.54087122203),                # Google RT-1 Robot Data (Large-Scale)
        ("kuka", 0.8341046294),
        # ("bridge_oxe", 1.0)                                   # Version of Bridge V2 in Open-X GCP Bucket
        ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
        ("taco_play", 2.0),
        ("jaco_play", 1.0),
        ("berkeley_cable_routing", 1.0),
        ("roboturk", 2.0),
        # ("nyu_door_opening_surprising_effectiveness", 1.0),   # Note --> only contains wrist camera images (skip?)
        ("viola", 2.0),
        ("berkeley_autolab_ur5", 2.0),
        ("toto", 1.0),
        ("language_table", 0.1),
        ("stanford_hydra_dataset_converted_externally_to_rlds", 2.0),
        ("austin_buds_dataset_converted_externally_to_rlds", 1.0),
        ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
        ("furniture_bench_dataset_converted_externally_to_rlds", 0.1),
        ("ucsd_kitchen_dataset_converted_externally_to_rlds", 2.0),
        ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),
        ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
        # ("bc_z", 0.2),                                        # Note --> raw data is broken!
        ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),
        ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),
        # ("uiuc_d3field", 1.0),                                # Note --> raw data is broken!
        ("utaustin_mutex", 1.0),
        ("berkeley_fanuc_manipulation", 2.0),
        ("cmu_stretch", 1.0),
    ],

    # === Open-X Magic Soup++ ===
    "oxe_magic_soup_plus": [
        ("fractal20220817_data", 0.54087122203),                # Google RT-1 Robot Data (Large-Scale)
        ("kuka", 0.8341046294),
        ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
        ("taco_play", 2.0),
        ("jaco_play", 1.0),
        ("berkeley_cable_routing", 1.0),
        ("roboturk", 2.0),
        ("viola", 2.0),
        ("berkeley_autolab_ur5", 2.0),
        ("toto", 1.0),
        ("language_table", 0.1),
        ("stanford_hydra_dataset_converted_externally_to_rlds", 2.0),
        ("austin_buds_dataset_converted_externally_to_rlds", 1.0),
        ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
        ("furniture_bench_dataset_converted_externally_to_rlds", 0.1),
        ("ucsd_kitchen_dataset_converted_externally_to_rlds", 2.0),
        ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),
        ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
        ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),
        ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),
        ("utaustin_mutex", 1.0),
        ("berkeley_fanuc_manipulation", 2.0),
        ("cmu_stretch", 1.0),
        ## New Datasets in MagicSoup++
        ("bc_z", 0.2),                                          # Note: use v0.1.0 --> later versions broken
        ("fmb_dataset", 1.0),
        ("dobbe", 0.2),
        ("droid", 0.06),
    ],

    "oxe_magic_soup_plus_minus": [
        ("fractal20220817_data", 1.0),                          # Google RT-1 Robot Data (Large-Scale)
        ("kuka", 0.8341046294),
        # ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
        ("taco_play", 2.0),
        ("jaco_play", 1.0),
        ("berkeley_cable_routing", 1.0),
        ("roboturk", 2.0),
        ("viola", 2.0),
        ("berkeley_autolab_ur5", 2.0),
        ("toto", 1.0),
        ("language_table", 0.1),
        # ("stanford_hydra_dataset_converted_externally_to_rlds", 2.0),
        ("austin_buds_dataset_converted_externally_to_rlds", 1.0),
        ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
        # ("furniture_bench_dataset_converted_externally_to_rlds", 0.1),
        ("ucsd_kitchen_dataset_converted_externally_to_rlds", 2.0),
        # ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),
        ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
        ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),
        # ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),
        # ("utaustin_mutex", 1.0),
        # ("berkeley_fanuc_manipulation", 2.0),
        ("cmu_stretch", 1.0),
        ## New Datasets in MagicSoup++
        # ("bc_z", 0.2),                                          # Note: use v0.1.0 --> later versions broken
        # ("fmb_dataset", 1.0),
        # ("dobbe", 0.2),
        # ("droid", 0.06),
    ],
    # "up_to_date_mixture": [
    #     ("fractal20220817_data", 1.0),              # 256 320            # Google RT-1 Robot Data (Large-Scale)
    #     # ("kuka", 0.8341046294),   # 512 640
    #     # ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
    #     ("taco_play", 2.0),
    #     ("jaco_play", 1.0),
    #     ("berkeley_cable_routing", 1.0),
    #     # ("roboturk", 2.0),  # 2144,
    #     ("viola", 2.0),
    #     # ("berkeley_autolab_ur5", 2.0),
    #     # ("toto", 1.0),  # 901
    #     ("language_table", 0.1),
    #     ("stanford_hydra_dataset_converted_externally_to_rlds", 2.0),
    #     ("austin_buds_dataset_converted_externally_to_rlds", 1.0),
    #     ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
    #     ("furniture_bench_dataset_converted_externally_to_rlds", 0.1),
    #     # ("ucsd_kitchen_dataset_converted_externally_to_rlds", 2.0),  # 150
    #     #("austin_sailor_dataset_converted_externally_to_rlds", 1.0),
    #     ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
    #     ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),
    #     #  ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),   # 520
    #      ("utaustin_mutex", 1.0),
    #      ("berkeley_fanuc_manipulation", 2.0),
    #     ("cmu_stretch", 1.0),
    #     ## New Datasets in MagicSoup++
    #     #("bc_z", 0.2),                                          # Note: use v0.1.0 --> later versions broken
    #     #("fmb_dataset", 1.0),
    #     ('fmb', 1.0),
    #     ("dobbe", 0.2),
    #     #("droid", 0.06),
    #     ('robo_set', 1.),
    # ],
    #if name in ['viola', 'dobbe', 'robo_set', 'cmu_stretch', 'taco_play', 'jaco_play', 'berkeley_autolab_ur5', 'fractal20220817_data',  'berkeley_cable_routing', ]:
    "resized" : [
       ("fractal20220817_data", 1.0),
        ("dobbe", 0.2),
        ("droid", 0.06),
        ('robo_set', 1.),
         ("viola", 2.0),
         ("kuka", 0.8341046294),
         ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
         ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
         ("furniture_bench_dataset_converted_externally_to_rlds", 1.0),
         ("jaco_play", 1.0),
         ("language_table", 0.1),
          ("toto", 1.0),  # 901
        #  ("berkeley_autolab_ur5", 2.0),
        #  ("berkeley_cable_routing", 1.0),
        # ("cmu_stretch", 1.0),
         
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
    ],
                    #  PRE austin_buds_dataset_converted_externally_to_rlds/
                    #        PRE austin_sirius_dataset_converted_externally_to_rlds/
                    #        PRE berkeley_autolab_ur5/
                    #        PRE berkeley_cable_routing/
                    #        PRE berkeley_fanuc_manipulation/
                    #        PRE bridge_orig/
                    #        PRE cmu_stretch/
                    #        PRE dlr_edan_shared_control_converted_externally_to_rlds/
                    #        PRE dobbe/
                    #        PRE fractal20220817_data/
                    #        PRE furniture_bench_dataset_converted_externally_to_rlds/
                    #        PRE iamlab_cmu_pickup_insert_converted_externally_to_rlds/
                    #        PRE jaco_play/
                    #        PRE nyu_franka_play_dataset_converted_externally_to_rlds/
                    #        PRE robo_set/
                    #        PRE roboturk/
                    #        PRE stanford_hydra_dataset_converted_externally_to_rlds/
                    #        PRE taco_play/
                    #        PRE toto/
                    #        PRE ucsd_kitchen_dataset_converted_externally_to_rlds/
                    #        PRE utaustin_mutex/
                    #        PRE viola/
    "resized3" : [
       ("fractal20220817_data", 1.0),
        ("dobbe", 0.2),
        ("droid", 0.06),
        ('robo_set', 1.),
         ("viola", 2.0),
         ("kuka", 0.8341046294),
         ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
         ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
         ("furniture_bench_dataset_converted_externally_to_rlds", 1.0),
         ("jaco_play", 1.0),
         ("language_table", 0.1),
          ("toto", 1.0),  # 901
    ],
    "resized6" : [
       ("fractal20220817_data", 1.0),
        ("dobbe", 0.2),
        ("droid", 0.06),
        # ('robo_set', 2.), 
         ("viola", 2.0),
         ("kuka", 0.8341046294),     # 10
         ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
         ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
         ("furniture_bench_dataset_converted_externally_to_rlds", 1.0),
         ("jaco_play", 1.0),
         ("language_table", 0.1),
          ("toto", 1.0),  # 901
    ],  
    "similar_images" : [
        # ("dobbe", 0.2),
        ("droid", 0.15),
        ('robo_set', 0.8),
        #  ("viola", 2.0),
        #  ("kuka", 0.8341046294),
         ("roboturk", 2.0),  # 2144,
         ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
         ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0), # TODO
         ("stanford_hydra_dataset_converted_externally_to_rlds", 2.0),
         ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),
         ("berkeley_fanuc_manipulation", 2.0),
        #  ("furniture_bench_dataset_converted_externally_to_rlds", 1.0),
        #  ("jaco_play", 1.0),
         ("language_table", 0.1),
        #   ("toto", 1.0),  # 901
        ("berkeley_autolab_ur5", 2.0),
    ],     
    "state_images" : [
        # ("dobbe", 0.2),
        ("droid", 0.15),
        ('robo_set', 0.8),
        #  ("viola", 2.0),
        #  ("kuka", 0.8341046294), # no state
        #  ("roboturk", 2.0),  # 2144,
         ("bridge_state", 1.0),                                   # Original Version of Bridge V2 from Project Website
         ("nyu_franka_play_dataset_converted_externally_to_rlds_state", 3.0), # TODO
         ("stanford_hydra_dataset_converted_externally_to_rlds_state", 2.0),
         ("dlr_edan_shared_control_converted_externally_to_rlds_state", 1.0),
         ("berkeley_fanuc_manipulation_state", 2.0),
        #  ("furniture_bench_dataset_converted_externally_to_rlds", 1.0),
        #  ("jaco_play", 1.0),
        #  ("language_table", 0.1),
        #   ("toto", 1.0),  # 901
        # ("berkeley_autolab_ur5", 2.0),
    ],   
    "proprio": [
 # ("dobbe", 0.2),
        # ("droid_state", 0.15),
        ('robo_set_state', 0.8),
        # #  ("viola", 2.0),
         ("kuka_state", 0.8341046294), # no state
        # #  ("roboturk", 2.0),  # 2144,
         ("bridge_orig_state", 1.0),                                   # Original Version of Bridge V2 from Project Website
         ("nyu_franka_play_dataset_converted_externally_to_rlds_state", 3.0), # TODO
         ("stanford_hydra_dataset_converted_externally_to_rlds_state", 2.0),
         ("dlr_edan_shared_control_converted_externally_to_rlds_state", 1.0),
         ("berkeley_fanuc_manipulation_state", 2.0),
        #  ("furniture_bench_dataset_converted_externally_to_rlds", 1.0),
        #  ("jaco_play", 1.0),
        #  ("language_table", 0.1),
        #   ("toto", 1.0),  # 901
        # ("berkeley_autolab_ur5", 2.0),
    ],
    "resized7" : [
       ("fractal20220817_data", 0.5), # no wrist
        ("dobbe", 0.2),  # wrist
        ("droid", 0.06), # wrist
        ('robo_set', 1.), # wrist
         ("viola", 2.0),  # wrist
         ("kuka", 0.8341046294), # no wrist
         ("bridge_orig", 1.2),  # wrist                                  # Original Version of Bridge V2 from Project Website
         ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0), # no wrist
         ("furniture_bench_dataset_converted_externally_to_rlds", 0.2), # wrist
         ("stanford_hydra_dataset_converted_externally_to_rlds", 2.0),  #  wrist
         ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0), # no wrist
         ("berkeley_fanuc_manipulation", 2.0), # wrist
         ("jaco_play", 1.0), # wrist
         ("language_table", 0.1), # no wrist
          ("toto", 1.0),  # 901 # no wrist
    ],   
    "resized7_with_wrist" : [
        ("dobbe", 0.2),  # wrist
        ("droid", 0.06), # wrist
        ('robo_set', 1.), # wrist
         ("viola", 2.0),  # wrist
         ("bridge_orig", 1.2),  # wrist                                  # Original Version of Bridge V2 from Project Website
         ("furniture_bench_dataset_converted_externally_to_rlds", 0.2), # wrist
         ("stanford_hydra_dataset_converted_externally_to_rlds", 2.0),  #  wrist
         ("berkeley_fanuc_manipulation", 2.0), # wrist
         ("jaco_play", 1.0), # wrist
    ],   
    "resized7_nodroid" : [
       ("fractal20220817_data", 0.5),
        ("dobbe", 0.2),
        # ("droid", 0.06),
        ('robo_set', 1.),
         ("viola", 2.0),
         ("kuka", 0.8341046294),
         ("bridge_orig", 1.2),                                   # Original Version of Bridge V2 from Project Website
         ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
         ("furniture_bench_dataset_converted_externally_to_rlds", 0.2),
         ("stanford_hydra_dataset_converted_externally_to_rlds", 2.0),
         ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),
         ("berkeley_fanuc_manipulation", 2.0),
         ("jaco_play", 1.0),
         ("language_table", 0.1),
          ("toto", 1.0),  # 901
    ],       
    "resized5" : [
       ("fractal20220817_data", 1.0),
        ("dobbe", 0.2),
        ("droid", 0.6),
        ('robo_set', 2.),
         ("viola", 2.0),
         ("kuka", 0.8341046294),
         ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
         ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
         ("furniture_bench_dataset_converted_externally_to_rlds", 1.0),
         ("jaco_play", 1.0),
         ("language_table", 0.1),
          ("toto", 1.0),  # 901
    ],
    "resized4" : [
       ("fractal20220817_data", 1.0),
        ("dobbe", 0.2),
        # ("droid", 0.5),
        
        # ('robo_set', 2.),
         ("viola", 2.0),
         ("kuka", 0.8341046294),
         ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
         ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
         ("furniture_bench_dataset_converted_externally_to_rlds", 1.0),
         ("jaco_play", 1.0),
         ("language_table", 0.1),
          ("toto", 1.0),  # 901
          ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),
          ("berkeley_fanuc_manipulation", 2.0),
          ("berkeley_autolab_ur5", 2.0),
    ],

    "resized2" : [
       ("fractal20220817_data", 1.0),
    ],
    "resized1" : [
       ("fractal20220817_data", 1.0),
        ("dobbe", 0.2),
        ("droid", 0.06),
        ('robo_set', 1.),
         ("viola", 2.0),
        #  ("berkeley_autolab_ur5", 2.0),
        #  ("berkeley_cable_routing", 1.0),
        ("cmu_stretch", 1.0),
        #  ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
         ("austin_buds_dataset_converted_externally_to_rlds", 1.0),
         ("austin_sirius_dataset_converted_externally_to_rlds", 1.0), # requires to fix
         ("berkeley_fanuc_manipulation", 2.0),
         ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),
         ("furniture_bench_dataset_converted_externally_to_rlds", 1.0),
        #  ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0), # requires to fix
         ("jaco_play", 1.0),
         ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
        #  ("roboturk", 2.0),  # 2144,
        #  ("stanford_hydra_dataset_converted_externally_to_rlds", 2.0),
        #  ("taco_play", 2.0),
        #  ("toto", 1.0),  # 901
        #  ("ucsd_kitchen_dataset_converted_externally_to_rlds", 2.0),  # 150  disable
        #  ("utaustin_mutex", 1.0),
    ],




    "debug" : [
        ("austin_buds_dataset_converted_externally_to_rlds", 1.0),
         ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
         ("berkeley_fanuc_manipulation", 2.0),
         ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),
#        ("fractal20220817_data", 1.0),
#         ("dobbe", 0.2),
#         ('robo_set', 1.),
#          ("viola", 2.0),
#          ("berkeley_autolab_ur5", 2.0),
#          ("berkeley_cable_routing", 1.0),
#         ("cmu_stretch", 1.0),
#          ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
#         ("austin_buds_dataset_converted_externally_to_rlds", 1.0),
#         ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
#         ("berkeley_fanuc_manipulation", 2.0),
#         ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),
#         ("furniture_bench_dataset_converted_externally_to_rlds", 1.0),
#         ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0), # ok
#  #        ("jaco_play", 1.0), # ok
#          ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
#          ("roboturk", 2.0),  # 2144,
#          ("stanford_hydra_dataset_converted_externally_to_rlds", 2.0),
#          ("taco_play", 2.0),
#          ("toto", 1.0),  # 901
#          ("ucsd_kitchen_dataset_converted_externally_to_rlds", 2.0),  # 150
#          ("utaustin_mutex", 1.0),
#         # ("vima", 1.0),
            
    ],
    "up_to_date_mixture": [
        ("fractal20220817_data", 1.0),              # 256 320            # Google RT-1 Robot Data (Large-Scale)
        ("kuka", 0.8341046294),   # 512 640
        ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
         ("taco_play", 2.0),
         ("jaco_play", 1.0),
         ("berkeley_cable_routing", 1.0),
        ("roboturk", 2.0),  # 2144,
         ("viola", 2.0),
         ("berkeley_autolab_ur5", 2.0),
        ("toto", 1.0),  # 901
        ("language_table", 0.1),
        ("stanford_hydra_dataset_converted_externally_to_rlds", 2.0),
        ("austin_buds_dataset_converted_externally_to_rlds", 1.0),
        ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
        ("furniture_bench_dataset_converted_externally_to_rlds", 0.1),
        ("ucsd_kitchen_dataset_converted_externally_to_rlds", 2.0),  # 150
        # ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),
        ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),    # requies to 
        ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),
         ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),   # 520
         ("utaustin_mutex", 1.0),
         ("berkeley_fanuc_manipulation", 2.0),
        ("cmu_stretch", 1.0),
        ## New Datasets in MagicSoup++
        #("bc_z", 0.2),                                          # Note: use v0.1.0 --> later versions broken
        #("fmb_dataset", 1.0),
        # ('fmb', 1.0),
        ("dobbe", 0.2),
        # ("droid", 0.06),
        # ('robo_set', 1.),
        # sh pjlab_run_c.sh 1 1 embodied INTERN3 reserved python3 vla-scripts/hook_dataset.py --data_name berkeley_cable_routing;\
        # sh pjlab_run_c.sh 1 1 embodied INTERN3 reserved python3 vla-scripts/hook_dataset.py --data_name roboturk;\
        # sh pjlab_run_c.sh 1 1 embodied INTERN3 reserved python3 vla-scripts/hook_dataset.py --data_name viola; \
        # sh pjlab_run_c.sh 1 1 embodied INTERN3 reserved python3 vla-scripts/hook_dataset.py --data_name berkeley_autolab_ur5;\
        # sh pjlab_run_c.sh 1 1 embodied INTERN3 reserved python3 vla-scripts/hook_dataset.py --data_name toto;\
        # sh pjlab_run_c.sh 1 1 embodied INTERN3 reserved python3 vla-scripts/hook_dataset.py --data_name language_table;\
        # sh pjlab_run_c.sh 1 1 embodied INTERN3 reserved python3 vla-scripts/hook_dataset.py --data_name stanford_hydra_dataset_converted_externally_to_rlds;\
        # sh pjlab_run_c.sh 1 1 embodied INTERN3 reserved python3 vla-scripts/hook_dataset.py --data_name austin_buds_dataset_converted_externally_to_rlds;\
        # sh pjlab_run_c.sh 1 1 embodied INTERN3 reserved python3 vla-scripts/hook_dataset.py --data_name nyu_franka_play_dataset_converted_externally_to_rlds;\
        # sh pjlab_run_c.sh 1 1 embodied INTERN3 reserved python3 vla-scripts/hook_dataset.py --data_name furniture_bench_dataset_converted_externally_to_rlds;\
        # sh pjlab_run_c.sh 1 1 embodied INTERN3 reserved python3 vla-scripts/hook_dataset.py --data_name ucsd_kitchen_dataset_converted_externally_to_rlds;\
        # sh pjlab_run_c.sh 1 1 embodied INTERN3 reserved python3 vla-scripts/hook_dataset.py --data_name austin_sirius_dataset_converted_externally_to_rlds;\
        # sh pjlab_run_c.sh 1 1 embodied INTERN3 reserved python3 vla-scripts/hook_dataset.py --data_name dlr_edan_shared_control_converted_externally_to_rlds;\
        # sh pjlab_run_c.sh 1 1 embodied INTERN3 reserved python3 vla-scripts/hook_dataset.py --data_name iamlab_cmu_pickup_insert_converted_externally_to_rlds;\
        # sh pjlab_run_c.sh 1 1 embodied INTERN3 reserved python3 vla-scripts/hook_dataset.py --data_name utaustin_mutex;\
        # sh pjlab_run_c.sh 1 1 embodied INTERN3 reserved python3 vla-scripts/hook_dataset.py --data_name berkeley_fanuc_manipulation;\
        # sh pjlab_run_c.sh 1 1 embodied INTERN3 reserved python3 vla-scripts/hook_dataset.py --data_name cmu_stretch;
    ],

    # === T-DROID Dataset ===
    "tdroid_carrot_in_bowl": [
        ("tdroid_carrot_in_bowl", 1.0),
    ],
    "tdroid_pour_corn_in_pot": [
        ("tdroid_pour_corn_in_pot", 1.0),
    ],
    "tdroid_flip_pot_upright": [
        ("tdroid_flip_pot_upright", 1.0),
    ],
    "tdroid_move_object_onto_plate": [
        ("tdroid_move_object_onto_plate", 1.0),
    ],
    "tdroid_knock_object_over": [
        ("tdroid_knock_object_over", 1.0),
    ],
    "tdroid_cover_object_with_towel": [
        ("tdroid_cover_object_with_towel", 1.0),
    ],

    # === DROID Finetuning Datasets ===
    "droid_wipe": [
        ("droid_wipe", 1.0),
    ],
}
# fmt: on
