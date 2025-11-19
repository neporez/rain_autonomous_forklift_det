from pathlib import Path
from easydict import EasyDict as dict

DATASET = dict(
    NAME='WarehouseDataset',
    TYPE=Path(__file__).stem,
    DATA_PATH='data/warehouse',
    CLASS_NAMES=['Pedestrian'],
    POINT_CLOUD_RANGE=[-75.2, -75.2, -5.0, 75.2, 75.2, 3.0],

    DATA_SPLIT=dict(train='train',
                    test='val'),

    INFO_PATH=dict(train=['warehouse_infos_train.pkl'], val=['warehouse_infos_val.pkl'],
                   test=['warehouse_infos_test.pkl']),


    DATA_AUGMENTOR=dict(
        DISABLE_AUG_LIST=['placeholder'],
        AUG_CONFIG_LIST=[
            dict(NAME='gt_sampling',
                 DB_INFO_PATH=['warehouse_dbinfos_train.pkl'],
                 PREPARE=dict(
                     filter_by_min_points=['Pedestrian:20'],
                 ),
                 SAMPLE_GROUPS=['Pedestrian:0'],
                 NUM_POINT_FEATURES=4,
                 REMOVE_EXTRA_WIDTH=[0.0, 0.0, 0.0],
                 LIMIT_WHOLE_SCENE=True),
            dict(NAME='random_world_flip',
                 ALONG_AXIS_LIST=['x', 'y']),
            dict(NAME='random_world_rotation',
                 WORLD_ROT_ANGLE=[-0.78539816, 0.78539816]),
            dict(NAME='random_world_scaling',
                 WORLD_SCALE_RANGE=[0.95, 1.05])
        ]
    ),
    POINT_FEATURE_ENCODING=dict(
        encoding_type='absolute_coordinates_encoding',
        used_feature_list=['x', 'y', 'z', 'intensity'],
        # used_feature_list=['x', 'y', 'z'],
        src_feature_list=['x', 'y', 'z', 'intensity'],
    ),

    DATA_PROCESSOR=[
        dict(NAME='mask_points_and_boxes_outside_range',
             REMOVE_OUTSIDE_BOXES=True),
        dict(NAME='shuffle_points',
             SHUFFLE_ENABLED=dict(train=True, test=False)),
        dict(NAME='sample_points',
             NUM_POINTS=dict(train=32768, test=32768),
             )
        # dict(NAME='sample_points_by_voxel_once',
        #      VOXEL_SIZE=[0.1, 0.1, 0.15],
        #      MAX_POINTS_PER_VOXEL=5,
        #      MAX_NUMBER_OF_VOXELS=dict(train=80000, test=90000),
        #      NUM_POINTS=dict(train=65536, test=65536),
        # )
    ],
    metrics=[
        # dict(key='Car_3d/moderate_R40', summary='best', goal='maximize', save=True),
        # dict(key='Pedestrian_3d/moderate_R40', summary='best', goal='maximize'),
        # dict(key='Cyclist/moderate_R40', summary='best', goal='maximize')
    ]
)
