
class NaviConfig:
    llm_id=16
    visualize= False
    save_images= True

    NO_GPU= 0                 # 1: ignore IDs above and run on CPU, 0: run on GPUs with IDs above
    GPU_ID= 1
    NUM_ENVIRONMENTS= 1      # number of environments (per agent process)
    DUMP_LOCATION= 'datadump'   # path to dump models and log
    matching_etc= 'dummy'       # dummy experiment name
    seed= 1


    # Environment Config
    class ENVIRONMENT:
        llm_endurance=10
        reach_distance= 0.8     # in meters
        round_rate=11 
        threshold=5 # in meters
        max_num_sub_task_episodes= 5
        forward_distance=0.3    # meters
        turn_angle= 30.0        # agent turn angle (in degrees)
        min_depth= 0.5          # minimum depth for depth sensor (in metres)
        max_depth= 5.0          # maximum depth for depth sensor (in metres)
        env_frame_width= 640
        env_frame_height= 480
        frame_width= 160
        frame_height= 120
        sensor_pose= [0., 1.1 , 0.]


    class AGENT:
        max_steps= 500          # maximum number of steps before stopping an episode
        panorama_start= 1       # 1= turn around 360 degrees when starting an episode, 0= don't
        exploration_strategy= 'seen_frontier'  # exploration strategy ("seen_frontier", "been_close_to_frontier")
        radius= 0.05            # robot radius (in meters)
        store_all_categories= False  # whether to store all semantic categories in the map or just task-relevant ones

    class SEMANTIC_MAP:
        semantic_categories= 'langnav_cat' # map semantic channel categories ("coco_indoor", "longtail_indoor", "mukul_indoor")
        num_sem_categories= 23           # number of map semantic channel categories (16, 257, 35)
        global_downscaling= 2    # ratio of global over local map
        du_scale= 1              # frame downscaling before projecting to point cloud

        hfov= 90
        camera_height= 0.88
        num_processes= 1

        map_size_cm= 2400
        map_resolution= 5
        vision_range= 100        # diameter of local map region visible by the agent (in cells)
        cat_pred_threshold= 5.0  # number of depth points to be in bin to classify it as a certain semantic category
        explored_map_threshold= 1.0  # number of depth points to be in bin to consider it as explored
        obstacle_map_threshold= 1.0  # number of depth points to be in bin to consider it as obstacle

        frame_width= 160
        frame_height= 120
        env_frame_width= 640
        env_frame_height= 480

        # erosion and filtering to reduce the number of spurious artifacts
        dilate_obstacles= False
        dilate_size= 3
        dilate_iter= 1
        exploration_type= 'default'
        max_depth= 5.0 # hacky (for goat agent module)

        explored_radius= 150 # radius (in centimeters) of visually explored region
        goal_filtering= True
        record_instance_ids= True  # whether to predict and store instance ids in the map

    class PLANNER:
        collision_threshold= 0.20       # forward move distance under which we consider there's a collision (in meters)
        min_obs_dilation_selem_radius= 2    # radius (in cells) of obstacle dilation structuring element
        obs_dilation_selem_radius= 2    # radius (in cells) of obstacle dilation structuring element
        goal_dilation_selem_radius= 10  # radius (in cells) of goal dilation structuring element
        use_dilation_for_stg= False      # use dilated goals for estimating short-term goals - or just reaching
        map_downsample_factor= 1            # optional downsampling of traversible and goal map before fmm distance call (1 for no downsampling, 2 for halving resolution)
        map_update_frequency= 1             # compute fmm distance map every n steps 
        step_size= 5                    # maximum distance of the short-term goal selected by the planner
        discrete_actions= True         # discrete motion planner output space or not
        planner_type= "old"             # ("old", "new") where "new" is the latest one being used for spot in real world

    class SUPERGLUE:
        max_keypoints= 1024
        keypoint_threshold= 0.005
        nms_radius= 4
        superglue_model= 'indoor'           # or outdoor
        sinkhorn_iterations= 20
        match_threshold= 0.2
        score_function= 'confidence_sum'    # or match_count
        score_thresh_image= 24.5  # real-world experiments used 6.0
        score_thresh_lang= 0.24
        match_projection_threshold= 0.2   # confidence must be at least this high to project as goal point.
        goto_past_pose= False
        batching= False
    
    coco_full_cat2name={
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
    }