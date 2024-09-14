from navi_config import NaviConfig
import json
from load_sim_utils import sim_settings,make_cfg
from typing import Dict

from modelscope import AutoModel, AutoTokenizer
from constants import MapConstants as MC
import cv2
import numpy as np
import torch
import scipy
import math
import time
import json
import logging, os
import quaternion

import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from sklearn.cluster import DBSCAN
import skimage.morphology




from core.interfaces import Observations
from instance_memory import InstanceMemory
from goat_matching import GoatMatching

from dummy_policy import DummyPolicy

from model import SemanticMappingClean
from utils.fmm_planner import FMMPlanner
import utils.pose as pu
import utils.visualization as vu 
from utils.constants import color_palette

import numpy as np
import cv2
import habitat_sim
from perception.detection.detic.detic_perception import DeticPerception


class NavWithLLM():
    def __init__(self,
                sim,
                 device_id: int = 0):
        super().__init__()
        self.llm_goal_start=999
        self.should_stop_accumulation=0
        self.last_stop_time = None

        self.use_llm=True
        # used in update map
        self.sim=sim
        self.need_llm=True
        self.look_around_max_step=360/NaviConfig.ENVIRONMENT.turn_angle
        self.look_around_step=0

        self.llm = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True, torch_dtype=torch.float16)
        self.llm = self.llm.to(device=f'cuda:{NaviConfig.GPU_ID}')

        self.tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)
        self.llm.eval()

        self.device_id = device_id
        self.device = torch.device(f"cuda:{self.device_id}")
        self.args = NaviConfig

        self.time_step=0

        if NaviConfig.AGENT.panorama_start:
            self.panorama_start_steps = int(360 /
                                            NaviConfig.ENVIRONMENT.turn_angle)
        else:
            self.panorama_start_steps = 0
        self.panorama_rotate_steps = int(360 / NaviConfig.ENVIRONMENT.turn_angle)

        self.segmentation = DeticPerception(vocabulary="coco", sem_gpu_id=device_id)
        self.cate_id_to_name = self.segmentation.category_id_to_name                  
        
        self.num_environments = NaviConfig.NUM_ENVIRONMENTS
        self.num_scenes = self.num_environments
        self.num_sem_categories = len(self.segmentation.categories_mapping)
        


        self.record_instance_ids = NaviConfig.SEMANTIC_MAP.record_instance_ids
        if self.record_instance_ids:
            self.instance_memory = InstanceMemory(
                self.num_environments,
                NaviConfig.SEMANTIC_MAP.du_scale,
                debug_visualize=NaviConfig.visualize,
                config=NaviConfig,
                mask_cropped_instances=False,
                padding_cropped_instances=200,
                category_id_to_category_name=self.cate_id_to_name)

        self.goal_policy_config = NaviConfig.SUPERGLUE
        self.matching = GoatMatching(
            device=device_id,
            score_func=self.goal_policy_config.score_function,
            num_sem_categories=self.num_sem_categories,
            config=NaviConfig.SUPERGLUE,
            default_vis_dir=f"{NaviConfig.DUMP_LOCATION}/images",
            print_images=NaviConfig.save_images,
            instance_memory=self.instance_memory,
        )

        if self.goal_policy_config.batching:
            self.image_matching_function = self.matching.match_image_batch_to_image
        else:
            self.image_matching_function = self.matching.match_image_to_image

        self.expand2multi_binary = torch.eye(
            NaviConfig.SEMANTIC_MAP.num_sem_categories, device=self.device)

        self.sub_task_timesteps = None
        self.total_timesteps = None
        self.episode_panorama_start_steps = None
        self.reached_goal_panorama_rotate_steps = self.panorama_rotate_steps

        
        

        self.goal_image = None
        self.goal_mask = None
        self.goal_image_keypoints = None

        self.found_goal = torch.zeros(self.num_environments,
                                      1,
                                      dtype=bool,
                                      device=self.device)

        


        '''Parameters from module'''
        self.policy = DummyPolicy(
            exploration_strategy=NaviConfig.AGENT.exploration_strategy,
            device_id=self.device_id)
        self.goal_update_steps = self.policy.goal_update_steps
        self.goal_policy_config = NaviConfig.SUPERGLUE

        '''Parameters from map'''
        self.last_pose = None

        ''' Parameters for path planning '''
        self.stop = None

        ''' Parameters from map dev '''
        # --- Env Parameters
        self.last_sim_location = None
        self.goal_name = 'Testing'
        
        map_shape = (self.args.SEMANTIC_MAP.map_size_cm // self.args.SEMANTIC_MAP.map_resolution,
                     self.args.SEMANTIC_MAP.map_size_cm // self.args.SEMANTIC_MAP.map_resolution)
        self.collision_map = np.zeros(map_shape)
        self.visited = np.zeros(map_shape)
        self.visited_vis = np.zeros(map_shape)

        self.last_loc = None
        self.curr_loc = [self.args.SEMANTIC_MAP.map_size_cm / 100.0 / 2.0,
                         self.args.SEMANTIC_MAP.map_size_cm / 100.0 / 2.0, 0.]
        
        self.last_action = None
        self.col_width = None
        self.selem = skimage.morphology.disk(3)

        # --- Map Parameters ---
        self.res = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((self.args.ENVIRONMENT.frame_height, self.args.ENVIRONMENT.frame_width),
                               interpolation=Image.NEAREST)])

        
        self.vis_image = None
        self.rgb_vis = None 

        torch.set_grad_enabled(False)

        self.nc = self.args.SEMANTIC_MAP.num_sem_categories + MC.NON_SEM_CHANNELS  # num channels
        if self.args.SEMANTIC_MAP.record_instance_ids:
            self.nc += self.args.SEMANTIC_MAP.num_sem_categories
            

        # Calculating full and local map sizes
        map_size = self.args.SEMANTIC_MAP.map_size_cm // self.args.SEMANTIC_MAP.map_resolution
        self.full_w, self.full_h = map_size, map_size
        self.local_w = int(self.full_w / self.args.SEMANTIC_MAP.global_downscaling)
        self.local_h = int(self.full_h / self.args.SEMANTIC_MAP.global_downscaling)

        # Initializing full and local map
        self.full_map = torch.zeros(self.num_scenes, self.nc, self.full_w, self.full_h).float().to(self.device)
        self.local_map = torch.zeros(self.num_scenes, self.nc, self.local_w,
                                self.local_h).float().to(self.device)

        # Initial full and local pose
        self.full_pose = torch.zeros(self.num_scenes, 3).float().to(self.device)
        self.local_pose = torch.zeros(self.num_scenes, 3).float().to(self.device)


        # Origin of local map
        self.origins = np.zeros((self.num_scenes, 3))

        # Local Map Boundaries
        self.lmb = np.zeros((self.num_scenes, 4)).astype(int)

        # Planner pose inputs has 7 dimensions
        # 1-3 store continuous global agent location
        # 4-7 store local map boundaries
        self.planner_pose_inputs = np.zeros((self.num_scenes, 7))

        map_features_channels = 2 * MC.NON_SEM_CHANNELS + self.num_sem_categories
        if self.record_instance_ids:
            map_features_channels += self.num_sem_categories
        self.map_features = torch.zeros(
            self.num_scenes,
            map_features_channels,
            self.local_w,
            self.local_h,
            device=self.device,
        ).float()

        self.goal_map = torch.zeros(
            self.num_environments,
            1,
            *self.local_map.shape[2:],
            device=self.device,
        )

        self.frontier_map = torch.zeros(
            self.num_environments, 
            *self.local_map.shape[2:],
            device=self.device
        )

        self.semantic_goal_map = torch.zeros(
            self.num_environments, 
            *self.local_map.shape[2:],
            device=self.device
        )

        self.sem_map_module = SemanticMappingClean(self.instance_memory, device=self.device).to(self.device)
        self.sem_map_module.eval() 
        
        
        



    def reset_vectorized(self):
        """Initialize agent state."""
        self.total_timesteps = [0] * self.num_environments
        


        # self.instance_map.init_instance_map()
        self.episode_panorama_start_steps = self.panorama_start_steps
        self.reached_goal_panorama_rotate_steps = self.panorama_rotate_steps
        if self.instance_memory is not None:
            self.instance_memory.reset()
        
        


        self.goal_image = None
        self.goal_mask = None
        self.goal_image_keypoints = None

        self.found_goal[:] = False
        self.goal_map[:] *= 0


        self.init_map_and_pose()

    def reset(self,tgt):
        """Initialize agent state."""
        self.llm_goal_start=999
        self.last_stop_time=None
        self.should_stop_accumulation=0
        self.llm_last_goal_map=None
        self.last_sim_location=self.get_curr_pose()

        self.reset_vectorized()
        self.goal_image = None
        self.goal_mask = None
        self.goal_image_keypoints = None
        
        if self.args.visualize or self.args.save_images:
            self.vis_image = vu.init_vis_image(tgt, None)
        
        # from env parameters
        map_shape = (self.args.SEMANTIC_MAP.map_size_cm // self.args.SEMANTIC_MAP.map_resolution,
                     self.args.SEMANTIC_MAP.map_size_cm // self.args.SEMANTIC_MAP.map_resolution)
        self.collision_map = np.zeros(map_shape)
        self.visited = np.zeros(map_shape)
        self.visited_vis = np.zeros(map_shape)
        self.curr_loc = [self.args.SEMANTIC_MAP.map_size_cm / 100.0 / 2.0,
                         self.args.SEMANTIC_MAP.map_size_cm / 100.0 / 2.0, 0.]
        self.last_action = None
        self.col_width = 1
        

    def reset_sub_episode(self) -> None:
        """Reset for a new sub-episode since pre-processing is temporally dependent."""
        self.total_timesteps[0]=0

        self.goal_image = None
        self.goal_image_keypoints = None
        self.goal_mask = None

    
    def reset_vectorized_for_env(self, e: int):
        """Initialize agent state for a specific environment."""
        self.total_timesteps[e] = 0
        self.sub_task_timesteps[e] = [0] * self.max_num_sub_task_episodes

        # self.semantic_map.init_map_and_pose_for_env(e)
        self.episode_panorama_start_steps = self.panorama_start_steps
        self.reached_goal_panorama_rotate_steps = self.panorama_rotate_steps
        if self.instance_memory is not None:
            self.instance_memory.reset_for_env(e)

        self.goal_image = None
        self.goal_image_keypoints = None
        self.goal_mask = None


    def get_curr_pose(self):
        agent_state = self.sim.agents[0].get_state()

        x = -agent_state.position[2]
        y = -agent_state.position[0]
        axis = quaternion.as_euler_angles(agent_state.rotation)[0]
        if (axis % (2 * np.pi)) < 0.1 or (axis %
                                          (2 * np.pi)) > 2 * np.pi - 0.1:
            o = quaternion.as_euler_angles(agent_state.rotation)[1]
        else:
            o = 2 * np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
        if o > np.pi:
            o -= 2 * np.pi
        curr_sim_pose=(x, y, o)

        return curr_sim_pose

    def get_pose_change(self):
        """Returns dx, dy, do pose change of the agent relative to the last
        timestep."""
        
        curr_sim_pose=self.get_curr_pose()
        dx, dy, do = pu.get_rel_pose_change(
            curr_sim_pose, self.last_sim_location)
        self.last_sim_location = curr_sim_pose
        return dx, dy, do


    # --- map utils ---
    def get_local_map_boundaries(self, agent_loc, local_sizes, full_sizes):
        loc_r, loc_c = agent_loc
        local_w, local_h = local_sizes
        full_w, full_h = full_sizes

        if self.args.SEMANTIC_MAP.global_downscaling > 1:
            gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
            gx2, gy2 = gx1 + local_w, gy1 + local_h
            if gx1 < 0:
                gx1, gx2 = 0, local_w
            if gx2 > full_w:
                gx1, gx2 = full_w - local_w, full_w

            if gy1 < 0:
                gy1, gy2 = 0, local_h
            if gy2 > full_h:
                gy1, gy2 = full_h - local_h, full_h
        else:
            gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h

        return [gx1, gx2, gy1, gy2]
    
    def init_map_and_pose(self):
        self.full_map.fill_(0.)
        self.full_pose.fill_(0.)
        # start location
        self.full_pose[:, :2] = self.args.SEMANTIC_MAP.map_size_cm / 100.0 / 2.0

        locs = self.full_pose.cpu().numpy()
        self.planner_pose_inputs[:, :3] = locs
        for e in range(self.num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / self.args.SEMANTIC_MAP.map_resolution),
                            int(c * 100.0 / self.args.SEMANTIC_MAP.map_resolution)]

            self.full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

            self.lmb[e] = self.get_local_map_boundaries((loc_r, loc_c),
                                              (self.local_w, self.local_h),
                                              (self.full_w, self.full_h))

            self.planner_pose_inputs[e, 3:] = self.lmb[e]
            self.origins[e] = [self.lmb[e][2] * self.args.SEMANTIC_MAP.map_resolution / 100.0,
                          self.lmb[e][0] * self.args.SEMANTIC_MAP.map_resolution / 100.0, 0.]

        for e in range(self.num_scenes):
            self.local_map[e] = self.full_map[e, :,
                                    self.lmb[e, 0]:self.lmb[e, 1],
                                    self.lmb[e, 2]:self.lmb[e, 3]]
            self.local_pose[e] = self.full_pose[e] - \
                torch.from_numpy(self.origins[e]).to(self.device).float()

    @torch.no_grad()
    def update_maps(self,rgb,depth,semantic,instances,
                    ):
        '''
            rgb: h,w,c
            depth: h,w,c
            semantic: h,w,num_sem
            instances: h,w,num_inst
        '''
        pose_change=torch.tensor(self.get_pose_change()).unsqueeze(0)
        pose_change=pose_change.to(self.device)

        batch_size=1

        # Update local map and possibly llm goal map
        _, self.local_map, _, self.local_pose,self.llm_last_goal_map = \
            self.sem_map_module.forward(
                batch_size,rgb,depth, semantic,instances,
                pose_change, 
                self.local_map, 
                self.local_pose,
                torch.tensor(self.origins, device=self.device),
                torch.tensor(self.lmb, device=self.device),
                self.llm_last_goal_map)

        # Reset pose
        locs = self.local_pose.cpu().numpy()
        self.planner_pose_inputs[:, :3] = locs + self.origins # the new global pose
        self.local_map[:, 2, :, :].fill_(0.)  # Resetting current location channel

        # Update global map, origins, lmb
        for e in range(self.num_scenes):
            r, c = locs[e, 1], locs[e, 0] # y, x
            loc_r, loc_c = [int(r * 100.0 / self.args.SEMANTIC_MAP.map_resolution),
                            int(c * 100.0 / self.args.SEMANTIC_MAP.map_resolution)]
            self.local_map[e, 2:4, loc_r - 2:loc_r + 3, loc_c - 2:loc_c + 3] = 1.
            try:
                # print('set explored area')
                radius = self.args.SEMANTIC_MAP.explored_radius // self.args.SEMANTIC_MAP.map_resolution
                explored_disk = torch.from_numpy(skimage.morphology.disk(radius))
                # print('disk shape', explored_disk.shape)
                self.local_map[ e, 
                                1, 
                                loc_r - radius : loc_r + radius + 1,
                                loc_c - radius : loc_c + radius + 1,][explored_disk == 1] = 1
            except IndexError:
                print("An index error occurred when setting explored area.")
                pass


            self.sem_map_module._update_global_map_and_pose_for_env(
                e, 
                self.local_map,
                self.full_map,
                self.local_pose,
                self.full_pose,
                self.lmb,
                self.origins
                )

            self.full_pose[e] = self.local_pose[e] + \
                torch.from_numpy(self.origins[e]).to(self.device).float()
            
            # local map Recenter
            locs = self.full_pose[e].cpu().numpy()
            r, c = locs[1], locs[0]
            loc_r, loc_c = [int(r * 100.0 / self.args.SEMANTIC_MAP.map_resolution),
                            int(c * 100.0 / self.args.SEMANTIC_MAP.map_resolution)]

            self.lmb[e] = self.get_local_map_boundaries((loc_r, loc_c),
                                                (self.local_w, self.local_h),
                                                (self.full_w, self.full_h))
            
            self.planner_pose_inputs[e, 3:] = self.lmb[e]
            self.origins[e] = [self.lmb[e][2] * self.args.SEMANTIC_MAP.map_resolution / 100.0,
                            self.lmb[e][0] * self.args.SEMANTIC_MAP.map_resolution / 100.0, 0.]

            self.local_map[e] = self.full_map[e, :,
                                    self.lmb[e, 0]:self.lmb[e, 1],
                                    self.lmb[e, 2]:self.lmb[e, 3]]
            self.local_pose[e] = self.full_pose[e] - \
                torch.from_numpy(self.origins[e]).to(self.device).float()
    
   
    def should_stop(self,image,tgt):
        if tgt=='toilet':
            question = f'Is there {tgt} in the image? yes or no'
        else:
            question = f'is {tgt} in the foreground of the image? yes or no'

        image=Image.fromarray(image,'RGB')
        answer=self.llm_speak(image,question)
        print(answer+'\n')

        if 'yes' in answer:
            return True
        else:
            return False
    

    
    def _plan(self, map_pred,goal,curr_pose):
        """Function responsible for planning

        Args:
            
            'map_pred'  (ndarray): (M, M) map prediction
            'goal'      (ndarray): (M, M) goal locations
            'curr_pose' (ndarray): (7,) array  denoting pose (x,y,o)
                            and planning window (gx1, gx2, gy1, gy2)

        Returns:
            action (int): action id
        """
        args = self.args
        self.last_loc = self.curr_loc

        # Get Map prediction
        map_pred = np.rint(map_pred)

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = curr_pose

        planning_window = [gx1, gx2, gy1, gy2]
        planning_window=[int(num) for num in planning_window]
        gx1, gx2, gy1, gy2 = planning_window
        # Get curr loc
        self.curr_loc = [start_x, start_y, start_o]
        r, c = start_y, start_x
        start = [int(r * 100.0 / args.SEMANTIC_MAP.map_resolution - gx1),
                 int(c * 100.0 / args.SEMANTIC_MAP.map_resolution - gy1)]
        start = pu.threshold_poses(start, map_pred.shape)

        self.visited[gx1:gx2, gy1:gy2][start[0] - 0:start[0] + 1,
                                       start[1] - 0:start[1] + 1] = 1

        if args.visualize or args.save_images:
            # Get last loc
            last_start_x, last_start_y = self.last_loc[0], self.last_loc[1]
            r, c = last_start_y, last_start_x
            last_start = [int(r * 100.0 / args.SEMANTIC_MAP.map_resolution - gx1),
                          int(c * 100.0 / args.SEMANTIC_MAP.map_resolution - gy1)]
            last_start = pu.threshold_poses(last_start, map_pred.shape)
            self.visited_vis[gx1:gx2, gy1:gy2] = \
                vu.draw_line(last_start, start,
                             self.visited_vis[gx1:gx2, gy1:gy2])

        # Collision check
        # print('last action', self.last_action)
        if self.last_action == 'move_forward': # move forward
            x1, y1, t1 = self.last_loc
            x2, y2, _ = self.curr_loc
            buf = 4
            length = 2

            if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
                self.col_width += 2
                if self.col_width == 7:
                    length = 4
                    buf = 3
                self.col_width = min(self.col_width, 5)
            else:
                self.col_width = 1

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            if dist < args.PLANNER.collision_threshold:  # Collision
                width = self.col_width
                for i in range(length):
                    for j in range(width):
                        wx = x1 + 0.05 * \
                            ((i + buf) * np.cos(np.deg2rad(t1))
                             + (j - width // 2) * np.sin(np.deg2rad(t1)))
                        wy = y1 + 0.05 * \
                            ((i + buf) * np.sin(np.deg2rad(t1))
                             - (j - width // 2) * np.cos(np.deg2rad(t1)))
                        r, c = wy, wx
                        r, c = int(r * 100 / args.SEMANTIC_MAP.map_resolution), \
                            int(c * 100 / args.SEMANTIC_MAP.map_resolution)
                        [r, c] = pu.threshold_poses([r, c],
                                                    self.collision_map.shape)
                        self.collision_map[r, c] = 1

        stg, stop = self._get_stg(map_pred, start, np.copy(goal),
                                  planning_window)
        (stg_x, stg_y) = stg
        angle_st_goal = math.degrees(math.atan2(stg_x - start[0],
                                                stg_y - start[1]))
        angle_agent = (start_o) % 360.0
        if angle_agent > 180:
            angle_agent -= 360

        relative_angle = (angle_agent - angle_st_goal) % 360.0
        if relative_angle > 180:
            relative_angle -= 360

        if relative_angle > self.args.ENVIRONMENT.turn_angle / 2.:
            action = 'turn_right'  # Right: 3
        elif relative_angle < -self.args.ENVIRONMENT.turn_angle / 2.:
            action = 'turn_left'  # Left: 2
        else:
            action = 'move_forward'  # Forward: 1

        return action,stop

    def _get_stg(self, grid, start, goal, planning_window):
        """Get short-term goal"""

        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = 0, 0
        x2, y2 = grid.shape

        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h + 2, w + 2)) + value
            new_mat[1:h + 1, 1:w + 1] = mat
            return new_mat

        traversible = skimage.morphology.binary_dilation(
            grid[x1:x2, y1:y2],
            self.selem) != True
        traversible[self.collision_map[gx1:gx2, gy1:gy2]
                    [x1:x2, y1:y2] == 1] = 0
        traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1

        traversible[int(start[0] - x1) - 1:int(start[0] - x1) + 2,
                    int(start[1] - y1) - 1:int(start[1] - y1) + 2] = 1

        traversible = add_boundary(traversible)
        goal = add_boundary(goal, value=0)

        planner = FMMPlanner(traversible)
        
        selem = skimage.morphology.disk(NaviConfig.ENVIRONMENT.round_rate)
        goal = skimage.morphology.binary_dilation(
            goal, selem) != True
        goal = 1 - goal * 1.
        planner.set_multi_goal(goal)

        state = [start[0] - x1 + 1, start[1] - y1 + 1]
        # print('state',state)
        stg_x, stg_y, _, stop = planner.get_short_term_goal(state)

        stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1

        return (stg_x, stg_y), stop

    def _visualize(self, exp_pred,pose_pred,
                   map_pred,sem_map_pred, goal_map,
                   epi_id, tgt='test'):
        args = self.args
        dump_dir = f"{args.DUMP_LOCATION}/{epi_id}_{tgt}"
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)
 
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = pose_pred

        sem_map = sem_map_pred

        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)

        sem_map += 5

        no_cat_mask = sem_map == 20
        map_mask = np.rint(map_pred) == 1
        exp_mask = np.rint(exp_pred) == 1
        vis_mask = self.visited_vis[gx1:gx2, gy1:gy2] == 1

        sem_map[no_cat_mask] = 0
        m1 = np.logical_and(no_cat_mask, exp_mask)
        sem_map[m1] = 2

        m2 = np.logical_and(no_cat_mask, map_mask)
        sem_map[m2] = 1

        sem_map[vis_mask] = 3

        selem = skimage.morphology.disk(4)

        
        color_pal = [int(x * 255.) for x in color_palette]
        sem_map_vis = Image.new("P", (sem_map.shape[1],
                                      sem_map.shape[0]))
        sem_map_vis.putpalette(color_pal)
        sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
        sem_map_vis = sem_map_vis.convert("RGB")
        sem_map_vis = np.flipud(sem_map_vis)

        sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
        sem_map_vis = cv2.resize(sem_map_vis, (480, 480),
                                 interpolation=cv2.INTER_NEAREST)
        self.vis_image[50:530, 15:655] = self.rgb_vis
        self.vis_image[50:530, 670:1150] = sem_map_vis

        pos = (
            (start_x * 100. / args.SEMANTIC_MAP.map_resolution - gy1)
            * 480 / map_pred.shape[0],
            (map_pred.shape[1] - start_y * 100. / args.SEMANTIC_MAP.map_resolution + gx1)
            * 480 / map_pred.shape[1],
            np.deg2rad(-start_o)
        )

        agent_arrow = vu.get_contour_points(pos, origin=(670, 50))
        color = (int(color_palette[11] * 255),
                 int(color_palette[10] * 255),
                 int(color_palette[9] * 255))
        cv2.drawContours(self.vis_image, [agent_arrow], 0, color, -1)

        if args.visualize:
            # Displaying the image
            cv2.imshow("Thread {}".format(tgt), self.vis_image)
            if goal_map is not None:
                plt.imshow(np.flipud(goal_map*200))
            cv2.waitKey(1)

        if args.save_images:
            img_save_path=dump_dir+f'/vis-{self.total_timesteps[0]}.png'
            cv2.imwrite(img_save_path, self.vis_image)

    def llm_speak(self,image,question):
        
        answer = self.llm.chat(
                image=image,
                msgs=[{'role': 'user', 'content': question}],
                tokenizer=self.tokenizer,
                sampling=True,
                temperature=0.7,
            )
            
            
        return answer.lower()
    
    def step(self, observes: Dict, tgt,epi_id,timestep) -> str:
        print(f'lang tgt now is {tgt}')
        print('llm_goal_start: ',self.llm_goal_start)
        
        rgb_image = observes['color_sensor'][..., :3][..., ::-1] # (480 640 3)
        
        obs = Observations(
            gps=[0,0],  ### TODO
            compass=[0],  ### TODO
            rgb=rgb_image,
            depth=observes['depth_sensor'],
            camera_pose=None,
            third_person_image=None)

        obs = self.segmentation.predict(obs)
        
        self.update_vis(obs,observes['depth_sensor'])
        
        llm_guide=None
        possilbe_llm_stop=False
        
        
        if self.use_llm:
            np_img=np.uint8(rgb_image[:,:,[2,1,0]])
            print("\033[1;31m SHOULD STOP \033[0m")
            possilbe_llm_stop=self.should_stop(np_img,tgt)
            if possilbe_llm_stop:
                if self.last_stop_time is None:
                    self.last_stop_time=timestep

                if self.last_stop_time+1==timestep:
                    self.should_stop_accumulation+=1
                else:
                    self.should_stop_accumulation=1
                self.last_stop_time=timestep
            

            if timestep > self.llm_goal_start+NaviConfig.ENVIRONMENT.llm_endurance: # 5 for debug
                print("\033[1;36m give up former llm point! \033[0m")
                # cleaning
                self.local_map[:, MC.NON_SEM_CHANNELS+16,:,:]=0
                self.need_llm=True
                self.llm_goal_start=999

            if self.need_llm:
                question = f'which direction should I go to find {tgt} based on the image? explore more, left, right or straight to the front?'
                image=Image.fromarray(np_img,'RGB')
                print("\033[1;31m WHERE \033[0m")
                answer=self.llm_speak(image,question)
                print(answer+'\n')  
                # def convert_direction(answer):
                #     ans=self.llm_speak(None,f'what is the best option based on this discription: {answer}. Speak only left, right, go straight and see more')
                #     return ans
                # direction=convert_direction(answer)

                def if_explore(answer):
                    if f'no {tgt}' in answer\
                            or 'does not contain' in answer\
                                or 'explore further' in answer \
                                or 'explore more' in answer \
                                or 'no direct indication' in answer\
                                or 'different' in answer\
                                or 'explore other' in answer\
                                or 'not possible' in answer\
                                or 'does not include' in answer\
                                    or 'elsewhere' in answer:
                            return True
                    else:
                        False
                
                
                if if_explore(answer):
                    # 'see more' in direction:
                    print('in llm explore more!')
                    llm_guide='explore more'
                
                elif 'left' in answer:
                    print('in llm left!')
                    obs.semantic=self.render(obs.semantic, 'left')
                    llm_guide='left'
                elif 'right' in answer:
                    print('in llm right!')
                    obs.semantic=self.render(obs.semantic, 'right')
                    llm_guide='right'
                elif 'go straight' in answer:
                    print('in llm foreground!')
                    # foreground is mainly for specific objs such as apple
                    # not suitable for a scene or an area
                    # pure foreground projection shall fail
                    obs.semantic=self.render(obs.semantic, 'foreground')
                    llm_guide='foreground'
                elif 'straight to the front' in answer:
                    print('in llm background!')
                    # center bottom
                    obs.semantic=self.render(obs.semantic, 'foreground')
                    llm_guide='background'

                else:
                    pass
            else:
                pass
        else:
            # debug
            obs.semantic=self.render(obs.semantic, 'foreground')
            pass
        

        semantic,instances=self.transform_semantic_and_instances(
            obs.semantic,obs.task_observations['instance_map'])
        

        # Downscaling factor
        ds = self.args.ENVIRONMENT.env_frame_width // self.args.ENVIRONMENT.frame_width  
        if ds != 1:
            rgb,depth,semantic=self.downscale(obs.rgb,obs.depth,semantic,ds)        

        instances=self.record_instance(instances,ds)
        
        self.update_maps(rgb,depth,semantic,instances)
        
        goal_map=None
        look_around=False
        final_goal=False
        current_proj=self.local_map[0, 4:4+self.num_sem_categories, :,:].argmax(0)
        
 
        if llm_guide=='explore more' :
            goal_map=self.frontier_goal_map()
            print('llm frontier point!')
        
        else:
            possible_llm_map=self.local_map[0, MC.NON_SEM_CHANNELS+16, :,:] # llm num is 16
            possible_llm_map=self.cluster_goal_map(possible_llm_map.cpu().numpy())
            if isinstance(possible_llm_map,torch.Tensor):
                possible_llm_map=possible_llm_map.cpu().numpy()

            if len(np.unique(possible_llm_map))>1:
                self.need_llm=False
                # llm goal step limitation
                if self.llm_goal_start == 999:
                    self.llm_goal_start=timestep

                goal_map=possible_llm_map
                print('llm point!')
            else:
                self.need_llm=True
                if self.look_around_step<self.look_around_max_step:
                    print('fixed look around...')
                    self.look_around_step+=1
                    look_around=True
                else:
                    goal_map=self.frontier_goal_map()
                    print('frontier point!')
    
        e=0
        obstacle_map=self.local_map[e, 0, :, :].cpu().numpy()
        explored_map=self.local_map[e, 1, :, :].cpu().numpy()
        
        if self.args.visualize or self.args.save_images:
            vis_sem_map=self.local_map[e, 4:4+self.num_sem_categories, :,
                                                    :].clone()

            vis_sem_map[15, :, :] = 1e-5
            temp=vis_sem_map[16,:,:].cpu().numpy()
            clustered=torch.from_numpy(self.cluster_goal_map(temp)).to(vis_sem_map.device)
            vis_sem_map[16,:,:]=clustered
            sem_map_pred= vis_sem_map.argmax(0).cpu().numpy()           
        
        
        if not self.need_llm:
            print('llm planning')
            action, stg_reached = self._plan(obstacle_map,
                                            goal_map,
                                            self.planner_pose_inputs[e])
            print(f'\033[0;33;40mreached short term goal: {stg_reached}\033[0m')
            if stg_reached:
                self.look_around_step=0
                self.need_llm=True
                self.llm_goal_start = 999

                # if possilbe_llm_stop:
                if final_goal or possilbe_llm_stop:
                    action='terminate'
                print("\033[1;31m cleaning last llm goal! \033[0m")
                
                # clean and get new llm guide
                # top-down map, binary 
                self.local_map[:, MC.NON_SEM_CHANNELS+16,:,:]=0
                
                # for visualization
                vis_sem_map[16,:,:]=0
        else:
            if look_around:
                action='turn_right'
            else:
                print('frontier planning')
                action, stg_reached = self._plan(obstacle_map,
                                                    goal_map,
                                                    self.planner_pose_inputs[e])
                if stg_reached:
                    self.look_around_step=0
            
        print('action:', action)
        self.last_action = action
        
        if NaviConfig.visualize or NaviConfig.save_images:
            self._visualize(
                goal_map=goal_map,
                exp_pred=explored_map,
                pose_pred=self.planner_pose_inputs[e],
                map_pred=obstacle_map,
                sem_map_pred=sem_map_pred,
                tgt=tgt,
                epi_id=epi_id)
        
        
        self.total_timesteps[0] = self.total_timesteps[0] + 1

        if action == 'terminate':
            print(f'find {tgt} task is done! spend {self.total_timesteps[0]} steps!')
            self.reset_sub_episode()
                

        logging.info("========== total timestep: {} ==========".format(self.total_timesteps))
        print("======= total timestep: {} =======".format(self.total_timesteps))
        

        cv2.waitKey(10)
    
        return action

    def frontier_goal_map(self)->np.array:
        '''
            return ndarray
        '''
        self.map_features = self.sem_map_module._get_map_features(self.local_map, self.full_map).unsqueeze(1)
        map_features = self.map_features.flatten(0, 1)
        frontier_map = self.policy.get_frontier_map(map_features)
        self.goal_map = frontier_map
        self.goal_map=self.goal_map.squeeze(0).squeeze(0)
        
        return self.goal_map.cpu().numpy()

    def record_instance(self,instances,ds):
        instance_ids = np.unique(instances)
        
        instance_id_to_idx = {
            instance_id: idx
            for idx, instance_id in enumerate(instance_ids)
        }

        instances = torch.from_numpy(
            np.vectorize(instance_id_to_idx.get)(instances)).to(
                self.device)
        ### 1st channel: background, 2...: instances
        instances = torch.eye(len(instance_ids),
                                device=self.device)[instances]
        
        instances = instances[ds // 2::ds, ds // 2::ds]

        return instances
    
    def register_semantic(self, key):
        self.unregistered_cat = []
        reg_key=list(self.segmentation.categories_mapping.keys())
        if key not in reg_key:
            if key != 0:
                self.unregistered_cat.append(key)
            return 0
        else:
            return key
        

    def transform_semantic_and_instances(self,
                                         obs_semantic,obs_instance_map):
            
        # Get rid of unregistered categories
        semantic = np.vectorize(self.register_semantic)(obs_semantic)
        # semantic = np.vectorize(lambda k: k if k in reg_key else 0)(obs_semantic)
        
        if self.record_instance_ids:
            instances = obs_instance_map
            # those with unregistered categories
            instances_mask = semantic == 0 
            instances[instances_mask] = -1
        
        # id mapping
        semantic = np.vectorize(
            self.segmentation.categories_mapping.get)(semantic)
        
        

        semantic = self.expand2multi_binary[semantic].to(self.device)
        # semantic 生成category个纬度，且每个维度上都是对应cat的0 1 map

        return semantic,instances
    
    def downscale(self,obs_rgb,obs_depth,semantic,ds):
        # downscaling rgb, depth, and semantic
        rgb = torch.from_numpy(obs_rgb.copy()).to(self.device)
        ### Do not transform the depth
        depth = (torch.from_numpy(obs_depth).unsqueeze(-1).to(self.device)) * 100


        rgb = np.asarray(self.res(rgb.cpu().numpy().astype(np.uint8)))
        rgb = torch.tensor(rgb, device=self.device)
        depth = depth[ds // 2::ds, ds // 2::ds]
        semantic = semantic[ds // 2::ds, ds // 2::ds]
        
        return rgb,depth,semantic

    def render(self,semantic,pos):
        
        h,w=semantic.shape[-2:]
        square_size = 200 

        if pos in ['left','right']:
            start_row = (h - square_size) // 2
            end_row = start_row + square_size
            
            if pos=='left':
                start_col = w//4  - square_size//2    
            elif pos=='right':
                start_col = w-w//4  - square_size//2
        elif pos in ['background','foreground']:
            start_col = (w- square_size)//2
            if pos=='background':
                start_row = h//4 - square_size// 2
            elif pos=='foreground':
                start_row = h- h//2 - square_size// 2
                
            
        end_row = start_row + square_size
        end_col = start_col + square_size
        
        semantic[start_row:end_row, start_col:end_col] = 16
        
        return semantic
    
    def update_vis(self,obs,observe_depth):
        depth_image = np.clip(observe_depth, 0, 10) / 10 * 255
        
        depth_image = np.repeat(depth_image[..., None].astype(np.uint8),
                                3,
                                axis=-1)
        # for paper vis, this is for seg vis
        # self.rgb_vis = obs.task_observations["semantic_frame"]
        self.rgb_vis=obs.rgb
        agent_view = np.concatenate(
            [self.rgb_vis, depth_image], axis=1)
        
        if NaviConfig.visualize:
            cv2.imshow('view', agent_view)

    
    def cluster_goal_map(self,goal_map) -> None:
        """
        Perform optional clustering of the goal channel to mitigate noisy projection
        splatter.
        """
        # cluster goal points

        try:
            c = DBSCAN(eps=6, min_samples=1)
            data = np.array(goal_map.nonzero()).T
            c.fit(data)

            # mask all points not in the largest cluster
            mode = scipy.stats.mode(c.labels_, keepdims=False).mode.item()
            mode_mask = (c.labels_ != mode).nonzero()
            x = data[mode_mask]
            goal_map_ = np.copy(goal_map)
            goal_map_[x] = 0.0

            # adopt masked map if non-empty
            if goal_map_.sum() > 0:
                goal_map = goal_map_
        except Exception as e:
            print(e)
            return goal_map

        return goal_map
 

if __name__=='__main__':
    
    env_path='./example_glb/Dd4bFSTQ8gi.glb'
    sim_settings["scene"] = env_path
    cfg = make_cfg(sim_settings)
    simulator = habitat_sim.Simulator(cfg)
    goal='some where I can rest'


    agent_traj=[]

    # reset
    agent = simulator.initialize_agent(sim_settings["default_agent"])
    # Set agent state
    agent_state = habitat_sim.AgentState()
    agent_traj.append(agent_state.position)

    # start navigation
    nav_with_llm=NavWithLLM(simulator,device_id=NaviConfig.GPU_ID)
    
    nav_with_llm.reset(goal)
    action='turn_right'

    max_step=200
    step=0
    while True:
        
        obs=simulator.step(action)
        action=nav_with_llm.step(obs,goal,
                                       epi_id=999,timestep=step)
        
        # record agent state
        agent_state = agent.get_state()
        now_pos=agent_state.position
        agent_traj.append(now_pos)
        step+=1

        if action=='terminate' or step==max_step:
            break

        
    simulator.close()
    exit()