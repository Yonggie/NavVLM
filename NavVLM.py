import random
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

vlm_model='openbmb/MiniCPM-V-2_6'
class NavWithLLM():
    def __init__(self,
                sim,
                 device_id=NaviConfig.GPU_ID):
        super().__init__()
        self.llm_goal_start=999
        self.history_buffer=[]

        self.use_llm=True
        # used in update map
        self.sim=sim
        self.need_llm=True
        self.look_around_max_step=360/NaviConfig.ENVIRONMENT.turn_angle
        self.look_around_step=0

        self.llm = AutoModel.from_pretrained(vlm_model, trust_remote_code=True, torch_dtype=torch.float16)
        self.llm = self.llm.to(device=f'cuda:{device_id}')

        self.tokenizer = AutoTokenizer.from_pretrained(vlm_model, trust_remote_code=True)
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

        self.segmentation = DeticPerception(vocabulary="coco", sem_gpu_id=device_id)
        self.cate_id_to_name = self.segmentation.category_id_to_name                  
        
        self.num_env = NaviConfig.NUM_ENVIRONMENTS
        self.num_sem_categories = len(self.segmentation.categories_mapping)
        

        self.record_instance_ids = NaviConfig.SEMANTIC_MAP.record_instance_ids
        if self.record_instance_ids:
            self.instance_memory = InstanceMemory(
                self.num_env,
                NaviConfig.SEMANTIC_MAP.du_scale,
                debug_visualize=NaviConfig.visualize,
                config=NaviConfig,
                mask_cropped_instances=False,
                padding_cropped_instances=200,
                category_id_to_category_name=self.cate_id_to_name)

        self.goal_policy_config = NaviConfig.SUPERGLUE
        self.matching = GoatMatching(
            device=NaviConfig.GPU_ID,
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


        self.policy = DummyPolicy(
            exploration_strategy=NaviConfig.AGENT.exploration_strategy,
            device_id=self.device_id)

        

        # --- visualization Parameters ---
        self.trans = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((self.args.ENVIRONMENT.frame_height, self.args.ENVIRONMENT.frame_width),
                               interpolation=Image.NEAREST)])

        self.vis_image = None
        self.rgb_vis = None 

        torch.set_grad_enabled(False)

        self.nc = self.args.SEMANTIC_MAP.num_sem_categories + MC.NON_SEM_CHANNELS  # num channels
        if self.args.SEMANTIC_MAP.record_instance_ids:
            self.nc += self.args.SEMANTIC_MAP.num_sem_categories
            
        '''Parameters from slam'''
        self.last_pose = None
        self.last_sim_location = None
        
        map_size = self.args.SEMANTIC_MAP.map_size_cm // self.args.SEMANTIC_MAP.map_resolution
        self.full_w, self.full_h = map_size, map_size
        self.map_shape = (self.full_w, self.full_h)
        self.collision_map = np.zeros(self.map_shape)
        self.visited = np.zeros(self.map_shape)
        self.visited_vis = np.zeros(self.map_shape)

        self.last_loc = None
        self.center_loc = [self.args.SEMANTIC_MAP.map_size_cm / 100.0 / 2.0,
                         self.args.SEMANTIC_MAP.map_size_cm / 100.0 / 2.0, 0.]
        
        self.last_action = None
        self.col_width = None
        self.selem = skimage.morphology.disk(3)


        # Calculating full and local map sizes
        self.local_w = int(self.full_w / self.args.SEMANTIC_MAP.global_downscaling)
        self.local_h = int(self.full_h / self.args.SEMANTIC_MAP.global_downscaling)

        # Initializing full and local map
        self.full_map = torch.zeros(self.num_env, self.nc, self.full_w, self.full_h).float().to(self.device)
        self.local_map = torch.zeros(self.num_env, self.nc, self.local_w,
                                self.local_h).float().to(self.device)

        # Initial full and local pose
        self.full_pose = torch.zeros(self.num_env, 3).float().to(self.device)
        self.local_pose = torch.zeros(self.num_env, 3).float().to(self.device)


        # Origin of local map, in meter
        self.origins = np.zeros((self.num_env, 3))

        # Local Map Boundaries, in centimeter.  
        # everything local is centimeter, everything global(full) is meter
        self.lmb = np.zeros((self.num_env, 4)).astype(int)

        # Planner pose inputs has 7 dimensions
        # 1-3 store continuous global agent location
        # 4-7 store local map boundaries
        self.planner_pose_inputs = np.zeros((self.num_env, 7))

        map_features_channels = 2 * MC.NON_SEM_CHANNELS + self.num_sem_categories
        if self.record_instance_ids:
            map_features_channels += self.num_sem_categories
        self.map_features = torch.zeros(
            self.num_env,
            map_features_channels,
            self.local_w,
            self.local_h,
            device=self.device,
        ).float()



        self.sem_map_module = SemanticMappingClean(self.instance_memory, device=self.device).to(self.device)
        self.sem_map_module.eval() 
        
        

    def reset(self,tgt):
        """Initialize agent state."""
        self.time_step=0
        self.llm_goal_start=999
        self.llm_last_goal_map=None
        self.last_sim_location=self.get_curr_pose()

        if self.instance_memory is not None:
            self.instance_memory.reset()
        self.init_map_and_pose()
        
        
        
        if self.args.visualize or self.args.save_images:
            self.vis_image = vu.init_vis_image(tgt, None)
        
        # from env parameters
        self.collision_map = np.zeros(self.map_shape)
        self.visited = np.zeros(self.map_shape)
        self.visited_vis = np.zeros(self.map_shape)
        # init loc in the center of the map
        self.curr_loc = self.center_loc
        self.last_action = None
        self.col_width = 1
        

    def reset_timestep(self) -> None:
        """Reset for a new sub-episode since pre-processing is temporally dependent."""
        self.time_step=0



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
        for e in range(self.num_env):
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

        for e in range(self.num_env):
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
        for e in range(self.num_env):
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
        
        question = f'is {tgt} in the near foreground of the image? yes or no'

        image=Image.fromarray(image,'RGB')
        answer=self.llm_speak(image,question)
        print(answer+'\n')

        if 'yes' in answer:
            return True
        else:
            return False
    

    
    def _plan(self, obstacle_map,goal,local_pose_and_lmb):
        """Function responsible for planning

        Args:
            
            'obstacle_map'  (ndarray): (M, M) map prediction
            'goal'      (ndarray): (M, M) goal locations
            'local_pose_and_lmb' (ndarray): (7,) array  denoting pose (x,y,o) in meter
                            and planning window (gx1, gx2, gy1, gy2) in centimeter

        Returns:
            action (str): action 
        """
        args = self.args
        self.last_loc = self.curr_loc

        # Get Map prediction
        obstacle_map = np.rint(obstacle_map)

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = local_pose_and_lmb

        planning_window = [gx1, gx2, gy1, gy2]
        planning_window=[int(num) for num in planning_window]
        gx1, gx2, gy1, gy2 = planning_window
        # Get curr loc
        self.curr_loc = [start_x, start_y, start_o]
        r, c = start_y, start_x
        start = [int(r * 100.0 / args.SEMANTIC_MAP.map_resolution - gx1),
                 int(c * 100.0 / args.SEMANTIC_MAP.map_resolution - gy1)]
        start = pu.threshold_poses(start, obstacle_map.shape)

        self.visited[gx1:gx2, gy1:gy2][start[0] - 0:start[0] + 1,
                                       start[1] - 0:start[1] + 1] = 1

        if args.visualize or args.save_images:
            # Get last loc
            last_start_x, last_start_y = self.last_loc[0], self.last_loc[1]
            r, c = last_start_y, last_start_x
            last_start = [int(r * 100.0 / args.SEMANTIC_MAP.map_resolution - gx1),
                          int(c * 100.0 / args.SEMANTIC_MAP.map_resolution - gy1)]
            last_start = pu.threshold_poses(last_start, obstacle_map.shape)
            self.visited_vis[gx1:gx2, gy1:gy2] = \
                vu.draw_line(last_start, start,
                             self.visited_vis[gx1:gx2, gy1:gy2])

        # Collision check
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

        stg, stop = self._get_stg(obstacle_map, start, np.copy(goal),
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
            action = 'turn_right' 
        elif relative_angle < -self.args.ENVIRONMENT.turn_angle / 2.:
            action = 'turn_left' 
        else:
            action = 'move_forward' 

        return action,stop,stg

    def _get_stg(self, obstacle_map, start, goal, planning_window):
        """Get short-term goal"""

        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = 0, 0
        x2, y2 = obstacle_map.shape

        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h + 2, w + 2)) + value
            new_mat[1:h + 1, 1:w + 1] = mat
            return new_mat

        expanded_obstacle=obstacle_map
        traversible =  (expanded_obstacle!= True).astype(int)
        traversible[self.collision_map[gx1:gx2, gy1:gy2]
                    [x1:x2, y1:y2] == 1] = 0
        traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1
        traversible[int(start[0]) - 1:int(start[0]) + 2,
                    int(start[1]) - 1:int(start[1]) + 2] = 1

        traversible = add_boundary(traversible)
        goal = add_boundary(goal, value=0)

        planner = FMMPlanner(traversible)
        
        selem = skimage.morphology.disk(NaviConfig.ENVIRONMENT.round_rate)
        goal = skimage.morphology.binary_dilation(
            goal, selem) != True
        goal = 1 - goal * 1.
        planner.set_multi_goal(goal)

        state = [start[0] + 1, start[1]  + 1]

        stg_x, stg_y, _, stop = planner.get_short_term_goal(state)

        stg_x, stg_y = stg_x  - 1, stg_y - 1

        return (stg_x, stg_y), stop

    def _visualize(self, exp_pred,pose_pred,
                   map_pred,sem_map_pred, goal_map,
                   stg=None, tgt='test'):
        args = self.args
        dump_dir = f"{args.DUMP_LOCATION}/hm3d/{tgt}"
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)
 
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = pose_pred
        # (start_x, start_y) (12.0, 12.0)
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
        # goal_mat = 1 - skimage.morphology.binary_dilation(
        #     goal, selem) != True

        # goal_mask = goal_mat == 1
        # sem_map[goal_mask] = 4
        
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
            img_save_path=dump_dir+f'/vis-{self.time_step}.png'
            cv2.imwrite(img_save_path, self.vis_image)

    def llm_speak(self,image,question):
        
        answer = self.llm.chat(
                image=image,
                msgs=[{'role': 'user', 'content': question}],
                tokenizer=self.tokenizer,
                sampling=True,
                temperature=0.7,
                system_prompt='You are now a robot in the home'
            )
            
        return answer.lower()
    
    def llm_speak_multi_img(self,images,question):
        content=images+[question]
        answer = self.llm.chat(
                image=None,
                msgs=[{'role': 'user', 'content': content}],
                tokenizer=self.tokenizer,
                sampling=True,
                temperature=0.7,
                system_prompt='You are now a robot in the home, you speak neatly.' # pass system_prompt if needed
            )
            
        return answer.lower()
   
    def is_stuck(self,xyos):
        # o in radian
        i=1
        total_dist=0
        for i in range(len(xyos)):
            dist = np.sqrt(np.sum(np.square(xyos[i-1][:2] - xyos[i][:2])))
            total_dist+=dist
        if total_dist<1 and abs(xyos[0][-1]-xyos[-1][-1])<10*np.pi/180:
            return True
        
    def vlm_direction(self,np_img,tgt,obs:Observations):
        llm_guide='initialized'
        if self.time_step==0:
            # single
            question = f'which direction should I go to find {tgt} based on the image? explore more, left, right or straight to the front?'
            image=Image.fromarray(np_img,'RGB')
            answer=self.llm_speak(image,question)
        else:
            # sequence
            direction_quest=f'''The first image is your current observation.
            In the second image, left is last step observation and right is the history trajectory topdown map. 
            Based on the observation and history map, to find {tgt}, 
            Which direction should you take? explore more, left, right, or go forward straight to the front? 
            speak only the option I give you'''
            current_obs=Image.fromarray(np_img,'RGB')
            history_map=self.vis_image
            answer=self.llm_speak_multi_img([current_obs,history_map],direction_quest)

        print(answer+'\n')
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
        elif 'go straight' in answer or 'go forward' in answer:
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

        return llm_guide


    def step(self, observes: Dict, tgt, timestep) -> str:
        print(f'lang tgt now is {tgt}')
        
        rgb_image = observes['color_sensor'][..., :3][..., ::-1] # (480 640 3)
        
        # to prevent stuck to the end
        stuck=False
        
        self.history_buffer.append(np.array(self.last_sim_location))
    
        if len(self.history_buffer)==6:
            self.history_buffer.pop(0)            
            stuck=self.is_stuck(self.history_buffer)
            
        if stuck:
            print("\033[1;36m TURN BACK DETECTED! \033[0m")
            self.history_buffer=[]
            self.local_map[:, MC.NON_SEM_CHANNELS+16,:,:]=0
            self.need_llm=True
            # clean all, including llm
            self.local_map[:, MC.NON_SEM_CHANNELS:MC.NON_SEM_CHANNELS+self.num_sem_categories,0,0]=0
            # just clean specific color.
            # tgt_id=self.find_corresponding_id(tgt)
            # self.local_map[:, MC.NON_SEM_CHANNELS+tgt_id,0,0]=0

            
        obs = Observations(
            gps=[0,0],  ### dummy
            compass=[0],  ### dummy
            rgb=rgb_image,
            depth=observes['depth_sensor'],
            camera_pose=None,
            third_person_image=None)

        obs = self.segmentation.predict(obs)
        
        self.update_vis(obs,observes['depth_sensor'])        
        possilbe_llm_stop=False
        

        if self.use_llm:
            np_img=np.uint8(rgb_image[:,:,[2,1,0]])
            print("\033[1;31m SHOULD STOP \033[0m")

            possilbe_llm_stop=self.should_stop(np_img,tgt)
            

            if timestep > self.llm_goal_start+NaviConfig.ENVIRONMENT.llm_endurance: # 5 for debug
                print("\033[1;36m give up former llm point! \033[0m")
                # cleaning
                self.local_map[:, MC.NON_SEM_CHANNELS+16,:,:]=0
                self.need_llm=True
                self.llm_goal_start=999

            if self.need_llm:
                print("\033[1;31m WHERE TO GO \033[0m")
                # go forward has trouble
                llm_guide=self.vlm_direction(np_img,tgt,obs)
            else:
                llm_guide='not used'
        else:
            # debug heuristic rendering
            obs.semantic=self.render(obs.semantic, 'foreground')
        

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
        detect_goal=False
        current_proj=self.local_map[0, 4:4+self.num_sem_categories, :,:].argmax(0)
        
        
        
        tgt_id=self.find_corresponding_id(tgt=tgt)
        if tgt_id==0:
            possible_detect_goal=self.local_map[0, 4, :,:]
        else:
            possible_detect_goal=(current_proj==(tgt_id)).int()
        
        if len(possible_detect_goal.unique())>1:
            print(f'tat_cat is {NaviConfig.cat2name.get(tgt_id)}')
            goal_map=possible_detect_goal.cpu().numpy()
            goal_map=self.cluster_goal_map(goal_map)
            self.need_llm=False
            detect_goal=True
            print('to detection target area!')
        
        else:
            if llm_guide=='explore more' :
                goal_map=self.frontier_goal_map()
                print('llm assigned frontier!')
            
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
                    print('llm target area!')
                else:
                    self.need_llm=True
                    if self.look_around_step<self.look_around_max_step:
                        print('fixed look around...')
                        self.look_around_step+=1
                        look_around=True
                    else:
                        goal_map=self.frontier_goal_map()
                        print('frontier the last!')
        
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
            action, stg_reached,stg = self._plan(obstacle_map,
                                            goal_map,
                                            self.planner_pose_inputs[e])
            print(f'\033[0;33;40mreached short term goal: {stg_reached}\033[0m')
            if stg_reached:
                self.look_around_step=0
                self.need_llm=True
                self.llm_goal_start = 999

                if detect_goal and possilbe_llm_stop:
                    print(f'this is detect goal is { detect_goal}, possilbe_llm_stop is {possilbe_llm_stop}')
                    action='terminate'
                elif detect_goal and not possilbe_llm_stop:
                    # chances are the detection goal is wrong
                    # clean the detection results.
                    self.local_map[e, MC.NON_SEM_CHANNELS+tgt_id, :, :]=0
                    # arbitary random go forward, left, right to skip this view point
                    action=random.choice(['move_forward', 'turn_left', 'turn_right'])

                    
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
                action, stg_reached,stg = self._plan(obstacle_map,
                                                    goal_map,
                                                    self.planner_pose_inputs[e],
                                                    )
                if stg_reached:
                    self.look_around_step=0
            
        
        if stuck:
            action='turn_back'
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
                )
        
        
        self.time_step = self.time_step + 1

        if action == 'terminate':
            print(f'find {tgt} task is done! spend {self.time_step} steps!')
            self.reset_timestep()
                

        print(f"======= total timestep: [{self.time_step}] =======")
        

        cv2.waitKey(10)
    
        return action

    
    def find_corresponding_id(self,tgt,rgb_image=None):
        tgt_cat='initialized'
        if NaviConfig.name2id.get(tgt) is not None:
            tgt_id=NaviConfig.name2id.get(tgt)
            tgt_cat=tgt
        else:
            # np_img=np.uint8(rgb_image[:,:,[2,1,0]])
            # image=Image.fromarray(np_img,'RGB')
            ans=self.llm_speak(None,
                            f'In {[name for name in NaviConfig.name2id.keys()]},\
                                what has the closest meaning to {tgt}, speak only the option')
            def find_closest_name(ans):
                for name in NaviConfig.name2id.keys():
                    if name in ans:
                        return name
            
            while NaviConfig.name2id.get(tgt_cat) is None:
                tgt_cat=find_closest_name(ans)
                print(f"closest name {ans} to {tgt}")
            tgt_id=NaviConfig.name2id.get(tgt_cat)
        
        return tgt_id
    
    def frontier_goal_map(self)->np.array:
        '''
            return ndarray
        '''
        self.map_features = self.sem_map_module._get_map_features(self.local_map, self.full_map).unsqueeze(1)
        map_features = self.map_features.flatten(0, 1)
        frontier_map = self.policy.get_frontier_map(map_features)
        squeezed=frontier_map.squeeze(0).squeeze(0)
        
        return squeezed.cpu().numpy()

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
    
    def expand(self,original,rate=NaviConfig.ENVIRONMENT.round_rate-6):
        selem = skimage.morphology.disk(rate)
        expanded = skimage.morphology.binary_dilation(
            original, selem)
        return expanded
    
    def downscale(self,obs_rgb,obs_depth,semantic,ds):
        # downscaling rgb, depth, and semantic
        rgb = torch.from_numpy(obs_rgb.copy()).to(self.device)
        ### Do not transform the depth
        depth = (torch.from_numpy(obs_depth).unsqueeze(-1).to(self.device)) * 100


        rgb = np.asarray(self.trans(rgb.cpu().numpy().astype(np.uint8)))
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
        # seg vis
        self.rgb_vis = obs.task_observations["semantic_frame"]
        # self.rgb_vis=obs.rgb
        

    
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
    
    # objects of all kind in hm3d
        
    choice=['sofa',
            'tv_monitor',
            'bed',
            'chair',
            'toilet',
            'plant',
            'table',
            'oven',
            'sink',
            'refrigerator',
            'book',
            'clock',
            'vase',
            'cup',
            'bottle',]
    goal=random.choice(choice)


    agent_traj=[]

    # reset
    agent = simulator.initialize_agent(sim_settings["default_agent"])
    # Set agent state
    agent_state = habitat_sim.AgentState()
    agent_traj.append(agent_state.position)
    
    nav_with_llm=NavWithLLM(simulator)

    max_step=400
    step=0
    stuck=0
    obs=simulator.reset()
    nav_with_llm.reset(goal)
    
    while True:
        action=nav_with_llm.step(obs,goal,
                                    timestep=step)
        if action=='terminate' or step==max_step:
            break

        obs=simulator.step(action)
        # record agent state
        agent_state = agent.get_state()
        now_pos=agent_state.position
        agent_traj.append(now_pos)
        step+=1
        

    
    cv2.destroyAllWindows()    
    simulator.close()
    exit()