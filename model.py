# The code is adapted from 
# https://github.com/devendrachaplot/Object-Goal-Navigation/blob/master/model.py &
# https://github.com/facebookresearch/home-robot/blob/goat_v0/src/home_robot/home_robot/mapping/semantic/categorical_2d_semantic_map_module.py

import cv2
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from collections import defaultdict
import skimage.morphology
from torch import IntTensor, Tensor
from navi_config import NaviConfig
from utils.model import get_grid, ChannelPool
import utils.depth_utils as du

import torch
import numpy as np

def get_transform_matrices(pose:torch.Tensor):
    """
    Input:
        `pose` FloatTensor(bs, 3)
    Output:
        `rot_matrix` FloatTensor(bs, 2, 3)
        `trans_matrix` FloatTensor(bs, 2, 3)

    """
    pose = pose.float()
    x = pose[:, 0]
    y = pose[:, 1]
    t = pose[:, 2]

    bs = x.size(0)
    t = t * np.pi / 180.
    cos_t = t.cos()
    sin_t = t.sin()

    # 构建旋转矩阵
    rot_matrix = torch.tensor([
        [cos_t, -sin_t],
        [sin_t, cos_t],
    ])
    

    # 构建平移矩阵
    trans_matrix = torch.tensor([
        [x],
        [y],
    ])
    
    
    return rot_matrix, trans_matrix

def relative_pose(R1, t1, R2, t2):
    '''
    # R1,R2 (2,2) 
    # t1,t2 (2,1)
    '''

    # 计算相对旋转矩阵
    R_rel = R2 @ R1.T    # 2,2

    # 计算相对平移向量
    # t_rel = R1.T @ (t2 - t1) # 2, 1
    t_rel=(t2-t1)/2
    # t_rel=(t1-t2)

    return R_rel, t_rel
    



class SemanticMappingClean(nn.Module):

    """
    Semantic_Mapping
    """

    def __init__(
            self, 
            instance_memory, 
            device,
            num_sem_categories=NaviConfig.SEMANTIC_MAP.num_sem_categories):
        super(SemanticMappingClean, self).__init__()

        self.device = device
        
        self.num_sem_categories = num_sem_categories
        
        self.screen_h = NaviConfig.SEMANTIC_MAP.frame_height
        self.screen_w = NaviConfig.SEMANTIC_MAP.frame_width
        self.resolution = NaviConfig.SEMANTIC_MAP.map_resolution
        self.z_resolution = NaviConfig.SEMANTIC_MAP.map_resolution
        self.global_downscaling = NaviConfig.SEMANTIC_MAP.global_downscaling
        self.map_size_cm = NaviConfig.SEMANTIC_MAP.map_size_cm // self.global_downscaling
        self.n_channels = 3
        self.vision_range = NaviConfig.SEMANTIC_MAP.vision_range
        self.dropout = 0.5
        self.fov = NaviConfig.SEMANTIC_MAP.hfov
        self.du_scale = NaviConfig.SEMANTIC_MAP.du_scale
        self.cat_pred_threshold = NaviConfig.SEMANTIC_MAP.cat_pred_threshold
        self.exp_pred_threshold = NaviConfig.SEMANTIC_MAP.explored_map_threshold
        self.map_pred_threshold = NaviConfig.SEMANTIC_MAP.obstacle_map_threshold

        self.max_height = int(360 / self.z_resolution)
        self.min_height = int(-40 / self.z_resolution)
        self.agent_height = NaviConfig.SEMANTIC_MAP.camera_height * 100.+ 30 # m to cm 
        self.shift_loc = [self.vision_range *
                          self.resolution // 2, 0, np.pi / 2.0]
        self.camera_matrix = du.get_camera_matrix(
            self.screen_w, self.screen_h, self.fov)

        self.pool = ChannelPool(1)

        vr = self.vision_range

        self.init_grid = torch.zeros(
            NaviConfig.SEMANTIC_MAP.num_processes, 1 + self.num_sem_categories, vr, vr,
            self.max_height - self.min_height
        ).float().to(self.device)
        self.feat = torch.ones(
            NaviConfig.SEMANTIC_MAP.num_processes, 1 + self.num_sem_categories,
            self.screen_h // self.du_scale * self.screen_w // self.du_scale
        ).float().to(self.device)

        self.instance_memory = instance_memory
        self.record_instance_ids = NaviConfig.SEMANTIC_MAP.record_instance_ids
        self.NON_SEM_CHANNELS = 4

        self.padding_for_instance_overlap = 1
        self.dilation_for_instances = 0
        

    def forward(self, batch_size, rgb, depth, semantic,instances,
                pose_change, maps_last, poses_last, origins, lmb, 
                former_llm_goal_map=None):
        '''
            return
            fp_map_pred, map_pred, pose_last, current_poses
        '''
        # note channel are put in the last for ALL of the vars 
        depth=depth.permute(2,0,1)
        rgb=rgb.permute(2,0,1)
        semantic=semantic.permute(2,0,1)
        instances=instances.permute(2,0,1)


        h, w= depth.shape[-2:]
        c=rgb.shape[0]+depth.shape[0]\
            +semantic.shape[0]+instances.shape[0]
        bs=batch_size
        

        point_cloud_t = du.get_point_cloud_from_z_t(
            depth, self.camera_matrix, self.device, scale=self.du_scale)

        agent_view_t = du.transform_camera_view_t(
            point_cloud_t, self.agent_height, 0, self.device)

        agent_view_centered_t = du.transform_pose_t(
            agent_view_t, self.shift_loc, self.device)
        
        voxel_channels = 1 + self.num_sem_categories
        num_instance_channels = c - 4 - self.num_sem_categories
        voxel_channels += num_instance_channels
        # print('voxel channels:', voxel_channels)
        init_grid = torch.zeros(
            bs,
            voxel_channels,
            self.vision_range,
            self.vision_range,
            self.max_height - self.min_height,
            dtype=torch.float32,
        ).to(self.device)

        feat = torch.ones(
            bs,
            voxel_channels,
            self.screen_h // self.du_scale * self.screen_w // self.du_scale,
            dtype=torch.float32,
        ).to(self.device)
       
        # print('[model] maps_last shape', maps_last.shape)

        max_h = self.max_height
        min_h = self.min_height
        xy_resolution = self.resolution
        z_resolution = self.z_resolution
        vision_range = self.vision_range
        XYZ_cm_std = agent_view_centered_t.float()
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] / xy_resolution)
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] -
                               vision_range // 2.) / vision_range * 2.
        XYZ_cm_std[..., 2] = XYZ_cm_std[..., 2] / z_resolution
        XYZ_cm_std[..., 2] = (XYZ_cm_std[..., 2] -
                              (max_h + min_h) // 2.) / (max_h - min_h) * 2.
        
        feat_input = torch.cat([semantic, instances], dim=0)
        feat[:, 1:, :] = nn.AvgPool2d(self.du_scale)(
            feat_input.unsqueeze(0)
        ).view(bs, c - 4, h // self.du_scale * w // self.du_scale)
        

        XYZ_cm_std = XYZ_cm_std.permute(0, 3, 1, 2)
        XYZ_cm_std = XYZ_cm_std.view(XYZ_cm_std.shape[0],
                                     XYZ_cm_std.shape[1],
                                     XYZ_cm_std.shape[2] * XYZ_cm_std.shape[3])

        voxels = du.splat_feat_nd(
            init_grid, feat, XYZ_cm_std).transpose(2, 3)
        
        min_z = int(25 / z_resolution - min_h)
        max_z = int((self.agent_height + 1) / z_resolution - min_h)+10
        # print(min_z,max_z)
        agent_height_proj = voxels[..., min_z:max_z].sum(4)
        all_height_proj = voxels.sum(4)

        fp_map_pred = agent_height_proj[:, 0:1, :, :]
        fp_exp_pred = all_height_proj[:, 0:1, :, :]
        fp_map_pred = fp_map_pred / self.map_pred_threshold
        fp_exp_pred = fp_exp_pred / self.exp_pred_threshold
        fp_map_pred = torch.clamp(fp_map_pred, min=0.0, max=1.0)
        fp_exp_pred = torch.clamp(fp_exp_pred, min=0.0, max=1.0)

        

        # "c" is equal to the number of map channels
        # agent view is the new local map
        agent_view = torch.zeros(bs, c,
                                 self.map_size_cm // self.resolution,
                                 self.map_size_cm // self.resolution
                                 ).to(self.device)

        x1 = self.map_size_cm // (self.resolution * 2) - self.vision_range // 2
        x2 = x1 + self.vision_range
        y1 = self.map_size_cm // (self.resolution * 2)
        y2 = y1 + self.vision_range
        agent_view[:, 0:1, y1:y2, x1:x2] = fp_map_pred
        agent_view[:, 1:2, y1:y2, x1:x2] = fp_exp_pred
        agent_view[:, 4:, y1:y2, x1:x2] = torch.clamp(
            agent_height_proj[:, 1:, :, :] / self.cat_pred_threshold,
            min=0.0, max=1.0)


        def get_new_pose_batch(pose, rel_pose_change):
            next_pose=pose.clone()
            # y
            next_pose[:, 1] += rel_pose_change[:, 0] * \
                torch.sin(next_pose[:, 2] / 57.29577951308232) \
                + rel_pose_change[:, 1] * \
                torch.cos(next_pose[:, 2] / 57.29577951308232)
            # x
            next_pose[:, 0] += rel_pose_change[:, 0] * \
                torch.cos(next_pose[:, 2] / 57.29577951308232) \
                - rel_pose_change[:, 1] * \
                torch.sin(next_pose[:, 2] / 57.29577951308232)
            # orientation
            next_pose[:, 2] += rel_pose_change[:, 2] * 57.29577951308232

            next_pose[:, 2] = torch.fmod(next_pose[:, 2] - 180.0, 360.0) + 180.0
            next_pose[:, 2] = torch.fmod(next_pose[:, 2] + 180.0, 360.0) - 180.0

            return next_pose

        next_poses = get_new_pose_batch(poses_last, pose_change)

        # Process instances
        
        semantic_channels= semantic.unsqueeze(0)
        instance_channels = instances.unsqueeze(0)
        
        if num_instance_channels > 0:
            # print('processing instances...')
            self.instance_memory.process_instances(
                semantic_channels,
                instance_channels,
                point_cloud_t,
                torch.concat([next_poses + origins, lmb], axis=1)
                .cpu()
                .float(),  # store the global pose
                image=rgb.unsqueeze(0),
            )

        st_pose = next_poses.clone().detach()

        st_pose[:, :2] = - (st_pose[:, :2]
                            * 100.0 / self.resolution
                            - self.map_size_cm // (self.resolution * 2)) /\
            (self.map_size_cm // (self.resolution * 2))
        st_pose[:, 2] = 90. - (st_pose[:, 2])

        rot_mat, trans_mat = get_grid(st_pose, agent_view.size(),
                                      self.device)

        rotated = F.grid_sample(agent_view, rot_mat, align_corners=True)
        translated = F.grid_sample(rotated, trans_mat, align_corners=True)

        if former_llm_goal_map is not None:
            origin_goal_map = np.copy(former_llm_goal_map)
            # former_llm_goal_map (240,240) ndarray
            # poses_last (1,7)  x,y,t, ...
            # next_poses (1,7)  x,y,t, ...
            R1,t1=get_transform_matrices(poses_last)
            R2,t2=get_transform_matrices(next_poses)
            # R1,R2 (1, 2,2) 
            # t1,t2 (1, 2,1)
            
            R1,R2,t1,t2=R1.squeeze(0),R2.squeeze(0),t1.squeeze(0),t2.squeeze(0)
            R_rel,t_rel = relative_pose(R1,t1,R2,t2)
            
            # yesterday
            # # R_rel  (2,2)
            # # t_rel  (2,3)
            # R_rel=torch.cat([R_rel,torch.zeros(2,1)],dim=1)
            # # R_rel (2,3)
            
            # today R (2,2) t (2,1)

            # T=torch.tensor([
            #     [R_rel[0,0],R_rel[0,1],t_rel[0,0]],
            #     [R_rel[1,0],R_rel[1,1],t_rel[1,0]],
            #     # [0,         0,         1]
            # ])

            T=torch.tensor([
                [1,0,t_rel[0,0]],
                [0,1,t_rel[1,0]],
                # [0,         0,         1]
            ])

            rel_T=F.affine_grid(T.unsqueeze(0),(1,1,240,240),align_corners=True)
            rel_T=rel_T.to(self.device)

            
            former_llm_goal_map = torch.from_numpy(former_llm_goal_map).float().to(self.device)
            former_llm_goal_map = former_llm_goal_map.unsqueeze(0).unsqueeze(0)
            

            former_llm_goal_map = F.grid_sample(former_llm_goal_map, rel_T, align_corners=True)

            former_llm_goal_map = former_llm_goal_map.squeeze(0).squeeze(0)
            former_llm_goal_map = former_llm_goal_map.cpu().numpy()
            cv2.imwrite('goal_map.png',np.flipud(origin_goal_map*200))
            
            cv2.imwrite('now.png',np.flipud(former_llm_goal_map*200))
                
    

        ### TODO: aggregate instance channels for translated
        # translated -> (1, 4 + 16 + num_ins, h, w)

        translated = self._aggregate_instance_map_channels_per_category(
            translated, num_instance_channels
        )
        
        maps2 = torch.cat((maps_last.unsqueeze(1), translated.unsqueeze(1)), 1)

        
        local_map, _ = torch.max(maps2, 1)

        if self.record_instance_ids:
            # overwrite channels containing instance IDs
            local_map[:, self.NON_SEM_CHANNELS + self.num_sem_categories : self.NON_SEM_CHANNELS + 2 * self.num_sem_categories,]\
                  = translated[
                :,
                self.NON_SEM_CHANNELS
                + self.num_sem_categories : self.NON_SEM_CHANNELS
                + 2 * self.num_sem_categories,
            ]
        

        return fp_map_pred, local_map,\
              poses_last, next_poses,\
                former_llm_goal_map
    
    def _aggregate_instance_map_channels_per_category(
        self, curr_map, num_instance_channels
    ):
        """Aggregate map channels for instances (input: one binary channel per instance in [0, 1])
        by category (output: one channel per category containing instance IDs)."""

        # first extract instance channels
        # TODO: NON_SEM_CHANNELS
        top_down_instance_one_hot = curr_map[ 
            :,
            (self.NON_SEM_CHANNELS + self.num_sem_categories) : (
                self.NON_SEM_CHANNELS + self.num_sem_categories + num_instance_channels
            ),
            :,
            :,
        ]
        # now we convert the top down instance map to get a map for storing instances per channel
        top_down_instances_per_category = torch.zeros(
            curr_map.shape[0],
            self.num_sem_categories,
            curr_map.shape[2],
            curr_map.shape[3],
            device=curr_map.device,
            dtype=curr_map.dtype,
        )

        if num_instance_channels > 0:
            # loop over envs
            # TODO Can we vectorize this across envs? (Only needed if we use multiple envs)
            for i in range(top_down_instance_one_hot.shape[0]):
                # create category id to instance id list mapping
                category_id_to_instance_id_list = defaultdict(list)
                # retrieve unprocessed instances
                unprocessed_instances = (
                    self.instance_memory.get_unprocessed_instances_per_env(i)
                )
                # loop over unprocessed instances
                for instance_id, instance in unprocessed_instances.items():
                    category_id_to_instance_id_list[instance.category_id].append(
                        instance_id
                    )

                # loop over categories
                # TODO Can we vectorize this across categories? (Only needed if speed bottleneck)
                for category_id in category_id_to_instance_id_list.keys():
                    if len(category_id_to_instance_id_list[category_id]) == 0:
                        continue
                    # get the instance ids for this category
                    instance_ids = category_id_to_instance_id_list[category_id]
                    # create a tensor by slicing top_down_instance_one_hot using the instance ids
                    instance_one_hot = top_down_instance_one_hot[i, instance_ids]
                    # add a channel with all values equal to 1e-5 as the first channel
                    instance_one_hot = torch.cat(
                        (
                            1e-5 * torch.ones_like(instance_one_hot[:1]),
                            instance_one_hot,
                        ),
                        dim=0,
                    )
                    # get the instance id map using argmax
                    instance_id_map = instance_one_hot.argmax(dim=0)
                    # add a zero to start of instance ids
                    instance_id = [0] + instance_ids
                    # update the ids using the list of instance ids
                    instance_id_map = torch.tensor(
                        instance_id, device=instance_id_map.device
                    )[instance_id_map]
                    # update the per category instance map
                    top_down_instances_per_category[i, category_id] = instance_id_map
        # TODO: NON_SEM_CHANNELS
        curr_map = torch.cat(
            (
                curr_map[:, : self.NON_SEM_CHANNELS + self.num_sem_categories],
                top_down_instances_per_category,
                curr_map[
                    :,
                    self.NON_SEM_CHANNELS
                    + self.num_sem_categories
                    + num_instance_channels :,
                ],
            ),
            dim=1,
        )

        return curr_map
    
    def _update_global_map_instances_for_one_channel(
        self,
        env_id: int,
        global_instances: Tensor,
        local_map: Tensor,
        x_range: tuple,
        y_range: tuple,
        max_instance_id: int,
    ) -> Tensor:
        """
        Update one instance channels in the global map from one instance channels in the local map:
        aggregate local instances with existing global instances or create new global instances.

        Args:
            global_instances (Tensor): The global map tensor.
            local_map (Tensor): The local map tensor.
            x_range (tuple): The range of indices in the x-axis for the local map in the global map.
            y_range (tuple): The range of indices in the y-axis for the local map in the global map.

        Returns:
            Tensor: The updated global instances tensor.

        """
        p = self.padding_for_instance_overlap  # default: 1
        d = self.dilation_for_instances  # default: 0

        H = global_instances.shape[0]
        W = global_instances.shape[1]

        x1, x2 = x_range
        y1, y2 = y_range

        # padding added on each side
        t_p = min(x1, p)
        b_p = min(H - x2, p)
        l_p = min(y1, p)
        r_p = min(W - y2, p)

        # the indices of the padded local_map in the global map
        x_start = x1 - t_p
        x_end = x2 + b_p
        y_start = y1 - l_p
        y_end = y2 + r_p

        local_map = torch.round(local_map)

        # pad the local map
        ### TODO: make it more clean by combining the two "F.pad"
        extended_local_map = F.pad(local_map.float(), (l_p, r_p), mode="replicate")
        extended_local_map = F.pad(
            extended_local_map.transpose(1, 0), (t_p, b_p), mode="replicate"
        ).transpose(1, 0)

        self.instance_dilation_selem = skimage.morphology.disk(d)
        # dilate the extended local map
        if d > 0:
            extended_dilated_local_map = torch.round(
                torch.tensor(
                    cv2.dilate(
                        extended_local_map.cpu().numpy(),
                        self.instance_dilation_selem,
                        iterations=1,
                    ),
                    device=local_map.device,
                    dtype=local_map.dtype,
                )
            )
        else:
            extended_dilated_local_map = torch.clone(extended_local_map)
        # Get the instances from the global map within the local map's region
        global_instances_within_local = global_instances[x_start:x_end, y_start:y_end]

        instance_mapping = self._get_local_to_global_instance_mapping(
            env_id,
            extended_dilated_local_map,
            global_instances_within_local,
            max_instance_id,
            torch.unique(extended_local_map),
        )
        # print('[model] instance_mapping:', instance_mapping)

        # Update the global map with the associated instances from the local map
        global_instances_in_local = np.vectorize(instance_mapping.get)(
            local_map.cpu().numpy()
        )
        global_instances[x1:x2, y1:y2] = torch.maximum(
            global_instances[x1:x2, y1:y2],
            torch.tensor(
                global_instances_in_local,
                dtype=torch.int64,
                device=global_instances.device,
            ),
        )
        return global_instances

    def _get_local_to_global_instance_mapping(
        self,
        env_id: int,
        extended_local_labels: Tensor,
        global_instances_within_local: Tensor,
        max_instance_id: int,
        local_instance_ids: Tensor,
    ) -> dict:
        """
        Creates a mapping of local instance IDs to global instance IDs.

        Args:
            extended_local_labels: Labels of instances in the extended local map.
            global_instances_within_local: Instances from the global map within the local map's region.

        Returns:
            A mapping of local instance IDs to global instance IDs.
        """
        instance_mapping = {}

        # Associate instances in the local map with corresponding instances in the global map
        for local_instance_id in local_instance_ids:
            if local_instance_id == 0:
                # ignore 0 as it does not correspond to an instance
                continue
            # pixels corresponding to
            local_instance_pixels = extended_local_labels == local_instance_id

            # Check for overlapping instances in the global map
            overlapping_instances = global_instances_within_local[local_instance_pixels]
            unique_overlapping_instances = torch.unique(overlapping_instances)

            unique_overlapping_instances = unique_overlapping_instances[
                unique_overlapping_instances != 0
            ]
            if len(unique_overlapping_instances) >= 1:
                # If there is a corresponding instance in the global map, pick the first one and associate it
                global_instance_id = int(unique_overlapping_instances[0].item())
                instance_mapping[local_instance_id.item()] = global_instance_id
            else:
                # If there are no corresponding instances, create a new instance
                global_instance_id = max_instance_id + 1
                instance_mapping[local_instance_id.item()] = global_instance_id
                max_instance_id += 1
            # update the id in instance memory
            self.instance_memory.update_instance_id(
                env_id, int(local_instance_id.item()), global_instance_id
            )
        instance_mapping[0.0] = 0
        return instance_mapping

    def _update_global_map_instances(
        self, e: int, global_map: Tensor, local_map: Tensor, lmb: Tensor
    ) -> Tensor:
        """
        Update instance channels in the global map from instance channels in the local map:
        aggregate local instances with existing global instances or create new global instances.

        Args:
            e (int): The index of the environment.
            global_map (Tensor): The global map tensor.
            local_map (Tensor): The local map tensor.
            lmb (Tensor): The tensor containing the ranges of indices for the local map in the global map.

        Returns:
            Tensor: The updated global map tensor.
        """
        # TODO Can we vectorize this across categories? (Only needed if speed bottleneck)
        for i in range(self.num_sem_categories):
            # print('[model] _update_global_map_instances')
            if (
                torch.sum(
                    local_map[e, self.NON_SEM_CHANNELS + i + self.num_sem_categories]
                )
                > 0
            ):
                max_instance_id = ( # the max instance id stored in the prev global map
                    torch.max(
                        global_map[
                            e,
                            self.NON_SEM_CHANNELS
                            + self.num_sem_categories : self.NON_SEM_CHANNELS
                            + 2 * self.num_sem_categories,
                        ]
                    )
                    .int()
                    .item()
                )
                # print('[model] category {} found'.format(i))
                # print('[model] max_instance_id', max_instance_id)
                # if the local map has any object instances, update the global map with instance ids
                instances = self._update_global_map_instances_for_one_channel(
                    e,
                    global_map[e, self.NON_SEM_CHANNELS + i + self.num_sem_categories],
                    local_map[e, self.NON_SEM_CHANNELS + i + self.num_sem_categories],
                    (lmb[e, 0], lmb[e, 1]),
                    (lmb[e, 2], lmb[e, 3]),
                    max_instance_id,
                )
                global_map[
                    e, i + self.NON_SEM_CHANNELS + self.num_sem_categories
                ] = instances

        return global_map

    def _update_global_map_and_pose_for_env(
        self,
        e: int,
        local_map: Tensor,
        global_map: Tensor,
        local_pose: Tensor,
        global_pose: Tensor,
        lmb: Tensor,
        origins: Tensor,
    ):
        """Update global map and pose and re-center local map and pose for a
        particular environment.
        """

        if self.record_instance_ids:
            global_map = self._update_global_map_instances(
                e, global_map, local_map, lmb
            )
            global_map[
                e,
                : self.NON_SEM_CHANNELS + self.num_sem_categories,
                lmb[e, 0] : lmb[e, 1],
                lmb[e, 2] : lmb[e, 3],
            ] = local_map[e, : self.NON_SEM_CHANNELS + self.num_sem_categories]
            global_map[
                e,
                self.NON_SEM_CHANNELS + 2 * self.num_sem_categories :,
                lmb[e, 0] : lmb[e, 1],
                lmb[e, 2] : lmb[e, 3],
            ] = local_map[e, self.NON_SEM_CHANNELS + 2 * self.num_sem_categories :]
        else:
            global_map[e, :, lmb[e, 0] : lmb[e, 1], lmb[e, 2] : lmb[e, 3]] = local_map[
                e
            ]


    
    def _get_map_features(self, local_map: Tensor, global_map: Tensor) -> Tensor:
        """Get global and local map features.

        Arguments:
            local_map: local map of shape
             (batch_size, MC.NON_SEM_CHANNELS + num_sem_categories, M, M)
            global_map: global map of shape
             (batch_size, MC.NON_SEM_CHANNELS + num_sem_categories, M * ds, M * ds)

        Returns:
            map_features: semantic map features of shape
             (batch_size, 2 * MC.NON_SEM_CHANNELS + num_sem_categories, M, M)
        """
        map_features_channels = 2 * self.NON_SEM_CHANNELS + self.num_sem_categories

        if self.record_instance_ids:
            map_features_channels += self.num_sem_categories
        # if self.evaluate_instance_tracking:
        #     map_features_channels += self.max_instances + 1

        assert self.map_size_cm // self.resolution == local_map.shape[2]
        map_features = torch.zeros(
            local_map.size(0),
            map_features_channels,
            self.map_size_cm // self.resolution,
            self.map_size_cm // self.resolution,
            device=local_map.device,
            dtype=local_map.dtype,
        )

        # Local obstacles, explored area, and current and past position
        map_features[:, 0 : self.NON_SEM_CHANNELS, :, :] = local_map[
            :, 0 : self.NON_SEM_CHANNELS, :, :
        ]
        # Global obstacles, explored area, and current and past position
        map_features[
            :, self.NON_SEM_CHANNELS : 2 * self.NON_SEM_CHANNELS, :, :
        ] = nn.MaxPool2d(self.global_downscaling)(
            global_map[:, 0 : self.NON_SEM_CHANNELS, :, :]
        )
        # Local semantic categories
        map_features[:, 2 * self.NON_SEM_CHANNELS :, :, :] = local_map[
            :, self.NON_SEM_CHANNELS :, :, :
        ]

        # if debug_maps:
        #     plt.subplot(131)
        #     plt.imshow(local_map[0, 7])  # second object = cup
        #     plt.subplot(132)
        #     plt.imshow(local_map[0, 6])  # first object = chair
        #     # This is the channel in MAP FEATURES mode
        #     plt.subplot(133)
        #     plt.imshow(map_features[0, 12])
        #     plt.show()

        return map_features.detach()

