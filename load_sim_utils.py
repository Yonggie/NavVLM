

import habitat_sim
import yaml
import os
from tqdm import tqdm
from navi_config import NaviConfig


def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.allow_sliding = True
    sim_cfg.scene_id = settings["scene"]
    # sim_cfg.scene_dataset_config_file = settings["scene_dataset"]
    sim_cfg.enable_physics = settings["enable_physics"]

    # Note: all sensors must have the same resolution
    sensor_specs = []

    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = [settings["height"], settings["width"]]
    color_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(color_sensor_spec)

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(depth_sensor_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=NaviConfig.ENVIRONMENT.forward_distance)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=NaviConfig.ENVIRONMENT.turn_angle)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=NaviConfig.ENVIRONMENT.turn_angle)
        ),
        # "terminate": habitat_sim.agent.ActionSpec(
        #     "terminate",habitat_sim.agent.
        # )
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])



sim_settings = {
    "width": NaviConfig.ENVIRONMENT.env_frame_width,  # Spatial resolution of the observations
    "height": NaviConfig.ENVIRONMENT.env_frame_height,
    "scene": -1,  # Scene path, need to change!
    "default_agent": 0,
    "sensor_height": NaviConfig.ENVIRONMENT.sensor_pose[1],  # Height of sensors in meters
    "color_sensor": True,  # RGB sensor
    "seed": 1,  # used in the random navigation
    "enable_physics": False,  # kinematics only
}