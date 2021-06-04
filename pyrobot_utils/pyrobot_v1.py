#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os
from typing import Any

import numpy as np
import rospy
from gym import Space, spaces
from habitat.core.registry import registry
from habitat.core.simulator import (
    Config,
    SensorSuite,
    Simulator,
)
from habitat.core.utils import center_crop, try_cv2_import

cv2 = try_cv2_import()


def _locobot_base_action_space():
    return spaces.Dict(
        {
            "go_to_relative": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
            "go_to_absolute": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
        }
    )


def _locobot_camera_action_space():
    return spaces.Dict(
        {
            "set_pan": spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
            "set_tilt": spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
            "set_pan_tilt": spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
        }
    )


def _resize_observation(obs, observation_space, config):
    if obs.shape != observation_space.shape:
        if (
                config.CENTER_CROP is True
                and obs.shape[0] > observation_space.shape[0]
                and obs.shape[1] > observation_space.shape[1]
        ):
            obs = center_crop(obs, observation_space)

        else:
            obs = cv2.resize(
                obs, (observation_space.shape[1], observation_space.shape[0])
            )
    return obs


MM_IN_METER = 1000  # millimeters in a meter
ACTION_SPACES = {
    "LOCOBOT": {
        "BASE_ACTIONS": _locobot_base_action_space(),
        "CAMERA_ACTIONS": _locobot_camera_action_space(),
    }
}


@registry.register_simulator(name="PyRobot-v1")
class PyRobot(Simulator):
    r"""Simulator wrapper over PyRobot.

    PyRobot repo: https://github.com/facebookresearch/pyrobot
    To use this abstraction the user will have to setup PyRobot
    python3 version. Please refer to the PyRobot repository
    for setting it up. The user will also have to export a
    ROS_PATH environment variable to use this integration,
    please refer to :ref:`habitat.core.utils.try_cv2_import` for
    more details on this.

    This abstraction assumes that reality is a simulation
    (https://www.youtube.com/watch?v=tlTKTTt47WE).

    Args:
        config: configuration for initializing the PyRobot object.
    """

    def __init__(self, config: Config) -> None:
        self._config = config

        robot_sensors = []
        for sensor_name in self._config.SENSORS:
            sensor_cfg = getattr(self._config, sensor_name)
            sensor_type = registry.get_sensor(sensor_cfg.TYPE)

            assert sensor_type is not None, "invalid sensor type {}".format(
                sensor_cfg.TYPE
            )
            robot_sensors.append(sensor_type(sensor_cfg))
        self._sensor_suite = SensorSuite(robot_sensors)

        config_pyrobot = {
            "base_controller": self._config.BASE_CONTROLLER,
            "base_planner": self._config.BASE_PLANNER,
        }

        assert (
                self._config.ROBOT in self._config.ROBOTS
        ), "Invalid robot type {}".format(self._config.ROBOT)
        self._robot_config = getattr(self._config, self._config.ROBOT.upper())

        self._action_space = self._robot_action_space(
            self._config.ROBOT, self._robot_config
        )

        self._robot = Robot(
            self._config.ROBOT, base_config=config_pyrobot
        )

    def get_robot_observations(self):
        return {
            "rgb": self._robot.camera.get_rgb(),
            "depth": self._robot.camera.get_depth(),
            "bump": self._robot.base.base_state.bumper,
        }

    @property
    def sensor_suite(self) -> SensorSuite:
        return self._sensor_suite

    @property
    def base(self):
        return self._robot.base

    @property
    def camera(self):
        return self._robot.camera

    def _robot_action_space(self, robot_type, robot_config):
        action_spaces_dict = {}
        for action in robot_config.ACTIONS:
            action_spaces_dict[action] = ACTION_SPACES[robot_type.upper()][
                action
            ]
        return spaces.Dict(action_spaces_dict)

    @property
    def action_space(self) -> Space:
        return self._action_space

    def reset(self):
        self._robot.camera.reset()

        observations = self._sensor_suite.get_observations(
            robot_obs=self.get_robot_observations()
        )
        return observations

    def step(self, action, action_params):
        r"""Step in reality. Currently the supported
        actions are the ones defined in :ref:`_locobot_base_action_space`
        and :ref:`_locobot_camera_action_space`. For details on how
        to use these actions please refer to the documentation
        of namesake methods in PyRobot
        (https://github.com/facebookresearch/pyrobot).
        """
        if action in self._robot_config.BASE_ACTIONS:
            getattr(self._robot.base, action)(**action_params)
        elif action in self._robot_config.CAMERA_ACTIONS:
            getattr(self._robot.camera, action)(**action_params)
        else:
            raise ValueError("Invalid action {}".format(action))

        observations = self._sensor_suite.get_observations(
            robot_obs=self.get_robot_observations()
        )

        return observations

    def render(self, mode: str = "rgb") -> Any:
        observations = self._sensor_suite.get_observations(
            robot_obs=self.get_robot_observations()
        )

        output = observations.get(mode)
        assert output is not None, "mode {} sensor is not active".format(mode)

        return output

    def get_agent_state(
            self, agent_id: int = 0, base_state_type: str = "odom"
    ):
        assert agent_id == 0, "No support of multi agent in {} yet.".format(
            self.__class__.__name__
        )
        state = {
            "base": self._robot.base.get_state(base_state_type),
            "camera": self._robot.camera.get_state(),
        }
        return state

    def seed(self, seed: int) -> None:
        raise NotImplementedError("No support for seeding in reality")


class Robot:
    """
    This is the main interface class that is composed of
    key robot modules (base, arm, gripper, and camera).
    This class builds robot specific objects by reading a
    configuration and instantiating the necessary robot module objects.
    """

    def __init__(
            self,
            robot_name,
            use_arm=True,
            use_base=True,
            use_camera=True,
            use_gripper=True,
            arm_config={},
            base_config={},
            camera_config={},
            gripper_config={},
            common_config={},
    ):
        """
        Constructor for the Robot class
        :param robot_name: robot name
        :param use_arm: use arm or not
        :param use_base: use base or not
        :param use_camera: use camera or not
        :param use_gripper: use gripper or not
        :param arm_config: configurations for arm
        :param base_config: configurations for base
        :param camera_config: configurations for camera
        :param gripper_config: configurations for gripper
        :type robot_name: string
        :type use_arm: bool
        :type use_base: bool
        :type use_camera: bool
        :type use_gripper: bool
        :type arm_config: dict
        :type base_config: dict
        :type camera_config: dict
        :type gripper_config: dict
        """

        root_path = os.path.dirname(os.path.realpath(__file__))
        cfg_path = os.path.join(root_path, "cfg")
        robot_pool = []
        for f in os.listdir(cfg_path):
            if f.endswith("_config.py"):
                robot_pool.append(f[: -len("_config.py")])
        root_node = "pyrobot."
        self.configs = None
        this_robot = None
        for srobot in robot_pool:
            if srobot in robot_name:
                this_robot = srobot
                mod = importlib.import_module(
                    "pyrobot_utils." + "cfg." + "{:s}_config".format(srobot)
                )
                cfg_func = getattr(mod, "get_cfg")
                if srobot == "locobot" and "lite" in robot_name:
                    self.configs = cfg_func("create")
                else:
                    self.configs = cfg_func()
        if self.configs is None:
            raise ValueError(
                "Invalid robot name provided, only the following"
                " are currently available: {}".format(robot_pool)
            )
        self.configs.freeze()
        try:
            rospy.init_node("pyrobot_utils", anonymous=True)
        except rospy.exceptions.ROSException:
            rospy.logwarn("ROS node [pyrobot_utils] has already been initialized")

        root_node += this_robot
        root_node += "."
        if self.configs.HAS_COMMON:
            mod = importlib.import_module(root_node + self.configs.COMMON.NAME)
            common_class = getattr(mod, self.configs.COMMON.CLASS)
            setattr(
                self,
                self.configs.COMMON.NAME,
                common_class(self.configs, **common_config),
            )
        if self.configs.HAS_ARM and use_arm:
            mod = importlib.import_module(root_node + "arm")
            arm_class = getattr(mod, self.configs.ARM.CLASS)
            if self.configs.HAS_COMMON:
                arm_config[self.configs.COMMON.NAME] = getattr(
                    self, self.configs.COMMON.NAME
                )
            self.arm = arm_class(self.configs, **arm_config)
        if self.configs.HAS_BASE and use_base:
            mod = importlib.import_module(root_node + "base")
            base_class = getattr(mod, self.configs.BASE.CLASS)
            if self.configs.HAS_COMMON:
                base_config[self.configs.COMMON.NAME] = getattr(
                    self, self.configs.COMMON.NAME
                )
            self.base = base_class(self.configs, **base_config)
        if self.configs.HAS_CAMERA and use_camera:
            mod = importlib.import_module(root_node + "camera")
            camera_class = getattr(mod, self.configs.CAMERA.CLASS)
            if self.configs.HAS_COMMON:
                camera_config[self.configs.COMMON.NAME] = getattr(
                    self, self.configs.COMMON.NAME
                )
            self.camera = camera_class(self.configs, **camera_config)
        if self.configs.HAS_GRIPPER and use_gripper and use_arm:
            mod = importlib.import_module(root_node + "gripper")
            gripper_class = getattr(mod, self.configs.GRIPPER.CLASS)
            if self.configs.HAS_COMMON:
                gripper_config[self.configs.COMMON.NAME] = getattr(
                    self, self.configs.COMMON.NAME
                )
            self.gripper = gripper_class(self.configs, **gripper_config)

        # sleep some time for tf listeners in subclasses
        rospy.sleep(2)
