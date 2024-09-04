
import sys
# from isaacgym import gymapi
# from isaacgym import gymutil
from omni.isaac.core import SimulationContext
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.utils.extensions import enable_extension
import omni.kit

import numpy as np
import torch

# Base class for RL tasks
class BaseTask():

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.sim_context = SimulationContext(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0, backend="physx",
                                        device="cuda:0")
        self.scene = Scene()
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = parse_device_str(self.sim_device)
        self.headless = headless

        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True, otherwise returned tensors are copied to CPU by physX.
        if sim_device_type=='cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'

        # graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self.sim_device_id
        if self.headless == True:
            self.graphics_device_id = -1

        self.num_envs = cfg.env.num_envs
        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float)
        else: 
            self.privileged_obs_buf = None
            # self.num_privileged_obs = self.num_obs

        self.extras = {}
        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if not self.headless:
            # Enable the necessary extensions for keyboard control and viewer
            enable_extension("omni.isaac.keyboard")

            # Set up the camera (example of setting camera properties)
            set_camera_view("/OmniverseKit_Persp", position=[3, 3, 2], target=[0, 0, 0])
            # Set up keyboard shortcuts
            omni.kit.commands.execute("IsaacSimMenu.BindKeyEvent", key="Escape", action="Quit")
            omni.kit.commands.execute("IsaacSimMenu.BindKeyEvent", key="V", action="Toggle Viewer Sync")


    @staticmethod
    def parse_device_str(device_str):
        device = torch.device(device_str)
        device_type = device.type
        device_id = device.index if device.index is not None else 0
        return device_type, device_id
    def get_observations(self):
        return self.obs_buf
    
    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset_idx(self, env_ids):
        """Reset selected robots"""
        raise NotImplementedError

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def step(self, actions):
        raise NotImplementedError

    def render(self, sync_frame_time=True):
        if not omni.kit.app.get_app().is_running():
            sys.exit()

        # Check for keyboard events
        input_map = omni.kit.app.get_app().get_extension_manager().get_extension("omni.kit.input.map")
        input_events = input_map.get_input_event_queue()
        for event in input_events:
            if event.input == "Escape" and event.value == 1.0:
                sys.exit()
            elif event.input == "V" and event.value == 1.0:
                self.enable_viewer_sync = not self.enable_viewer_sync

        # Step simulation and render
        self.sim_context.step(render=self.enable_viewer_sync)

        if sync_frame_time:
            self.sim_context.render(sync_frame_time=True)