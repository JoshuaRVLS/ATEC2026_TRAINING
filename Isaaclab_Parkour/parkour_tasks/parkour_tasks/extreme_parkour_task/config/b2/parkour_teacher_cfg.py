from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass

from atec_rl_lab.assets.robots.b2 import UNITREE_B2_PIPER_CFG
from parkour_isaaclab.envs import ParkourManagerBasedRLEnvCfg
from parkour_isaaclab.terrains.extreme_parkour.config.parkour import EXTREME_PARKOUR_TERRAINS_CFG
from parkour_tasks.default_cfg import ParkourDefaultSceneCfg, VIEWER

from .parkour_mdp_cfg import (
    B2ActionsCfg,
    B2EventCfg,
    B2ProprioLidarObservationsCfg,
    B2ProprioLidarRewardsV2Cfg,
    B2ProprioLidarTerminationsV2Cfg,
    B2TeacherObservationsCfg,
    B2TeacherRewardsCfg,
    CommandsCfg,
    ParkourEventsCfg,
    TerminationsCfg,
)


@configclass
class B2ParkourTeacherSceneCfg(ParkourDefaultSceneCfg):
    robot: ArticulationCfg = UNITREE_B2_PIPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.375, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.15, size=[1.65, 1.5]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=2,
        track_air_time=True,
        debug_vis=False,
        force_threshold=1.0,
    )

    def __post_init__(self):
        self.terrain.terrain_generator = EXTREME_PARKOUR_TERRAINS_CFG


@configclass
class UnitreeB2TeacherParkourEnvCfg(ParkourManagerBasedRLEnvCfg):
    scene: B2ParkourTeacherSceneCfg = B2ParkourTeacherSceneCfg(num_envs=4096, env_spacing=1.0)
    observations: B2TeacherObservationsCfg = B2TeacherObservationsCfg()
    actions: B2ActionsCfg = B2ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: B2TeacherRewardsCfg = B2TeacherRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    parkours: ParkourEventsCfg = ParkourEventsCfg()
    events: B2EventCfg = B2EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**18
        self.scene.height_scanner.update_period = self.sim.dt * self.decimation
        self.scene.contact_forces.update_period = self.sim.dt * self.decimation
        self.scene.terrain.terrain_generator.curriculum = True
        self.actions.joint_pos.use_delay = False
        self.actions.joint_pos.history_length = 1
        self.events.random_camera_position = None


@configclass
class UnitreeB2TeacherParkourEnvCfg_EVAL(UnitreeB2TeacherParkourEnvCfg):
    viewer = VIEWER

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 256
        self.episode_length_s = 20.0
        self.parkours.base_parkour.debug_vis = True
        self.commands.base_velocity.debug_vis = True
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.random_difficulty = True
            self.scene.terrain.terrain_generator.difficulty_range = (0.0, 1.0)
        self.events.randomize_rigid_body_com = None
        self.events.randomize_rigid_body_mass = None
        self.events.push_by_setting_velocity.interval_range_s = (6.0, 6.0)
        self.commands.base_velocity.resampling_time_range = (60.0, 60.0)
        for key, sub_terrain in self.scene.terrain.terrain_generator.sub_terrains.items():
            if key in ["parkour", "parkour_hurdle", "parkour_step", "parkour_gap"]:
                sub_terrain.noise_range = (0.02, 0.02)
                sub_terrain.proportion = 0.25


@configclass
class UnitreeB2TeacherParkourEnvCfg_PLAY(UnitreeB2TeacherParkourEnvCfg_EVAL):
    viewer = VIEWER

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 60.0
        self.scene.num_envs = 16
        self.parkours.base_parkour.debug_vis = True
        self.commands.base_velocity.debug_vis = True
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.difficulty_range = (0.7, 1.0)
        self.events.push_by_setting_velocity = None
        for key, sub_terrain in self.scene.terrain.terrain_generator.sub_terrains.items():
            if key == "parkour_flat":
                sub_terrain.proportion = 0.0
            else:
                sub_terrain.proportion = 0.2
                sub_terrain.noise_range = (0.02, 0.02)


@configclass
class UnitreeB2ProprioLidarParkourEnvCfg(UnitreeB2TeacherParkourEnvCfg):
    observations: B2ProprioLidarObservationsCfg = B2ProprioLidarObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4096
        self.scene.height_scanner.pattern_cfg.resolution = 0.15
        self.scene.height_scanner.pattern_cfg.size = [2.4, 1.6]
        self.rewards.reward_collision.weight = -2.0
        self.rewards.reward_tracking_goal_vel.weight = 2.5
        self.rewards.reward_tracking_yaw.weight = 0.75
        self.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.5)
        self.commands.base_velocity.heading_control_stiffness = 1.0


@configclass
class UnitreeB2ProprioLidarParkourEnvCfg_EVAL(UnitreeB2ProprioLidarParkourEnvCfg):
    viewer = VIEWER

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 256
        self.parkours.base_parkour.debug_vis = True
        self.commands.base_velocity.debug_vis = True
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.random_difficulty = True
            self.scene.terrain.terrain_generator.difficulty_range = (0.0, 1.0)
        self.events.randomize_rigid_body_com = None
        self.events.randomize_rigid_body_mass = None
        self.events.push_by_setting_velocity.interval_range_s = (6.0, 6.0)
        self.commands.base_velocity.resampling_time_range = (60.0, 60.0)


@configclass
class UnitreeB2ProprioLidarParkourEnvCfg_PLAY(UnitreeB2ProprioLidarParkourEnvCfg_EVAL):
    viewer = VIEWER

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 60.0
        self.scene.num_envs = 16
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.difficulty_range = (0.3, 0.8)
        self.events.push_by_setting_velocity = None


@configclass
class UnitreeB2ProprioLidarParkourEnvCfg_V2(UnitreeB2ProprioLidarParkourEnvCfg):
    rewards: B2ProprioLidarRewardsV2Cfg = B2ProprioLidarRewardsV2Cfg()
    terminations: B2ProprioLidarTerminationsV2Cfg = B2ProprioLidarTerminationsV2Cfg()

    def __post_init__(self):
        super().__post_init__()
        self._apply_locomotion_v2_reward_profile()
        self.parkours.base_parkour.next_goal_threshold = 0.45
        self.parkours.base_parkour.curriculum_move_up_scale = 0.5
        self.parkours.base_parkour.curriculum_move_down_scale = 0.25
        self.parkours.base_parkour.curriculum_min_up_distance = 0.8
        self.commands.base_velocity.ranges.lin_vel_x = (0.20, 0.40)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.heading_control_stiffness = 0.8
        self.commands.base_velocity.resampling_time_range = (8.0, 8.0)
        self.events.push_by_setting_velocity = None
        self.events.randomize_rigid_body_com = None
        self.events.randomize_rigid_body_mass = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.max_init_terrain_level = 0
            self.scene.terrain.terrain_generator.random_difficulty = True
            self.scene.terrain.terrain_generator.difficulty_range = (0.0, 0.15)
            for key, sub_terrain in self.scene.terrain.terrain_generator.sub_terrains.items():
                if key == "parkour_flat":
                    sub_terrain.proportion = 1.0
                else:
                    sub_terrain.proportion = 0.0
                    sub_terrain.noise_range = (0.0, 0.01)

    def _apply_locomotion_v2_reward_profile(self):
        self.rewards.reward_collision.weight = -1.0
        self.rewards.reward_tracking_goal_vel.weight = 0.0
        self.rewards.reward_tracking_yaw.weight = 0.0
        self.rewards.reward_hip_pos.weight = 0.0
        self.rewards.reward_dof_error.weight = -1.0
        self.rewards.reward_ang_vel_xy.weight = -0.05
        self.rewards.reward_action_rate.weight = -0.01
        self.rewards.reward_lin_vel_z.weight = -2.0
        self.rewards.reward_orientation.weight = 0.0
        self.rewards.reward_feet_stumble.weight = -1.0
        self.rewards.reward_torques.weight = -1.0e-5
        self.rewards.reward_dof_acc.weight = -1.0e-7
        self.rewards.reward_delta_torques.weight = -1.0e-7
        self.rewards.reward_track_lin_vel_xy_exp.weight = 8.0
        self.rewards.reward_forward_displacement.weight = 0.0
        self.rewards.reward_no_forward_progress.weight = -8.0
        self.rewards.reward_track_forward_velocity.weight = 0.0
        self.rewards.reward_forward_velocity_positive.weight = 4.0
        self.rewards.reward_base_height.weight = 0.0
        self.rewards.reward_backward_velocity.weight = -2.0
        self.rewards.reward_joint_mirror.weight = -0.05
        self.rewards.reward_action_mirror.weight = 0.0
        self.rewards.reward_feet_height_body.weight = -5.0
        self.rewards.reward_feet_air_time.weight = 0.2
        self.rewards.reward_feet_contact_count.weight = -0.1
        self.rewards.reward_trot_gait.weight = 0.0
        self.rewards.reward_feet_slide.weight = 0.0
        self.rewards.reward_feet_contact_without_cmd.weight = 0.0
        self.rewards.reward_upward.weight = 0.0
        self.rewards.reward_upright_alive.weight = 0.0
        self.rewards.reward_fall_penalty.weight = 0.0
        self.rewards.reward_progress_to_goal.weight = 0.0
        self.rewards.reward_progress_from_start.weight = 0.0
        self.rewards.reward_goal_reached.weight = 0.0


@configclass
class UnitreeB2ProprioLidarParkourEnvCfg_V2_EVAL(UnitreeB2ProprioLidarParkourEnvCfg_V2):
    viewer = VIEWER

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 256
        self.parkours.base_parkour.debug_vis = True
        self.commands.base_velocity.debug_vis = True
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.random_difficulty = True
            self.scene.terrain.terrain_generator.difficulty_range = (0.0, 0.30)
        self.events.randomize_rigid_body_com = None
        self.events.randomize_rigid_body_mass = None
        if self.events.push_by_setting_velocity is not None:
            self.events.push_by_setting_velocity.interval_range_s = (8.0, 8.0)
        self.commands.base_velocity.resampling_time_range = (60.0, 60.0)


@configclass
class UnitreeB2ProprioLidarParkourEnvCfg_V2_PLAY(UnitreeB2ProprioLidarParkourEnvCfg_V2_EVAL):
    viewer = VIEWER

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 60.0
        self.scene.num_envs = 16
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.difficulty_range = (0.0, 0.30)
        self.events.push_by_setting_velocity = None
