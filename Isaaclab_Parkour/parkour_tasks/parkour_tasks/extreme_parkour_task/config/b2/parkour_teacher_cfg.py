from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass

from atec_rl_lab.assets.robots.b2 import UNITREE_B2_CFG
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
    robot: ArticulationCfg = UNITREE_B2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
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
        self.scene.height_scanner.pattern_cfg.size = [1.65, 1.5]
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
        self.parkours.base_parkour.next_goal_threshold = 0.35
        self.parkours.base_parkour.curriculum_move_up_scale = 0.35
        self.parkours.base_parkour.curriculum_move_down_scale = 0.15
        self.parkours.base_parkour.curriculum_min_up_distance = 1.2
        self.commands.base_velocity.ranges.lin_vel_x = (0.15, 0.45)
        self.commands.base_velocity.heading_control_stiffness = 1.2
        self.events.push_by_setting_velocity.interval_range_s = (12.0, 12.0)


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
            self.scene.terrain.terrain_generator.difficulty_range = (0.0, 1.0)
        self.events.randomize_rigid_body_com = None
        self.events.randomize_rigid_body_mass = None
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
            self.scene.terrain.terrain_generator.difficulty_range = (0.2, 0.8)
        self.events.push_by_setting_velocity = None
