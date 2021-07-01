from tiago_rl.envs import BulletRobotEnv


class TIAGoPALGripperEnv(BulletRobotEnv):

    def __init__(self, initial_state=None, *args, **kwargs):
        joints = [
            'torso_lift_joint',
            'arm_1_joint',
            'arm_2_joint',
            'arm_3_joint',
            'arm_4_joint',
            'arm_5_joint',
            'arm_6_joint',
            'arm_7_joint',
            'gripper_right_finger_joint',
            'gripper_left_finger_joint',
        ]

        initial_state = initial_state or [
            0.,
            2.71,
            -0.173,
            1.44,
            1.79,
            0.23,
            -0.0424,
            -0.0209,
            0.045,
            0.045
        ]

        BulletRobotEnv.__init__(self,
                                joints=joints,
                                initial_state=initial_state,
                                cam_yaw=89.6000747680664,
                                cam_pitch=-35.40000915527344,
                                cam_distance=1.6000027656555176,
                                robot_model="tiago_pal_gripper.urdf",
                                object_model="objects/object.urdf",
                                object_pos=[0.73, 0.07, 0.6],
                                table_model="objects/table.urdf",
                                table_pos=[0.7, 0, 0.27],
                                *args, **kwargs)

    def _is_success(self):
        return False

    def _compute_reward(self):
        return 0
