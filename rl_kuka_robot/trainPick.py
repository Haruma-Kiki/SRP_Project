import numpy as np
import gym
import os
import sys
from arguments import get_args
from mpi4py import MPI
from subprocess import CalledProcessError
from ddpg_agent import ddpg_agent
from ddpg import ddpg

from pybullet_robot_envs.envs.kuka_envs.kuka_reach_gym_env_her import kukaReachGymEnvHer
from pybullet_robot_envs.envs.kuka_envs.kuka_push_gym_env_her import kukaPushGymEnvHer
from pybullet_robot_envs.envs.kuka_envs.kuka_pick_gym_env_her import kukaPickGymEnvHer
import robot_data

from tensorboardX import SummaryWriter


"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)

"""


def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0],
              'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0],
              'action_max': env.action_space.high[0],
              }
    params['max_timesteps'] = int(env._maxSteps)
    return params


def launch(args):
    # create the ddpg_agent
    # env = gym.make(args.env_name)
    rend = False
    discreteAction = 0
    numControlledJoints = 9
    actionRepeat = 1
    fixed = False
    maxStep = 1000
    # env = kukaReachGymEnvHer(urdfRoot=robot_data.getDataPath(), renders=rend, useIK=0, isDiscrete=discreteAction,
    #                          numControlledJoints=numControlledJoints, fixedPositionObj=fixed, includeVelObs=True)
    env = kukaPickGymEnvHer(urdfRoot=robot_data.getDataPath(),maxSteps=maxStep,renders=rend, useIK=0, isDiscrete=discreteAction,actionRepeat=actionRepeat,
                             numControlledJoints=numControlledJoints, fixedPositionObj=fixed, includeVelObs=True, reward_type=1)
                             
    # get the environment parameters
    env_params = get_env_params(env)
    # create the ddpg agent to interact with the environment
    args.replay_strategy = 'normal'
    ddpg_trainer = ddpg(args, env, env_params)
    ddpg_trainer.learn()


if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # get the params
    args = get_args()
    launch(args)
