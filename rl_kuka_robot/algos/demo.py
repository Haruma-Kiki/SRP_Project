import torch
from models import actor
from arguments import get_args
import gym
import numpy as np


from pybullet_robot_envs.envs.kuka_envs.kuka_reach_gym_env_her import kukaReachGymEnvHer
from pybullet_robot_envs.envs.kuka_envs.kuka_push_gym_env_her import kukaPushGymEnvHer
from pybullet_robot_envs.envs.kuka_envs.kuka_reach_gym_env_obstacle import kukaReachGymEnvOb
from pybullet_robot_envs.envs.kuka_envs.kuka_reach_obs import kukaReachGymEnvblock
import robot_data

# process the inputs
def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs

if __name__ == '__main__':
    args = get_args()
    # load the model param
    model_path = args.save_dir + args.env_name + '/model.pt'
    print('model path:', model_path)
    o_mean, o_std, g_mean, g_std, model = torch.load(model_path, map_location=lambda storage, loc: storage)
    # create the environment
    # env = gym.make(args.env_name)
    rend = True
    discreteAction = 0
    numControlledJoints = 6
    fixed = False
    actionRepeat = 1
if args.env_name.startswith('reach'):
    env = kukaReachGymEnvHer(urdfRoot=robot_data.getDataPath(),actionRepeat=actionRepeat,renders=rend, useIK=0, isDiscrete=discreteAction,
                            numControlledJoints=numControlledJoints, fixedPositionObj=fixed, includeVelObs=True, reward_type=1)
elif args.env_name.startswith('push'):
    env = kukaPushGymEnvHer(urdfRoot=robot_data.getDataPath(),actionRepeat=actionRepeat,renders=rend, useIK=0, isDiscrete=discreteAction,
                            numControlledJoints=numControlledJoints, fixedPositionObj=fixed, includeVelObs=True, reward_type=1)
elif args.env_name.startswith('obreach'):
    env = kukaReachGymEnvOb(urdfRoot=robot_data.getDataPath(), renders=rend, useIK=0, isDiscrete=discreteAction,
                            numControlledJoints=numControlledJoints, fixedPositionObj=fixed, includeVelObs=True, 
                            reward_type=1,obstacles_num=args.obs_num)
elif args.env_name.startswith('blockreach'):
    env = kukaReachGymEnvblock(urdfRoot=robot_data.getDataPath(), renders=rend, useIK=0, isDiscrete=discreteAction,
                            numControlledJoints=numControlledJoints, fixedPositionObj=fixed, includeVelObs=True, 
                            reward_type=1)                            
else:
    env = kukaReachGymEnvHer(urdfRoot=robot_data.getDataPath(),actionRepeat=actionRepeat,renders=rend, useIK=0, isDiscrete=discreteAction,
                            numControlledJoints=numControlledJoints, fixedPositionObj=fixed, includeVelObs=True, reward_type=reward_type)
                            
    # get the env param
    observation = env.reset()
    # get the environment params
    env_params = {'obs': observation['observation'].shape[0], 
                  'goal': observation['desired_goal'].shape[0], 
                  'action': env.action_space.shape[0], 
                  'action_max': env.action_space.high[0],
                  }
    # create the actor network
    actor_network = actor(env_params)
    actor_network.load_state_dict(model)
    actor_network.eval()
    args.demo_length = 20
    success = 0
    import time
    time.sleep(5)
    for i in range(args.demo_length):
        observation = env.reset()
        # start to do the demo
        obs = observation['observation']
        g = observation['desired_goal']
        for t in range(env._maxSteps):
            # env.render()
            inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args)
            with torch.no_grad():
                pi = actor_network(inputs)
            action = pi.detach().numpy().squeeze()
            # put actions into the environment
            observation_new, reward, done, info = env.step(action)
            success += info['is_success']
            obs = observation_new['observation']
            if done:
                break
        env.render()
        print('the episode is: {}, is success: {}'.format(i, info['is_success']))
    print(success)
