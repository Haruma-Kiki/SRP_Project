import torch
import os
from datetime import datetime
import numpy as np
from mpi4py import MPI
from .models import actor, critic
from .utils import sync_networks, sync_grads
from .replay_buffer import replay_buffer
from .normalizer import normalizer
from .her import her_sampler
import pandas as pd

# from tensorboardX import SummaryWriter

"""
ddpg with HER (MPI-version)

"""
class ddpg:
    def __init__(self, args, env, env_params):
        self.args = args
        self.env = env
        self.env_params = env_params
        # 设置训练设备（GPU or CPU）
        self.device = torch.device("cuda" if self.args.cuda and torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")
        # 创建main网络中的actor网络和critic网络
        self.actor_network = actor(env_params).to(self.device)
        self.critic_network = critic(env_params).to(self.device)
        # sync the networks across the cpus
        sync_networks(self.actor_network)
        sync_networks(self.critic_network)
        # 目标actor网络和目标critic网络
        self.actor_target_network = actor(env_params).to(self.device)
        self.critic_target_network = critic(env_params).to(self.device)
        # 复制网络参数
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        # cuda，然而本垃圾电脑不支持cuda，就没尝试
        if self.args.cuda and torch.cuda.is_available():
            self.actor_network.cuda()
            #print(self.actor_network.parameters.)
            print('1111111111111')
            print(next(self.actor_network.parameters()).device)
            self.critic_network.cuda()
            self.actor_target_network.cuda()
            self.critic_target_network.cuda()
        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)
        # her部分，其实不存储数据，只是作为一个数据转换器，将普通数据转化为her数据
        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, self.env.compute_reward)
        # create the replay buffer
        self.buffer = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)
        # create the normalizer
        self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)

        # add tensorboardX tool
        
        # create the dict for store the model
        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)
            # path to save the model
            self.model_path = os.path.join(self.args.save_dir, self.args.env_name)
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)
            print('model path', self.model_path)
        
        # self.writer = SummaryWriter('./logs')
        self.reward_list = []
        self.reward_record = []
        self.success_rate_list = []
        self.success_list = []

    def learn(self):
        """
        train the network
        """
        # start to collect samples
        for epoch in range(self.args.n_epochs):
            for cycle in range(self.args.n_cycles):
                mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
                reward_total = 0
                for _ in range(self.args.num_rollouts_per_mpi):
                    reward_temp = 0
                    # reset the rollouts
                    ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
                    # reset the environment
                    observation = self.env.reset()
                    obs = observation['observation']
                    ag = observation['achieved_goal']
                    g = observation['desired_goal']
                    # 先收集数据，填充replay buffer
                    for t in range(self.env_params['max_timesteps']):
                        # if t % 100 == 0:
                        #     print('[{}] epoch {} cycyle {} rank {} step {}'.format(datetime.now(), epoch, cycle, MPI.COMM_WORLD.Get_rank(),t))
                        with torch.no_grad():
                            input_tensor = self._preproc_inputs(obs, g)
                            pi = self.actor_network(input_tensor)
                            action = self._select_actions(pi)
                        # 执行动作并获取反馈
                        observation_new, reward, done, info = self.env.step(action)
                        reward_total += reward
                        reward_temp += reward
                        obs_new = observation_new['observation']
                        ag_new = observation_new['achieved_goal']
                        # append rollouts
                        ep_obs.append(obs.copy())
                        ep_ag.append(ag.copy())
                        ep_g.append(g.copy())
                        ep_actions.append(action.copy())
                        # re-assign the observation
                        obs = obs_new
                        ag = ag_new
                    
                    ep_obs.append(obs.copy())
                    ep_ag.append(ag.copy())
                    mb_obs.append(ep_obs)
                    mb_ag.append(ep_ag)
                    mb_g.append(ep_g)
                    mb_actions.append(ep_actions)
                    self.success_list.append([epoch, cycle, info['is_success']])
                    self.reward_list.append([epoch, cycle, reward_temp])
                print('[{}] epoch {} cycyle {} rank {} reward {}'.format(datetime.now(), epoch, cycle, MPI.COMM_WORLD.Get_rank(), reward_total))
                # convert them into arrays
                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_actions = np.array(mb_actions)
                # 将数据放入replay buffer
                self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
                self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])
                for _ in range(self.args.n_batches):
                    # train the network
                    self._update_network()
                # soft update
                self._soft_update_target_network(self.actor_target_network, self.actor_network)
                self._soft_update_target_network(self.critic_target_network, self.critic_network)
                if MPI.COMM_WORLD.Get_rank() == 0:
                    print('[{}] epoch is {}, cycle is {}, reward is {}.'.format(datetime.now(), epoch, cycle, reward_total))
                # self.reward_record.append([epoch, cycle, reward_total])
                
                # self.writer.add_scalar('reward', reward_total, epoch, epoch*50+cycle)

            # start to do the evaluation
            success_rate = self._eval_agent()
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('[{}] epoch is {}, eval success rate is {:.3f}'.format(datetime.now(), epoch, success_rate))
                torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.actor_network.state_dict()], \
                            self.model_path + '/model.pt')
            self.success_rate_list.append([epoch, success_rate])
        
        # 保存数据
        self.reward_list = pd.DataFrame(self.reward_list)
        self.reward_record = pd.DataFrame(self.reward_record)
        self.success_list = pd.DataFrame(self.success_list)
        self.success_rate_list = pd.DataFrame(self.success_rate_list)

        self.reward_list.to_excel(self.model_path+'/reward_list.xls')
        self.reward_record.to_excel(self.model_path+'/reward_record.xls')
        self.success_list.to_excel(self.model_path+'/success_list.xls')
        self.success_rate_list.to_excel(self.model_path+'/success_rate_list.xls')
        
    # pre_process the inputs
    def _preproc_inputs(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).to(self.device)
        if self.args.cuda and torch.cuda.is_available():
            inputs = inputs.cuda()
        return inputs
    
    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi):
        action = pi.cpu().numpy().squeeze()
        # add the gaussian
        action += self.args.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
        # random actions...
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                            size=self.env_params['action'])
        # choose if use the random actions
        action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)
        return action

    # update the normalizer
    def _update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs, 
                       'ag': mb_ag,
                       'g': mb_g, 
                       'actions': mb_actions, 
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       }
        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the network
    def _update_network(self):
        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size)
        # pre-process the observation and goal
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
        # start to do the update
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32).to(self.device)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32).to(self.device)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32).to(self.device)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32) .to(self.device)
        if self.args.cuda and torch.cuda.is_available():
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()
        # calculate the target Q value function
        with torch.no_grad():
            # do the normalization
            # concatenate the stuffs
            actions_next = self.actor_target_network(inputs_next_norm_tensor)
            q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()
            # clip the q value
            clip_return = 1 / (1 - self.args.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)
        # the q loss
        real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()
        # the actor loss
        actions_real = self.actor_network(inputs_norm_tensor)
        actor_loss = -self.critic_network(inputs_norm_tensor, actions_real).mean()
        actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()
        # record loss infomation
        # print('actor loss:', actor_loss.data)
        # print('critic loss:', critic_loss.data)
        # self.writer.add_scalar('actor loss', actor_loss.data)
        # self.writer.add_scalar('critic loss', critic_loss.data)
        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor_network)
        self.actor_optim.step()
        # update the critic_network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic_network)
        self.critic_optim.step()

    # do the evaluation
    # 训练一轮之后会进行一次测试，检测模型的效果
    def _eval_agent(self):
        total_success_rate = []
        for _ in range(self.args.n_test_rollouts):
            per_success_rate = []
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g)
                    pi = self.actor_network(input_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()
                observation_new, _, _, info = self.env.step(actions)
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                per_success_rate.append(info['is_success'])
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size()
