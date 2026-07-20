"""
Here, we wrap the original environment to make it easier
to use. When a game is finished, instead of mannualy reseting
the environment, we do it automatically.
"""
import numpy as np
import torch 
import time

def _format_observation(obs, device):
    """
    A utility function to process observations and
    move them to CUDA.
    """
    position = obs['position']
    if not device == "cpu":
        device = 'cuda:' + str(device)
    device = torch.device(device)
    factorized = 'x_state_single' in obs
    if factorized:
        formatted = {
            'z_single': torch.from_numpy(obs['z_single']).to(device),
            'x_state_single': torch.from_numpy(obs['x_state_single']).to(device),
            'x_action': torch.from_numpy(obs['x_action']).to(device),
            'legal_actions': obs['legal_actions'],
        }
    else:
        formatted = {
            'x_batch': torch.from_numpy(obs['x_batch']).to(device),
            'z_batch': torch.from_numpy(obs['z_batch']).to(device),
            'legal_actions': obs['legal_actions'],
        }
    x_no_action = torch.from_numpy(obs['x_no_action']).to(device)
    z = torch.from_numpy(obs['z']).to(device)
    return position, formatted, x_no_action, z

class Environment:
    def __init__(self, env, device):
        """ Initialzie this environment wrapper
        """
        self.env = env
        self.device = device
        self.episode_return = None

    def initial(self):
        initial_position, initial_obs, x_no_action, z = _format_observation(self.env.reset(), self.device)
        initial_reward = torch.zeros(1, 1)
        self.episode_return = torch.zeros(1, 1)
        initial_done = torch.ones(1, 1, dtype=torch.bool)

        return initial_position, initial_obs, dict(
            done=initial_done,
            episode_return=self.episode_return,
            obs_x_no_action=x_no_action,
            obs_z=z,
            timing=dict(env_step_ns=0, legal_actions_ns=0,
                        observation_ns=self.env.last_observation_ns),
            )
        
    def step(self, action):
        started_ns = time.perf_counter_ns()
        obs, reward, done, _ = self.env.step(action)
        env_step_ns = time.perf_counter_ns() - started_ns

        self.episode_return += reward
        episode_return = self.episode_return 

        if done:
            obs = self.env.reset()
            self.episode_return = torch.zeros(1, 1)

        position, obs, x_no_action, z = _format_observation(obs, self.device)
        reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done).view(1, 1)
        
        return position, obs, dict(
            done=done,
            episode_return=episode_return,
            obs_x_no_action=x_no_action,
            obs_z=z,
            timing=dict(
                env_step_ns=env_step_ns,
                legal_actions_ns=(
                    0 if done else self.env._env.last_legal_actions_ns
                ),
                observation_ns=self.env.last_observation_ns,
            ),
            )

    def close(self):
        close = getattr(self.env, 'close', None)
        if close is not None:
            close()
