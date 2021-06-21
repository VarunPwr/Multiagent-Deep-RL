# Gym Like class for multiagent-particle-envs
from multiagent.environment import MultiAgentEnv
from multiagent.policy import SinglePolicy
import multiagent.scenarios as scenarios
import numpy as np

class MultiagentEnv:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        scenario = scenarios.load('simple_spread.py').Scenario()
        world = scenario.make_world()
        self.env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, done_callback=scenario.done, shared_viewer = False)
        self.policies = [SinglePolicy(self.env,0) for i in range(num_actions)]
        self.action_space = 2
    def reset(self):
        self.env.reset()
        if self.num_actions is 1:
            return np.array(self.env.render(mode='rgb_array'))[0]
        else:
            return [np.array(img)[:,:,:3] for img in self.env.render(mode='rgb_array')]

    def step(self, actions):
        if isinstance(actions, list):
            pass
        else:
            actions = [actions]
        act_n = []
        for policy, action in zip(self.policies, actions):
            act_n.append(policy.action(action))
        _, reward, Done, _ = self.env.step(act_n)
        
        if isinstance(Done, list):
            pass
        else:
            Done = [Done]
        for done in Done:
            if done:
                 break

        if self.num_actions is 1:
            return np.array(self.env.render(mode='rgb_array'))[0,:,:,:3], reward[0], done, None
        else:
            return [np.array(img)[:,:,:3] for img in self.env.render(mode='rgb_array')], np.sum(reward), done, None
