import numpy as np
from multiagent.core import World, Agent, Landmark, DynamicLandmark
from multiagent.scenario import BaseScenario
import math

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        self.num_agents = 2
        num_landmarks = 100
        num_dlandmark = 25
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.075*1.5
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks + 2)]
        for i, landmark in enumerate(world.landmarks):
            if i >= num_landmarks:
                landmark.name = 'landmark_wall %d' % i
                landmark.length = 100
                landmark.breadth = 0.1
            else:
                landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 2*0.15*(1 + i/100)
        # Create dynamic landmark
        world.dlandmarks = [DynamicLandmark() for i in range(num_dlandmark)]
        for i, dlandmark in enumerate(world.dlandmarks):
            dlandmark.name = 'dlandmark %d' % i
            dlandmark.collide = False
            dlandmark.movable = False
            dlandmark.size = 0.15
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])

        for i, dlandmark in enumerate(world.dlandmarks):
            dlandmark.color = np.array([0.50, 0.50, 0.50])

        if(self.num_agents is 1):
            agent.state.p_pos = np.array([0.0, 0.0])
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        else:
            for i, agent in enumerate(world.agents):
                agent.state.p_pos = np.array([i/(self.num_agents-1)-0.5, 0.0])
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)
        
        for i, landmark in enumerate(world.landmarks):
            if 'wall' in landmark.name:
                landmark.state.p_pos = np.array([2*(i%2) - 1, 0])
                landmark.state.p_vel = np.zeros(world.dim_p)
                landmark.color = np.array([0.15, 0.85, 0.85])
            else:
                array = np.array([np.random.choice([1.4*j/(i+1)-0.7 for j in range(i+2)]), 0*np.random.uniform(-0.5, +0.5, 1)[0]])
                landmark.state.p_pos = array + np.array([0, 3 + 1.6*i**0.85])
                landmark.state.p_vel = np.zeros(world.dim_p)
        
        for i, dlandmark in enumerate(world.dlandmarks):
                array = np.array([np.random.uniform(0, 1), 0])
                dlandmark.state.p_pos = array + np.array([0, 3 + 0.8*100**0.85 + 0.8*i**0.85])
                dlandmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def done(self, agent, world):
        dists = 10000000
        for l in world.landmarks + world.dlandmarks:
            if 'wall' not in l.name:
                dist = np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))
                if dist < dists:
                    dists = dist
                    landmark_size = l.size
        for other in world.agents:
            if agent.name is not other.name:
                dist = np.sqrt(np.sum(np.square(agent.state.p_pos - other.state.p_pos)))
                if dist < agent.size + other.size:
                    return True
        if dists < agent.size + landmark_size:
            return True
        elif abs(agent.state.p_pos[0]) > 0.9 - agent.size:
            return True
            # else:
        return False


    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 10
        for l in world.landmarks + world.dlandmarks:
            if 'wall' not in l.name:
                # dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
                dists = np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos))) 
                rew = min(rew, min(1, 0.5*(dists - l.size)/(l.size + 0*agent.size)))
        
        # rew = rew - 100
        for a in world.agents:
            if agent.name is not a.name:
                dist = np.sqrt(np.sum(np.square(agent.state.p_pos - a.state.p_pos)))
                rew += min(1, 0.25*(dist - a.size)/(a.size + 0*agent.size))
        # check this change.
        dist = abs(abs(agent.state.p_pos[0]) - 0.9)
        rew += min(1, 0.5*dist/0.2)
        # if agent.collide:
        #     for a in world.agents:
        #         if self.is_collision(a, agent):
        #             rew -= 1
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks + world.dlandmarks:
            if 'wall' not in entity.name:  # world.entities:
                entity_pos.append(np.sqrt(np.sum(np.square(agent.state.p_pos - entity.state.p_pos))))
        # entity colors
        entity_color = []
        for entity in world.landmarks + world.dlandmarks:  # world.entities:
            entity_color.append(entity.color)
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.array(list(agent.state.p_vel) + list(agent.state.p_pos) + entity_pos )
