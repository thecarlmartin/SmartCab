from __future__ import division
import pandas as pd
import numpy as np
import math
import matplotlib as mpl
mpl.use('MacOSX')
import matplotlib.pyplot as plt
# --> Custom Imports Above

from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        self.alpha = 0.9
        self.gamma = 0.5
        self.epsilon = 0.9
        self.available_actions = ['stay', 'forward', 'left', 'right']
        self.metrics = pd.DataFrame(columns=['Success', 'CumSuccess', 'Total Reward', 'CumTotal Reward', 'Violations', 'Accidents', 'Trip Duration'])
        self.metrics.index.name = 'trial'
        self.trial = 0

        # NOTE: Adding the initialization of the Q Values Dictionary
        self.q_values = pd.DataFrame(columns=self.available_actions)
        self.q_values.index.name = 'state'

    def reset(self, destination=None):
        self.planner.route_to(destination)

        # TODO: Prepare for a new trip; reset any variables here, if required
        self.route_duration = None
        self.alpha -= self.alpha * 0.05
        self.epsilon -= self.epsilon * 0.6
        self.trial += 1
        self.metrics.loc[self.trial] = [0, 0, 0, 0, 0, 0, 0]

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # Setting the current state
        self.state = self.assemble_state(deadline, self.next_waypoint, inputs)

        # Selecting the appropriate action based on the policy
        action = self.policy(self.state)

        # Execute action and get reward
        if action == 'stay':
            reward = self.env.act(self, None)
        else:
            reward = self.env.act(self, action)

        location = self.env.agent_states[self]["location"]
        destination = self.env.agent_states[self]["destination"]


        # Code for tracking relevant agent metrics
        if location == destination:
            self.metrics.loc[self.trial, 'Success'] = 1

        if reward == -0.5:
            self.metrics.loc[self.trial, 'Violations'] += 1

        if reward == -1:
            self.metrics.loc[self.trial, 'Accidents'] += 1

        self.metrics.loc[self.trial, 'Trip Duration'] = (self.route_duration - deadline)/self.route_duration

        self.metrics.loc[self.trial, 'Total Reward'] += reward

        # Getting the new state of Smartcab and recording the learned Q value
        new_state = self.assemble_state(deadline - 1, self.planner.next_waypoint(), self.env.sense(self))

        self.q_values.loc[self.state, action] = (1 - self.alpha) * self.get_qvalue(self.state, action) + self.alpha * (reward + self.gamma * self.max_q(new_state))

        print "LearningAgent.update(): waypoint={}, deadline = {}, inputs = {}, state = {}, action = {}, reward = {}".format(self.next_waypoint, deadline, inputs, self.state, action, reward)  # [debug]

        self.q_values.to_csv('q_table.csv')
        self.metrics.to_csv('metrics.csv')

        self.metrics['CumSuccess'] = self.metrics['Success'].cumsum()
        self.metrics['CumTotal Reward'] = self.metrics['Total Reward'].cumsum()




    def assemble_state(self, deadline, next_waypoint, inputs):

        # Implementation of the deadline portion
        if self.route_duration == None:
            self.route_duration = deadline

        '''
        Available information in each state are:
        - deadline
        - waypoint
        - risk_level: drive_safe, drive_reckless or drive_accident
        States are set depending on combination of inputs. Consult the documentation for more details.
        '''

        if deadline/self.route_duration > 0.4:
            state_deadline = 1
        else:
            state_deadline = 0

        return "{}, {}, {}, {}, {}".format(state_deadline, next_waypoint, inputs['light'], inputs['left'], inputs['oncoming'], inputs['right'])

    def get_qvalue(self, state, action):

        if state in self.q_values.index:
            if not math.isnan(self.q_values.loc[state, action]):
                return self.q_values.loc[state, action]
            else:
                return 0
        else:
            # Q Matrix is initialized to zero
            return 0

    def max_q(self, state):

        if state in self.q_values.index:
            return self.q_values.loc[state].max()
        else:
            return 0

    def policy(self, state):

        # Choosing a random action 20% of the time and 80% of the time follow the max Q value
        action_category = np.random.choice([0, 1], p=[1 - self.epsilon, self.epsilon])

        if action_category == 0:
            if self.state in self.q_values.index:

                for action_key in self.available_actions:
                    # Prefering untaken actions over actions with high q value to reduce local optima
                    if math.isnan(self.q_values.loc[self.state, action_key]):
                        action = action_key
                        break
                    else:
                        action = self.q_values.loc[self.state].idxmax()
            else:
                action_category = 1

        if action_category == 1:
            print "Random Action Chosen"
            action = np.random.choice(self.available_actions)

        return action


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
