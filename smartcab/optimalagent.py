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

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        # NOTE: Custom variables below that are persistent for the entire duration of the simulation
        self.available_actions = ['stay', 'forward', 'left', 'right'] # I chose stay instead of None, because it resolved troubles I had with indexing

        #TODO: Revise Metrics
        self.metrics = pd.DataFrame(columns=['Success', 'CumSuccess', 'Total Reward', 'CumTotal Reward', 'Planner not Observed', 'Invalid Move', 'Trip Duration'])
        self.metrics.index.name = 'trial'

        self.trial = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)

        # NOTE: Resetting custom variables
        self.route_duration = None

        # NOTE: Preparing variables for new trial
        self.trial += 1
        self.metrics.loc[self.trial] = [0, 0, 0, 0, 0, 0, 0]

    def update(self, t):
        # NOTE: Getting Inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # NOTE: Saving the route duration for calculations and analysis
        if self.route_duration == None:
            self.route_duration = deadline

        # NOTE: Selecting the appropriate action based on the policy
        if self.next_waypoint == 'left':
            if inputs['light'] == 'green' and inputs['oncoming'] == None:
                action = self.next_waypoint
            else:
                action = 'stay'
        elif self.next_waypoint == 'forward':
            if inputs['light'] == 'green':
                action = self.next_waypoint
            else:
                action = 'stay'
        elif self.next_waypoint == 'right':
            if inputs['left'] == None:
                action = self.next_waypoint
            else:
                action = 'stay'
        else:
            action = 'stay'

        # NOTE: Execute action and get reward
        if action == 'stay':
            reward = self.env.act(self, None)
        else:
            reward = self.env.act(self, action)

        # NOTE: Tracking relevant agent metrics
        if self.env.agent_states[self]["location"] == self.env.agent_states[self]["destination"]:
            self.metrics.loc[self.trial, 'Success'] = 1

        # TODO: Adjust to fit reward structure
        if reward == -0.5:
            self.metrics.loc[self.trial, 'Planner not Observed'] += 1

        if reward == -1:
            print "Error in code with {} and {}".format(inputs, action)
            self.metrics.loc[self.trial, 'Invalid Move'] += 1

        self.metrics.loc[self.trial, 'Trip Duration'] = (self.route_duration - deadline)/self.route_duration
        self.metrics.loc[self.trial, 'Total Reward'] += reward

        print "LearningAgent.update(): waypoint={}, deadline = {}, inputs = {}, state = {}, action = {}, reward = {}".format(self.next_waypoint, deadline, inputs, self.state, action, reward)  # [debug]

        # NOTE: Saving q matrix and metrics to csv - this has to be optimized
        self.metrics['CumSuccess'] = self.metrics['Success'].cumsum()
        self.metrics['CumTotal Reward'] = self.metrics['Total Reward'].cumsum()

        self.metrics.to_csv('metrics_optimal.csv')

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
