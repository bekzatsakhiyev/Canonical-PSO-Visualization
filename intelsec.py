from mesa import Agent, Model
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.batchrunner import RandomActivation
import random
import numpy as np
NUM_PARTICLES = 50
INERTIA = 0.7
C1 = 1.5
C2 = 1.5
WIDTH, HEIGHT = 50, 50
def objective_function(x, y):
    return (x - 25) ** 2 + (y - 25) ** 2
class Particle(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.pos = np.array([random.uniform(0, WIDTH), random.uniform(0, HEIGHT)])
        self.velocity = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
        self.pbest = self.pos.copy()
        self.pbest_value = objective_function(*self.pbest)

    def step(self):
        r1, r2 = random.random(), random.random()
        self.velocity = (
            INERTIA * self.velocity
            + C1 * r1 * (self.pbest - self.pos)
            + C2 * r2 * (self.model.gbest - self.pos)
        )
        self.pos += self.velocity
        new_value = objective_function(*self.pos)
        if new_value < self.pbest_value:
            self.pbest = self.pos.copy()
            self.pbest_value = new_value
    def get_position(self):
        return int(self.pos[0]), int(self.pos[1])
class PSOModel(Model):
    def __init__(self, num_particles):
        self.num_particles = num_particles
        self.space = ContinuousSpace(WIDTH, HEIGHT, torus=False)
        self.schedule = RandomActivation(self)
        self.particles = [Particle(i, self) for i in range(num_particles)]
        for particle in self.particles:
            self.schedule.add(particle)
        self.gbest = min(self.particles, key=lambda p: p.pbest_value).pbest
    def step(self):
        self.schedule.step()
        self.gbest = min(self.particles, key=lambda p: p.pbest_value).pbest
def agent_portrayal(agent):
    return {"Shape": "circle", "r": 2, "Color": "blue", "Filled": "true", "Layer": 1}

grid = CanvasGrid(agent_portrayal, WIDTH, HEIGHT, 500, 500)
server = ModularServer(PSOModel, [grid], "Particle Swarm Optimization", {"num_particles": NUM_PARTICLES})
server.launch()
