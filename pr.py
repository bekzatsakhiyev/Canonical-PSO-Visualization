import agentpy as ap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
def rastrigin(x, y):
    return 20 + x**2 + y**2 - 10 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))
class PSOAgent(ap.Agent):
    def setup(self):
        self.position = np.array([self.model.random.uniform(-5, 5),
                                  self.model.random.uniform(-5, 5)])
        self.velocity = np.array([self.model.random.uniform(-1, 1),
                                  self.model.random.uniform(-1, 1)])
        self.pbest = self.position
        self.pbest_value = rastrigin(*self.pbest)
    def step(self):
        w = 0.5
        c1, c2 = 1.5, 1.5
        gbest = self.model.gbest
        r1, r2 = np.random.uniform(0, 1, 2)
        self.velocity = (w * self.velocity +
                         c1 * r1 * (self.pbest - self.position) +
                         c2 * r2 * (gbest - self.position))
        self.position += self.velocity
        self.position = np.clip(self.position, -5, 5)
        value = rastrigin(*self.position)
        if value < self.pbest_value:
            self.pbest = self.position
            self.pbest_value = value
class PSOModel(ap.Model):
    def setup(self):
        self.agents = ap.AgentList(self, 30, PSOAgent)
        self.gbest = min(self.agents, key=lambda a: a.pbest_value).pbest
        self.positions = []

    def step(self):
        self.agents.step()
        self.gbest = min(self.agents, key=lambda a: a.pbest_value).pbest
        self.positions.append(np.array([agent.position for agent in self.agents]))
model = PSOModel()
model.run(steps=50)
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
sc = ax.scatter([], [], c='blue')
ax.set_title("Canonical PSO Visualization")
ax.set_xlabel("X координата")
ax.set_ylabel("Y координата")
def update(frame):
   sc.set_offsets(model.positions[frame])
   return sc,
ani = animation.FuncAnimation(fig, update, frames=len(model.positions), interval=100, blit=True)

plt.show()
