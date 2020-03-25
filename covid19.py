import numpy as np
from scipy.spatial import distance
import pygame
from pygame.locals import *

# Total number of individuals
TOT = 120

# Window size
WIDTH, HEIGHT = 1000, 1000

# Graph height
GRAPH_HEIGHT = int(0.4 * HEIGHT)

# Playground Height
PLAY_HEIGHT = HEIGHT - GRAPH_HEIGHT

# Background
BACKGROUND_COLOR = 200, 200, 200
GRAPH_BACKGROUND_COLOR = 220, 220, 220
HEALTHY_COLOR = 255, 255, 0
CURED_COLOR = 0, 255, 0
SICK_COLOR = 255, 0, 0
INFECTED_COLOR = 255, 102, 255

# Possible states for each agent
HEALTHY, INFECTED, SICK, CURED = range(4)

# Agent image size
AGENT_SIZE = 30.0

# Bounce margin
BOUNCE_MARGIN = int(AGENT_SIZE/2.0)

# Max height
MAX_HEIGHT = PLAY_HEIGHT-BOUNCE_MARGIN

# Max width
MAX_WIDTH = WIDTH-BOUNCE_MARGIN

# Initial margin between agents
MARGIN = AGENT_SIZE / 10.0

# Vel modulus
VEL = 0.3

STEPS_PER_CYCLE = 5

# Counter threshold
THRESHOLD = 1800

# Time increment
TIME_INCREMENT = 0.3

# Percentual of people stopped
# This simulates social distancing
# PERC_STOPPED = 0.0
PERC_STOPPED = 0.5

# Quarantine (time since infection
# to put a person in quarantine)
#QUARANTINE_DELAY = 100*THRESHOLD
QUARANTINE_DELAY = THRESHOLD
#QUARANTINE_DELAY = 325

# Quarantine icon size
QUARANTINE_SIZE_MARGIN = 7
QUARANTINE_SIZE = AGENT_SIZE + 2*QUARANTINE_SIZE_MARGIN

# Press spacebar to start
# (if this is True you will need
#  to press spacebar to start the
#  simulation -- I use this to make
#  it easier to record a video)
PRESS_SPACE_TO_START = True

class Sim:
    def __init__(self):
        # Create agents in random positions
        self.pos = np.random.rand(TOT,2) * np.array([WIDTH-AGENT_SIZE, PLAY_HEIGHT-AGENT_SIZE]) \
                 + np.array([AGENT_SIZE/2.0, AGENT_SIZE/2.0])
        # Make sure we do not create agents on top of each other
        collisions = True
        while collisions:
            collisions = False
            for i,j in np.argwhere(distance.cdist(self.pos,self.pos) < AGENT_SIZE+MARGIN):
                if i != j:
                    self.pos[i] = np.random.rand(1,2) * np.array([WIDTH-AGENT_SIZE, PLAY_HEIGHT-AGENT_SIZE]) \
                                + np.array([AGENT_SIZE/2.0, AGENT_SIZE/2.0])
                    collisions = True
        # Initial velocities
        self.vel = 2.0*np.random.rand(TOT,2)-1.0
        # Normalize velocities
        norm = VEL/np.linalg.norm(self.vel,axis=1)
        self.vel[:,0] *= norm
        self.vel[:,1] *= norm
        # Initially all healthy
        self.status = np.zeros((TOT),dtype='uint32')
        self.count = np.zeros((TOT))
        # Except for one infected
        infected = np.random.randint(0,TOT)
        self.status[infected] = INFECTED
        self.count[infected] = 0
        # People in "social isolation" (stopped, velocity zero)
        self.stopped = np.zeros((TOT),dtype='uint32')
        while np.sum(self.stopped) < TOT*PERC_STOPPED:
            i = np.random.randint(0,TOT)
            self.stopped[i] = 1
            self.vel[i] = np.array([0.0,0.0])
        # People in quarantine
        self.quarantine = np.zeros((TOT),dtype='uint32')
        # Initialize screen
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        # Load images
        self.imgs = [pygame.image.load('healthy.png'), pygame.image.load('infect.png'), \
                     pygame.image.load('sick.png'), pygame.image.load('cured.png'), \
                     pygame.image.load('eyes.png'), pygame.image.load('quarantine.png')]
        # Resize images
        for i in range(len(self.imgs)-1):
            self.imgs[i] = pygame.transform.scale(self.imgs[i], (int(AGENT_SIZE), int(AGENT_SIZE)))
        self.imgs[-1] = pygame.transform.scale(self.imgs[-1], (int(QUARANTINE_SIZE), int(QUARANTINE_SIZE)))
        self.start = not PRESS_SPACE_TO_START
        # Counter
        self.time = 0.0
        self.time_bef = -1
    def update(self):
        # Update velocities
        self.pos += self.vel
        # Check window boundaries
        left_top_bounce = (self.pos < AGENT_SIZE/2.0).astype('uint32')
        self.vel += -2*left_top_bounce*self.vel
        width_bounce = (self.pos[:,0] > MAX_WIDTH).astype('uint32')
        self.vel[:,0] += -2*width_bounce*self.vel[:,0]
        height_bounce = (self.pos[:,1] > MAX_HEIGHT).astype('uint32')
        self.vel[:,1] += -2*height_bounce*self.vel[:,1]
        # Make sure no agent is out of bounds
        self.pos = left_top_bounce*BOUNCE_MARGIN + (1.0 - left_top_bounce)*self.pos
        self.pos[:,1] = height_bounce*MAX_HEIGHT + (1.0 - height_bounce)*self.pos[:,1]
        self.pos[:,0] = width_bounce*MAX_WIDTH + (1.0 - width_bounce)*self.pos[:,0]
        # Check collisions and infect agents
        for i, j in np.argwhere(distance.cdist(self.pos,self.pos) < AGENT_SIZE):
            if i > j:
                if (self.status[i] == SICK or self.status[i] == INFECTED) and self.quarantine[i] == 0 and self.status[j] == HEALTHY:
                    self.status[j] = INFECTED
                if (self.status[j] == SICK or self.status[j] == INFECTED) and self.quarantine[j] == 0 and self.status[i] == HEALTHY:
                    self.status[i] = INFECTED
                if self.stopped[i] == 0 and self.stopped[j] == 0 and self.quarantine[i] == 0 and self.quarantine[j] == 0:
                    self.vel[i], self.vel[j] = self.vel[j].copy(), self.vel[i].copy()
                else:
                    if self.stopped[i] >= 1 or self.quarantine[i] >= 1:
                        self.vel[j] = self.vel[j] - 2.0 * \
                                (self.pos[j]-self.pos[i]) * \
                                np.dot(self.vel[j]-self.vel[i],self.pos[j]-self.pos[i]) / AGENT_SIZE
                    else:
                        self.vel[i] = self.vel[i] - 2.0 * \
                                (self.pos[i]-self.pos[j]) * \
                                np.dot(self.vel[i]-self.vel[j],self.pos[i]-self.pos[j]) / AGENT_SIZE
        # Normalize velocities
        norm = VEL/np.linalg.norm(self.vel,axis=1)
        norm[np.isinf(norm)] = 1.0
        #print(norm)
        self.vel[:,0] *= norm
        self.vel[:,1] *= norm
        #print(self.vel)
        # Count the infected, to get sick
        infected = (self.status == INFECTED).astype('uint32')
        self.count += infected
        new_sick = (self.count == THRESHOLD).astype('uint32')
        self.status += new_sick
        self.count += infected * new_sick
        # Count the sick, to get cured
        sick = (self.status == SICK).astype('uint32')
        self.count += sick
        new_cured = (self.count == 2*THRESHOLD).astype('uint32')
        self.status += new_cured
        self.count += sick * new_cured
        # Put people in quarantine
        self.quarantine = (self.count > QUARANTINE_DELAY).astype('uint32') * (self.count < 2*THRESHOLD-1).astype('uint32')
        self.vel[:,0] *= 1.0 - self.quarantine
        self.vel[:,1] *= 1.0 - self.quarantine
        restore_vels = 2.0*np.random.rand(TOT,2)-1.0
        norm = VEL/np.linalg.norm(restore_vels,axis=1)
        restore_vels[:,0] *= norm
        restore_vels[:,1] *= norm
        self.vel[:,0] += (self.stopped == 0).astype('uint32')*(self.count == 2*THRESHOLD-1).astype('uint32')*restore_vels[:,0]
        self.vel[:,1] += (self.stopped == 0).astype('uint32')*(self.count == 2*THRESHOLD-1).astype('uint32')*restore_vels[:,1]
    def draw(self):
        # Draw the agents on the screen
        pygame.draw.rect(self.screen, BACKGROUND_COLOR, (0,0,WIDTH,PLAY_HEIGHT))
        for i in range(TOT):
            x = int(self.pos[i,0] - AGENT_SIZE/2.0)
            y = int(self.pos[i,1] - AGENT_SIZE/2.0)
            vx = int(self.vel[i,0] * 3.0/VEL)
            vy = int(self.vel[i,1] * 3.0/VEL)
            self.screen.blit(self.imgs[self.status[i]],(x,y))
            self.screen.blit(self.imgs[-2],(x+vx,y+vy))
            if self.quarantine[i] == 1:
                self.screen.blit(self.imgs[-1],(x-QUARANTINE_SIZE_MARGIN, y-QUARANTINE_SIZE_MARGIN))
        # Draw the graph
        time = int(self.time)
        if self.time_bef != time:
            self.time_bef = time
            healthy = GRAPH_HEIGHT * np.sum(self.status == HEALTHY) / TOT
            infected = GRAPH_HEIGHT * np.sum(self.status == INFECTED) / TOT
            sick = GRAPH_HEIGHT * np.sum(self.status == SICK) / TOT
            cured = GRAPH_HEIGHT * np.sum(self.status == CURED) / TOT
            pygame.draw.rect(self.screen, HEALTHY_COLOR, (time, PLAY_HEIGHT, time, int(HEIGHT - sick - infected - cured)))
            pygame.draw.rect(self.screen, CURED_COLOR, (time, int(HEIGHT - sick - infected - cured), time, int(HEIGHT - sick - infected)))
            pygame.draw.rect(self.screen, INFECTED_COLOR, (time, int(HEIGHT - sick - infected), time, int(HEIGHT - sick)))
            pygame.draw.rect(self.screen, SICK_COLOR, (time, int(HEIGHT - sick), time, int(HEIGHT)))
            pygame.draw.rect(self.screen, GRAPH_BACKGROUND_COLOR, (time+1,PLAY_HEIGHT,WIDTH,HEIGHT))
        pygame.display.flip()
    def run(self):
        # Main loop
        quit = False
        while not quit:
            if self.start:
                for i in range(STEPS_PER_CYCLE):
                    self.update()
                self.time += TIME_INCREMENT
            self.draw()
            # Check for quit or spacebar presses
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit = True
                elif event.type == pygame.KEYUP:
                    if event.key == K_SPACE:
                        self.start = True
                    if event.key == K_ESCAPE or event.key == K_q:
                        quit = True

# Main
if __name__ == '__main__':
    s = Sim()
    s.run()

