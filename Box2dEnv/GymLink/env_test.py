import sys
import math
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import seeding, EzPickle

# Tool use simulation for DRL benchmarking
#
# A collection of simple tool use-tasks in a 2D physics simulation.
# These were created to test RL algorithms ability to perform these tasks, specifically
# under conditions of sparse rewards.
#
# Created by Evguenni Penksik (University of Birmingham), adapted from LunarLander-v2 environment by Oleg Klimov.

## SET BASIC PROPERTIES

FPS = 50 # Default = 50
SCALE = 30.0   # affects how fast-paced the simulation is, forces should be adjusted as well. Default = 30

VIEWPORT_W = 600
VIEWPORT_H = 400

GRAVITY = -9.8
FORCE = 1

TASK = 'PUSH' # This should be set to 'PUSH' or 'LIFT'

## SPECIFY TOOL OBJECT GEOMETRIES

TOOL_1 = [(-10,30),(10,30),(10,-30),(-10,-30)]
TOOL_2 = [(-40,30),(-10,30),(-10,12),(-40,12)]
TOOL_3 = [(-40,-12),(-10,-12),(-10,-30),(-40,-30)]
TOOL_4 = [(10,8),(40,8),(40,-8),(10,-8)]

OBJ_1 = [(-10,30),(10,30),(10,-30),(-10,-30)]
OBJ_2 = [(-40,30),(-10,30),(-10,12),(-40,12)]
OBJ_3 = [(-40,-12),(-10,-12),(-10,-30),(-40,-30)]
OBJ_4 = [(10,8),(40,8),(40,-8),(10,-8)]

# Old geometries
# TOOL_POLY_1 = [(-10,10),(-30,10),(-30,0),(-10,0)]
# TOOL_POLY_2 = [(-10,20), (-10, -10), (-0, -10), (0, 20)]
# OBJ_POLY_1 = [(-10,10),(-30,10),(-30,0),(-10,0)]
# OBJ_POLY_2 = [(-10,20), (-10, -10), (-0, -10), (0, 20)]

SIZE = 35  # General scaling for geometry
TARGET_SIZE = 0.1

# TOOL_POLY_1 = [(SIZE,SIZE),(SIZE,-SIZE),(-SIZE,-SIZE),(-SIZE,SIZE)]
# OBJ_POLY_1 = [(SIZE,SIZE),(SIZE,-SIZE),(-SIZE,-SIZE),(-SIZE,SIZE)]

TARGET_POLY_1 = [(-TARGET_SIZE,-TARGET_SIZE),
                (+ TARGET_SIZE,-TARGET_SIZE),
                (+ TARGET_SIZE,+TARGET_SIZE),
                (- TARGET_SIZE,+TARGET_SIZE)]

class ContactDetector(contactListener):
    
    ## DEFINE COLLISION BEHAVIOUR
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
    def PostSolve(self,contact,manifold):
        if contact.fixtureA.body == self.env.tool:
            self.env.collisionObjectImpulse = manifold.normalImpulses[0]
        if contact.fixtureB.body == self.env.tool:
            self.env.collisionObjectImpulse = manifold.normalImpulses[0]
    def BeginContact(self, contact):
        if (self.env.tool == contact.fixtureA.body and self.env.moon == contact.fixtureB.body):
            self.env.tool_on_table = True
        if (self.env.tool == contact.fixtureB.body and self.env.moon == contact.fixtureA.body):
            self.env.tool_on_table = True
        if (self.env.tool == contact.fixtureA.body and self.env.object == contact.fixtureB.body):
            self.env.tool_on_object = True
            self.env.collidedThisStep = True
        if (self.env.tool == contact.fixtureB.body and self.env.object == contact.fixtureA.body):
            self.env.tool_on_object = True
            self.env.collidedThisStep = True
    def EndContact(self, contact):
        if (self.env.tool == contact.fixtureA.body and self.env.object == contact.fixtureB.body):
            self.env.tool_on_object = False
        if (self.env.tool == contact.fixtureB.body and self.env.object == contact.fixtureA.body):
            self.env.tool_on_object = False

class EnvTest(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    ## Set as true when using EnvTestContinuous
    continuous = False
    rotation = False
    task = TASK
    reward_level = 3
    rand_level = 3
    reward_scale = 6
    ## SET OTHER BASIC PROPERTIES

    def __init__(self):
        EzPickle.__init__(self)
        self.seed()
        self.viewer = None

        # Set global gravity here if required.
        # Otherwise gravity is applied as an acceleration on each individual object that requires it
        self.world = Box2D.b2World(gravity=(0,0), doSleep=True) 

        # Initilize objects in scene. TODO: Rename moon to ground
        self.moon = None
        self.tool = None
        self.object = None

        # Variables for calculating rewards/change in reward per frame, and collision markers
        self.prev_reward = None
        self.distance = (0.0,0.0)
        self.prev_distance = (0.0,0.0)
        self.prev_distance_L2 = 0.0
        self.prev_position = 0.0
        self.initial_distance = 0.0
        self.initial_distance_L2 = 0.0


        self.tool_on_table = False
        self.tool_on_object = False
        self.prev_tool_on_table = False
        self.prev_tool_on_object = False
        self.collisionObjectImpulse = 0.0
        self.collisionTableImpulse = 0.0
        self.collidedThisStep = False

        # Target position variable
        self.target_position = 0.0

        self.step_count = 0

        # Set format of observation and action variables. Size and range are required if changed, and are passed to the RL algorithm to size the network
        # self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32)

        if self.continuous:
            if not self.rotation:
                self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)
            elif self.rotation:
                self.action_space = spaces.Box(-1, +1, (3,), dtype=np.float32)
        elif not self.continuous:
            if not self.rotation:
                self.action_space = spaces.Discrete(8)
            elif self.rotation:
                self.action_space = spaces.Discrete(4)
            # Nop, fire left engine, main engine, right engine
            

        self.reset()

    def seed(self, seed=None):
        ## TODO add this as a parameter when initializing the env
        #self.np_random, seed = seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]

    def _destroy(self):
        ## Update this list for all objects created in scene
        if not self.moon: return
        self.world.contactListener = None
        self.world.DestroyBody(self.moon)
        self.moon = None
        self.world.DestroyBody(self.tool)
        self.world.DestroyBody(self.object)
        self.tool = None
        self.object = None

    def reset(self):
        ## Ensure all tracking variables (for reward calculations are reset here)

        self.step_count = 0

        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None
        self.tool_on_table = False
        self.tool_on_object = False
        self.prev_tool_on_table = False
        self.prev_tool_on_object = False

        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE

        ## Build the table/ground
        GROUND_POLY_1 = [(1,0),(1,H/4),(W-1,H/4),(W-1,0)]

        # TODO rename moon to something more appropriate
        self.moon = self.world.CreateStaticBody(shapes=polygonShape(vertices=[(x,y) for x,y in GROUND_POLY_1]))
        self.sky_polys = []
        for i in range(1):
            p1 = (0, 0)
            p2 = (W, 0)
            self.sky_polys.append([p1, p2, (p2[0],H), (p1[0],H)])

        self.moon.color1 = (0.8,0.8,0.8)
        self.moon.color2 = (0.8,0.8,0.8)

        # TODO Move this to basic parameters above?
        initial_y = VIEWPORT_H / SCALE
        FRICTION = 5.0
        DENSITY_1 = 10.0
        DENSITY_2 = 0.5
        ## SPECIFY TOOL AND OBJECT GEOMETRIES
        # TODO: Tidy this up. Use a list or something

        definedFixturesTool_1 = fixtureDef(shape=polygonShape(vertices=[ (x / SCALE,y / SCALE) for x,y in TOOL_1 ]),
            density=DENSITY_1,
            friction=FRICTION,
            restitution=0.0)
            
        definedFixturesTool_2 = fixtureDef(shape=polygonShape(vertices=[ (x / SCALE,y / SCALE) for x,y in TOOL_2 ]),
            density=DENSITY_1,
            friction=FRICTION,
            restitution=0.0)

        definedFixturesTool_3 = fixtureDef(shape=polygonShape(vertices=[ (x / SCALE,y / SCALE) for x,y in TOOL_3 ]),
            density=DENSITY_1,
            friction=FRICTION,
            restitution=0.0)

        definedFixturesTool_4 = fixtureDef(shape=polygonShape(vertices=[ (x / SCALE,y / SCALE) for x,y in TOOL_4 ]),
            density=DENSITY_1,
            friction=FRICTION,
            restitution=0.0)

        definedFixturesObj_1 = fixtureDef(shape=polygonShape(vertices=[ (x / SCALE,y / SCALE) for x,y in TOOL_1 ]),
            density=DENSITY_2,
            friction=FRICTION,
            restitution=0.0)
            
        definedFixturesObj_2 = fixtureDef(shape=polygonShape(vertices=[ (x / SCALE,y / SCALE) for x,y in TOOL_2 ]),
            density=DENSITY_2,
            friction=FRICTION,
            restitution=0.0)

        definedFixturesObj_3 = fixtureDef(shape=polygonShape(vertices=[ (x / SCALE,y / SCALE) for x,y in TOOL_3 ]),
            density=DENSITY_2,
            friction=FRICTION,
            restitution=0.0)

        definedFixturesObj_4 = fixtureDef(shape=polygonShape(vertices=[ (x / SCALE,y / SCALE) for x,y in OBJ_4 ]),
            density=DENSITY_2,
            friction=FRICTION,
            restitution=0.0)

        # Combine into list for body creation
        fixturesListTool = [definedFixturesTool_1, definedFixturesTool_2, definedFixturesTool_3, definedFixturesTool_4]
        fixturesListObj = [definedFixturesObj_1, definedFixturesObj_2, definedFixturesObj_3, definedFixturesObj_4]

        # Extent of randomness in tool starting position
        # Position and angle are randomised. Object is not currently randomised. 

        # Randomisation level:
        # self.rand_level = 3 # select 1-3

        if self.task == 'PUSH':
            if self.rand_level == 1 or self.rand_level == 2:
                rand_scale_pos = 150 # Use approx 150
                rand_scale_angle = 0
                rand_scale_rot = 0
            elif self.rand_level == 3:
                rand_scale_angle = 60 # plus/minus 60 from vertical
                rand_scale_pos = 150
                rand_scale_rot = 1
        elif self.task == 'LIFT':
            if self.rand_level == 1:
                rand_scale_pos = 150 # Use approx 150
                rand_scale_rot = 0
                rand_scale_angle = 0
            elif self.rand_level == 2 or self.rand_level == 3:
                rand_scale_angle = 60
                rand_scale_pos = 150
                rand_scale_rot = 1

        rand_half_circle = math.radians(np.random.uniform(90 - rand_scale_angle,90+rand_scale_angle))
        #print(rand_half_circle)
        rand_circle_x = math.cos(rand_half_circle) * (rand_scale_pos)
        rand_circle_y = math.sin(rand_half_circle) * (rand_scale_pos)
        #print(rand_circle_x, rand_circle_y)
        rand_angle = np.random.uniform(0,rand_scale_rot*math.pi*2)-((rand_scale_rot * math.pi))
        #print("Rand angle is: ", rand_angle)
        self.tool = self.world.CreateDynamicBody(position = (((VIEWPORT_W / SCALE / 2)+(rand_circle_x/SCALE)), (initial_y/4.0 + SIZE/SCALE) + (rand_circle_y/SCALE)),
            angle=rand_angle,
            fixtures = fixturesListTool)

        self.object = self.world.CreateDynamicBody(position = (VIEWPORT_W / SCALE / 2, (initial_y/4.0 + SIZE/SCALE)),
            angle=0,
            angularDamping = 10,
            linearDamping = 1,
            fixtures = fixturesListObj)

        # Set colours
        self.tool.color1 = (0.5,0.4,0.9)
        self.tool.color2 = (0.5,0.4,0.9)
        self.object.color1 = (0.3,0.3,0.5)
        self.object.color2 = (0.3,0.3,0.5)

        # Create object for target, for rendering (move poly spec to init above)
        if self.task == 'PUSH':
            # Set random target position. 
            # Object is not randomised
            if self.rand_level == 1:
                target_rand_scale = 100
                target_rand_x = 150
                target_direction_rand = 100
            elif self.rand_level == 2 or self.rand_level == 3:
                target_rand_scale = 100
                target_rand_x = 150#np.random.uniform(target_rand_scale)+target_rand_scale/2
                target_direction_rand = np.random.uniform(100)
            
            if target_direction_rand >= 50:
                self.target_position = target_rand_x/SCALE
                #print("Right")
            else:
                self.target_position = -target_rand_x/SCALE
                #print("Left")

            self.target = self.world.CreateStaticBody(position = (VIEWPORT_W/(2*SCALE) + self.target_position,VIEWPORT_H/(6*SCALE)), shapes=polygonShape(vertices=[(x,y) for x,y in TARGET_POLY_1]))

        elif self.task == 'LIFT':
            self.target_position = 120/SCALE
            self.target = self.world.CreateStaticBody(position = (VIEWPORT_W/(2*SCALE),VIEWPORT_H/(4*SCALE)+self.target_position), shapes=polygonShape
                    (vertices=[(x,y) for x,y in TARGET_POLY_1]))
        
        self.target.fixtures[0].sensor = True # Setting as sensor prevents collisions
      
        self.target.color1 = (0.2,0.2,0.2)
        self.target.color2 = (0.3,0.3,0.3)

        self.drawlist = [self.tool] + [self.object] + [self.moon] + [self.target]

        # Get first step on reset. Ensure action that is presented here is of the correct format (number of actions, etc)
        if self.continuous:
            for i in range(10):
                self.initial_step()
            return self.step(np.array([0.0,0.0,0.0]) if self.rotation else np.array([0.0,0.0]))[0]
        else:
            for i in range(10):
                self.initial_step()
            return self.step(3)[0]

    def initial_step(self):
        self.object.ApplyForceToCenter((0, self.object.mass * GRAVITY), False)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

    def set_reward(self, reward_level):
        print("Setting reward to: ", reward_level)
        self.reward_level = reward_level

    def set_random(self, random_level):
        print("Setting random to: ", random_level)
        self.rand_level = random_level

    def set_reward_scale(self, reward_scale):
        print("Setting scaling to: ", reward_scale)
        self.reward_scale = reward_scale


    def step(self, action):
        
        self.step_count += 1

        # Velocity of tool is set to zero at start of each step
        self.tool.linearVelocity = (0,0)
        self.tool.angularVelocity = 0.0
        maxSpeed = 0.5 # Sets magnitude of movements

        # Apply gravity to object
        self.object.ApplyForceToCenter((0, self.object.mass * GRAVITY), False)
        
        ## TODO: TIDY THIS UP AND DECIDE ON BEST APPROACH FOR DISCRETE AND CONTINUOUS
        vel = self.tool.linearVelocity
        if not self.continuous:
            if action == 0:
                if vel.y > -maxSpeed:
                    self.tool.ApplyForceToCenter((0, -self.tool.mass * FORCE), True)
            elif action == 2:
                if vel.y < maxSpeed:
                    self.tool.ApplyForceToCenter((0, self.tool.mass * FORCE), True)
            elif action == 1:
                if vel.x > -maxSpeed:
                    self.tool.ApplyForceToCenter((-self.tool.mass * FORCE, 0), True)
            elif action == 3:
                if vel.x < maxSpeed:
                    self.tool.ApplyForceToCenter((self.tool.mass * FORCE, 0), True)
        else:
            if not self.rotation:
                self.tool.ApplyLinearImpulse((action[0]*maxSpeed*FPS,action[1]*maxSpeed*FPS),self.tool.worldCenter,True)
            else:
                self.tool.ApplyLinearImpulse((action[0]*maxSpeed*FPS,action[1]*maxSpeed*FPS),self.tool.worldCenter,True)
                self.tool.ApplyAngularImpulse((action[2]*maxSpeed*FPS/5),True)


        reward = 0
        done = False

        ## DEPRECATED (NOW DONE DIRECTLY IN TENSORFORCE): STEP SIMULATION 4/6 TIMES USING SAME ACTION

        for _ in range(1): # Change to range(x) to repeat action x times. Not necessary anymore.
            self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        
            pos = self.tool.position
            vel = self.tool.linearVelocity
            pos_2 = self.object.position
            vel_2 = self.object.linearVelocity
            rot = self.tool.angle
            rot_2 = self.object.angle
            vel_mag = np.linalg.norm(vel)

            ## TODO decide what to include in state vector
            # Save state
            # Tool position, tool angle, object position, object angle
            if self.task == 'PUSH':
                position_target = self.target_position+VIEWPORT_W/(2*SCALE)
                state = [(pos.x - pos_2.x),
                    (pos.y - pos_2.y),
                    (rot),
                    #(pos_2.y),
                    (rot_2),
                    (pos_2.x - position_target)]
            elif self.task == 'LIFT':
                # Target calulated 
                position_target = self.target.position.y
                state = [(pos.x - pos_2.x),
                    (pos.y - pos_2.y),
                    (rot),
                    #(pos_2.x),
                    (rot_2),
                    (pos_2.y - position_target)]
            #assert len(state) == 5

            # Calculate distance between tool and object
            self.distance = (pos.x - pos_2.x, pos.y-pos_2.y)
            distance_L2 = np.sqrt(self.distance[0]*self.distance[0] + self.distance[1]*self.distance[1])
        
            # Calculate distance between object and target position
            if self.task == 'PUSH':
                distance_to_target = np.abs(pos_2.x - position_target)
                #print("pos_2.x is: ", pos_2.x)
                #print("position_target is: ", position_target)
                #print("Distance to target is: ", distance_to_target)
            elif self.task == 'LIFT':
                distance_to_target = np.abs(pos_2.y - position_target)
            
            # Runs if first step
            if self.prev_distance_L2 == 0:
                self.prev_distance_L2 = distance_L2
                self.prev_position = distance_to_target
                self.initial_distance = distance_to_target
                self.initial_distance_L2 = distance_L2
                #print("INITIAL DISTANCE IS ", self.initial_distance_L2)

            # self.reward_level = 2

            #print("Distance to target = ", pos_2.y, position_target)
            ## Same calculation for lift and push
            # Calculate reward based on distance between object and target
            if self.reward_level >= 2:
                if self.reward_level == 2:
                    reward_scale_local = self.reward_scale
                elif self.reward_level == 3:
                    reward_scale_local = self.reward_scale - 1
                position_reward = -(distance_to_target - self.prev_position)
                position_reward_total = reward_scale_local*(position_reward/self.initial_distance)
                reward += position_reward_total

            
            # Calculate reward based on distance between tool and object. 
            # Reward is gained by moving closer to the object, but only up to a point. Moving very close to the object makes no difference to reward
            # print("DistanceL2 is: ", distance_L2)
            distance_threshold = 2.5
            if self.reward_level >= 3:
                if distance_L2 > distance_threshold:
                    location_reward = -(distance_L2 - self.prev_distance_L2)
                else:
                    location_reward = 0
                reward += 1*(location_reward/(self.initial_distance_L2-distance_threshold))

            ##### BASIC REWARDS FOR ALL REWARD LEVELS
            # End episode with failure if tool too far from object
            if distance_L2 > 6:
                done = True
                reward -= 1
                #print("Tool too far from object")
            # End episode with success if object close to target
            if distance_to_target < 0.4 and (self.object.awake == False or vel_mag < 0.01):
                done = True
                reward += 0
                #print("SUCCESS!")
            # End episode with failure if object too far from target
            if distance_to_target > 8:
                done = True
                reward -= 1
                #print("Object too far from target")

            ###########################################
            ## Collision based rewards
            # Deprecated for now
            self.prev_tool_on_object = self.tool_on_object
            self.prev_tool_on_table = self.tool_on_table
            self.prev_distance_L2 = distance_L2
            self.prev_position = distance_to_target

            #if self.tool_on_object == True and self.collidedThisStep == True:
            #    if self.collisionObjectImpulse > 20:
            #        reward -= 2
            #        print("Tool broken!",self.collisionObjectImpulse)
            #        #done = True
            #    self.collidedThisStep = False
            #elif self.tool_on_object == True and self.collidedThisStep == False:
            #    if self.collisionObjectImpulse > 20:
            #        reward -= 2
            #        print("Tool crushed: ", self.collisionObjectImpulse)
            #        #done = True

        # Return state etc
        return np.array(state, dtype=np.float32), reward, done, {}

    def render(self, mode='human'):
        if self.step_count % 5 == 0:
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
                self.viewer.set_bounds(0, VIEWPORT_W / SCALE, 0, VIEWPORT_H / SCALE)

            for p in self.sky_polys:
                self.viewer.draw_polygon(p, color=(0,0,0))

            for obj in self.drawlist:
                for f in obj.fixtures:
                    trans = f.body.transform
                    if type(f.shape) is circleShape:
                        t = rendering.Transform(translation=trans * f.shape.pos)
                        self.viewer.draw_circle(f.shape.radius, 20, color=obj.color1).add_attr(t)
                        self.viewer.draw_circle(f.shape.radius, 20, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                    else:
                        path = [trans * v for v in f.shape.vertices]
                        self.viewer.draw_polygon(path, color=obj.color1)
                        path.append(path[0])
                        self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)
            return self.viewer.render(return_rgb_array = mode == 'rgb_array')
        else:
            return 0

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

## Different modes
class EnvTestContinuous(EnvTest):
    continuous = True

class EnvTestContinuousR(EnvTest):
    continuous = True
    rotation = True

class EnvTestRotation(EnvTest):
    rotation = True


## Deprecated left over from lunar lander. Probably needs deleting
def heuristic(env, s):
    # Heuristic for:
    # 1.  Testing.
    # 2.  Demonstration rollout.
    angle_targ = s[0] * 0.5 + s[2] * 1.0         # angle should point towards center (s[0] is horizontal coordinate,
                                                 # s[2] hor speed)
    if angle_targ > 0.4: angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4: angle_targ = -0.4
    hover_targ = 0.55 * np.abs(s[0])           # target y should be proporional to horizontal offset

    # PID controller: s[4] angle, s[5] angularSpeed
    angle_todo = (angle_targ - s[4]) * 0.5 - (s[5]) * 1.0
    #print("angle_targ=%0.2f, angle_todo=%0.2f" % (angle_targ, angle_todo))

    # PID controller: s[1] vertical coordinate s[3] vertical speed
    hover_todo = (hover_targ - s[1]) * 0.5 - (s[3]) * 0.5
    #print("hover_targ=%0.2f, hover_todo=%0.2f" % (hover_targ, hover_todo))

    if s[6] or s[7]: # legs have contact
        angle_todo = 0
        hover_todo = -(s[3]) * 0.5  # override to reduce fall speed, that's all we need after contact

    if env.continuous:
        a = np.array([hover_todo * 20 - 1, -angle_todo * 20])
        a = np.clip(a, -1, +1)
    else:
        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > 0.05: a = 2
        elif angle_todo < -0.05: a = 3
        elif angle_todo > +0.05: a = 1
    return a

def demo_heuristic_lander(env, seed=None, render=False):
    env.seed(seed)
    total_reward = 0
    steps = 0
    s = env.reset()
    while True:
        a = heuristic(env, s)
        s, r, done, info = env.step(a)
        total_reward += r

        if render:
            still_open = env.render()
            if still_open == False: break

        if steps % 20 == 0 or done:
            print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        steps += 1
        if done: break
    return total_reward


if __name__ == '__main__':
    demo_heuristic_lander(EnvTest(), render=True)
