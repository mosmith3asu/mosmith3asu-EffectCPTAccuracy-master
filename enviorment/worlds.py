
import numpy as np

WORLDS = {}

_ = 0 # empty
B = 1 # border
p = 2 # penalty

empty_world = np.array([
    [B,B,B,B,B,B,B],
    [B,_,_,_,_,_,B],
    [B,_,B,_,B,_,B],
    [B,_,_,_,_,_,B],
    [B,_,B,_,B,_,B],
    [B,_,_,_,_,_,B],
    [B,B,B,B,B,B,B]])


practice_startPos = [(100, 100), (100, 500), (500, 300)]
practice_world = np.array([
    [B,B,B,B,B,B,B],
    [B,_,_,p,_,p,B],
    [B,p,B,_,B,_,B],
    [B,_,_,p,_,_,B],
    [B,p,B,_,B,_,B],
    [B,_,_,p,_,p,B],
    [B,B,B,B,B,B,B],])


startPos1 = [(100,500),(500,400),(300,300)]
world1 = np.array([
    [B,B,B,B,B,B,B],
    [B,p,_,_,_,_,B],
    [B,_,B,p,B,_,B],
    [B,_,_,_,_,_,B],
    [B,_,B,p,B,_,B],
    [B,_,_,_,_,p,B],
    [B,B,B,B,B,B,B]])

startPos2 = [(100, 100), (500, 500), (100, 500)]
world2 = np.array([
    [B,B,B,B,B,B,B],
    [B,_,p,_,_,_,B],
    [B,_,B,_,B,_,B],
    [B,_,_,_,_,_,B],
    [B,_,B,_,B,_,B],
    [B,_,p,_,p,_,B],
    [B,B,B,B,B,B,B],])

startPos3 = [(100, 400),(300, 500), (500, 500)]
world3 = np.array([
    [B,B,B,B,B,B,B],
    [B,p,p,_,_,_,B],
    [B,p,B,_,B,_,B],
    [B,_,_,_,_,_,B],
    [B,_,B,_,B,p,B],
    [B,_,_,_,p,p,B],
    [B,B,B,B,B,B,B],])

startPos4 = [(200, 500), (400, 500), (300, 100)]
world4 = np.array([
    [B,B,B,B,B,B,B],
    [B,_,_,_,_,p,B],
    [B,p,B,_,B,_,B],
    [B,_,_,_,p,_,B],
    [B,_,B,_,B,_,B],
    [B,_,_,p,_,_,B],
    [B,B,B,B,B,B,B],])

startPos5 = [(100, 400), (300, 500), (300, 300)]
world5 = np.array([
    [B,B,B,B,B,B,B],
    [B,p,_,_,_,_,B],
    [B,p,B,_,B,_,B],
    [B,_,_,_,_,_,B],
    [B,_,B,p,B,_,B],
    [B,_,_,_,_,p,B],
    [B,B,B,B,B,B,B],])

startPos6 = [(100, 200), (500, 200), (300, 200)]
world6 = np.array([
    [B,B,B,B,B,B,B],
    [B,p,_,_,_,_,B],
    [B,_,B,_,B,_,B],
    [B,_,_,_,_,p,B],
    [B,_,B,_,B,_,B],
    [B,p,p,_,p,p,B],
    [B,B,B,B,B,B,B],])


startPos7 = [(300, 200), (100, 100), (400, 100)]
world7 = np.array([
    [B,B,B,B,B,B,B],
    [B,_,p,p,p,_,B],
    [B,_,B,_,B,_,B],
    [B,_,_,_,_,_,B],
    [B,_,B,_,B,_,B],
    [B,_,_,_,_,_,B],
    [B,B,B,B,B,B,B],])


WORLDS['empty_val'] = _
WORLDS['pen_val'] = p
WORLDS['border_val'] = B
WORLDS['empty_world'] = {'array':empty_world,'start':None}
WORLDS['practice_world'] = {'array':empty_world,'start':practice_startPos}
WORLDS[1] = {'array':world1,'start':startPos1}
WORLDS[2] = {'array':world2,'start':startPos2}
WORLDS[3] = {'array':world3,'start':startPos3}
WORLDS[4] = {'array':world4,'start':startPos4}
WORLDS[5] = {'array':world5,'start':startPos5}
WORLDS[6] = {'array':world6,'start':startPos6}
WORLDS[7] = {'array':world7,'start':startPos7}






# tile_width, tile_height = 100, 100
# world_sz = np.shape(world)
# world_params = {
#     'world_array': world,
#     'height': world_sz[0] * tile_height,
#     'width': world_sz[1] * tile_width,
#     'tile_width': tile_width,
#     'tile_height': tile_height,
#     'bg_color': (255, 255, 255),
#     'penalty_color': (255, 200, 200),
#     'boundary_color': (0, 0, 0),
#     'empty_color': (255, 255, 255),
#     'boundary_val': B,
#     'empty_val': _,
#     'penalty_val': p,
#     'caption': "Client",
# }


#
# class World():
#     def __init__(self, world_parameters = None):
#         import pygame
#         self.pygame = pygame
#         world_parameters = world_params if world_parameters is None else world_parameters
#         self.width =  world_parameters['width']
#         self.height =  world_parameters['height']
#         self.tile_width =  world_parameters['tile_width']
#         self.tile_height =  world_parameters['tile_height']
#
#         self.array = world_parameters['world_array']
#
#         self.empty_color = world_parameters['empty_color']
#         self.boundary_color = world_parameters['boundary_color']
#         self.penalty_color = world_parameters['penalty_color']
#         self.colors = [self.empty_color,self.boundary_color,self.penalty_color]
#
#         self.empty_val = world_parameters['empty_val']
#         self.boundary_val = world_parameters['boundary_val']
#         self.penalty_val = world_parameters['penalty_val']
#
#         self.evader_timer_loc = [0,0]
#         self.pursuer_timer_loc = [0, 0]
#
#     def draw(self,win,timers=None):
#
#         # Draw Tiles
#         for ix,x in enumerate(range(0,self.width, self.tile_width)):
#             for iy,y in enumerate(range(0, self.height,self.tile_width)):
#                 rect = self.pygame.Rect(x, y, self.width, self.height)
#                 val = self.array[iy,ix]
#                 self.pygame.draw.rect(win, self.colors[val], rect)
#
#         # Draw Turn Timers
#         timers = np.array([0,0]) if timers is None else timers
#

#
#
# class World():
#     def __init__(self,window_params):
#         self.width =  window_params['width']
#         self.height =  window_params['height']
#         self.tile_width =  window_params['tile_width']
#         self.tile_height =  window_params['tile_height']
#
#         self.array = window_params['world_array']
#
#         self.empty_color = window_params['empty_color']
#         self.boundary_color = window_params['boundary_color']
#         self.penalty_color = window_params['penalty_color']
#         self.colors = [self.empty_color,self.boundary_color,self.penalty_color]
#
#         self.empty_val = window_params['empty_val']
#         self.boundary_val = window_params['boundary_val']
#         self.penalty_val = window_params['penalty_val']
#
#         self.evader_timer_loc = [0,0]
#         self.pursuer_timer_loc = [0, 0]
#
#     def draw(self,win,timers=None):
#
#         # Draw Tiles
#         for ix,x in enumerate(range(0,self.width, self.tile_width)):
#             for iy,y in enumerate(range(0, self.height,self.tile_width)):
#                 rect = pygame.Rect(x, y, self.width, self.height)
#                 val = self.array[iy,ix]
#                 pygame.draw.rect(win, self.colors[val], rect)
#
#         # Draw Turn Timers
#         timers = np.array([0,0]) if timers is None else timers
#

