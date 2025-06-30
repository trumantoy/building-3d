import wgpu
import pygfx as gfx
# import pybullet as bullet

import numpy as np
import pylinalg as la

class Scene(gfx.Scene):
    steps  = list()
    
    def __init__(self):
        super().__init__()
        # bid = bullet.connect(bullet.DIRECT)
        # bullet.setPhysicsEngineParameter(erp=1,contactERP=1,frictionERP=1)
        # bullet.setGravity(0, 0, -9.81)
        pass

    def step(self,dt=1/240):
        if self.steps: 
            if not self.steps[0](): return
            self.steps.pop(0)

        for entity in self.children:
            if 'step' not in dir(entity): continue 
            entity.step(dt)
