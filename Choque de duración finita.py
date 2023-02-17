import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from tqdm import tqdm
from time import sleep

class Particle:
    
    def __init__(self,r0,v0,a0,t,m,radius,Id=0):
        
        self.dt = t[1] - t[0]
        
        self.r = r0
        self.v = v0
        self.a = a0
        
        self.R = np.zeros((len(t),len(r0)))
        self.V = np.zeros_like(self.R)
        self.A = np.zeros_like(self.R)
        
        self.radius = radius
        self.m = m
           
    def SetPosition(self,i):
        self.R[i] = self.r
        
    def GetPosition(self,scale=1):
        return self.R[::scale]
    
    def SetVelocity(self,i):
        self.V[i] = self.v

    def GetVelocity(self,scale=1):
        return self.V[::scale]

    def CheckLimits(self,Limits):
        
        for i in range(2):
        
            if self.r[i] + self.radius > Limits[i][1] and self.v[i] > 0.:
                self.v[i] = -1.0*self.v[i]
            if self.r[i] - self.radius < Limits[i][0] and self.v[i] < 0.:
                self.v[i] = -1.0*self.v[i]
    
    def Evolution(self,i,force):
    
        self.SetPosition(i)
        self.SetVelocity(i)
        
        self.r += self.dt*self.v
        self.v += self.dt*self.a
        self.a += force/self.m

def Force(particle1, particle2):
        
        r1 = particle1.GetPosition()
        r2 = particle2.GetPosition()
        radius1 = particle1.radius
        radius2 = particle2.radius
        
        if np.linalg.norm(r1 - r2) < (radius1 + radius2):
            
            vector_direccion_plano = r1 - r2
            
            x = -vector_direccion_plano[1]
            
            y = vector_direccion_plano[0]
            
            vector_normal_plano = np.array([x,y])
            
            force = 1*(np.linalg.norm(r1 - r2)**3)*vector_normal_plano
            
            force = np.array([force[0][0], force[1][1]]) #CON ESTO LAS DIMENSIONES SON IGUALES
             
        else:
            
            force = np.array([0.,0.])
            
        return force

def RunSimulation1(t,Wall):
    
    r0 = np.array([-15.,1.])
    v0 = np.array([10.,0.])
    a0 = np.array([0.,0.])
    
    r1 = np.array([0.,-1.5])
    v1 = np.array([0.,0.])
    a1 = np.array([0.,0.])
    
    
    p1 = Particle(r0,v0,a0,t,1.,2.)
    p2 = Particle(r1,v1,a1,t,1.,2.)
    
    Wall_ = Wall.copy()
    
    
    for it in tqdm(range(len(t)), desc='Running simulation', unit=' Steps'):
        sleep(0.0001)
        force = Force(p1, p2)
        p1.Evolution(it,force)
        p1.CheckLimits(Wall_)
        p2.Evolution(it,force)
        p2.CheckLimits(Wall_)
    
    return p1,p2

Limits = np.array([[-30.,30.],[-30.,30.]])
dt = 0.05
tmax = 10
t = np.arange(0.,tmax,dt)
Particles = RunSimulation1(t,Limits)
scale = 1
t = t[::scale]
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)

def init():
    ax.set_xlim(Limits[0][0],Limits[0][1])
    ax.set_ylim(Limits[1][0],Limits[1][1])
    
def Update(i,Particles):
    
    ax.clear()
    init()
    ax.set_title(r'$ t=%.2f \ s$' %(t[i]))
    
    
    x1 = Particles[0].GetPosition(scale)[i,0]
    y1 = Particles[0].GetPosition(scale)[i,1]
    vx1 = Particles[0].GetVelocity(scale)[i,0]
    vy1 = Particles[0].GetVelocity(scale)[i,1]
    
    x2 = Particles[1].GetPosition(scale)[i,0]
    y2 = Particles[1].GetPosition(scale)[i,1]
    vx2 = Particles[1].GetVelocity(scale)[i,0]
    vy2 = Particles[1].GetVelocity(scale)[i,1]
    
    
    
    circle1 = plt.Circle((x1,y1),Particles[0].radius, fill=True, color='k')
    ax.add_patch(circle1)
    
    ax.arrow(x1,y1,vx1,vy1,color='r',head_width=0.2,length_includes_head=True)
    
    circle2 = plt.Circle((x2,y2),Particles[1].radius, fill=True, color='k')
    ax.add_patch(circle2)
    ax.arrow(x2,y2,vx2,vy2,color='r',head_width=0.2,length_includes_head=True)
    

Animation = anim.FuncAnimation(fig,Update,frames=len(t),init_func=init)

    
