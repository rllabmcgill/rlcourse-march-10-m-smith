import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
#import seaborn




r = np.array([0,0,0,0])
v = np.array([[1,2,1,0],[1,0,1,2]]).T
p = np.array(
	[[1,5,6,7],
	 [5,11,9,4],
	 [6,9,6,6],
	 [7,4,6,2]]
	)

p = p / p.sum(axis=1)[:,np.newaxis]

d_last = np.zeros(p.shape)
d = np.copy(p)
for i in range(25):
	d = d.dot(p)
d = d[0]


alpha = 0.007
gamma = 0.8

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for j in range(1):
	xs = []
	ys = []
	zs = []
	xs2 = []
	ys2 = []
	zs2 = []
	i_s = np.random.choice(len(r))
	theta = np.array([-1.2363675,0.4256])#np.random.randn(2)
	theta2 = np.copy(theta)

	for i in range(1000):
		d1 = 0
		d2 = 0
		i_s = np.random.choice(len(r))

		xs.append(theta[0])
		ys.append(theta[1])
		zs.append(d.dot(v.dot(theta)**2))
		xs2.append(theta2[0])
		ys2.append(theta2[1])
		zs2.append(d.dot(v.dot(theta2)**2))

		for b in range(1):
			s =  np.random.choice(len(r), p=p[i_s])
			d1 += alpha * 0.5 * (r[s] + gamma*v[s].dot(theta) - v[i_s].dot(theta)) * v[i_s]
			d2 += alpha *  d[i_s]*(-v[i_s].dot(theta2)) * v[i_s]
			i_s = s

		theta += d1 / 1
		theta2 += d2 / 1

	amount = len(xs)
	#for i in range(len(xs)):
		
	for i in range(len(xs)):
		ax.plot(xs[i:i+2],ys[i:i+2],zs[i:i+2],color=[0.2, 0.0, 0.9])
		ax.plot(xs2[i:i+2],ys2[i:i+2],zs2[i:i+2],color=[1, 0.7, 0])
	"""
	for angle in range(0, 360):
	    ax.view_init(30, angle)
	    plt.draw()
	    plt.pause(.00001)
"""
X = np.arange(min(xs[0] - 0.2,-0.5), max(0.5, xs[0] + 0.2), 0.015)
Y = np.arange(min(ys[0] - 0.2,-0.5), max(0.5, ys[0] + 0.2), 0.015)
X, Y = np.meshgrid(X, Y)
Z = ((v[None,None,:,0] * X[:,:,None] + v[None,None,:,1] * Y[:,:,None])**2 * d[None,None,:]).sum(axis=2)
cmap = clrs.Colormap('cool', N=len(xs))
#surf = ax.plot_surface(X, Y, Z, linewidth=0, antialiased=False)
ax.plot_wireframe(X, Y, Z, color=[0.2,0.8,0.5])

for k in [0,200,400,600,900]:
	startx = xs2[k]
	starty = ys2[k]
	startz = d.dot(v.dot([startx,starty])**2)
	b = 2*(((-v.dot([startx,starty]))[:,np.newaxis] * v) * d[:,np.newaxis] ).sum(axis=0)
	b = b / np.abs(b).sum()
	c = 0.1
	endx = startx - b[0] * c
	endy = starty - b[1] * c
	endz = startz + b.dot(b) * c

	startx = startx + b[0] * c
	starty = starty + b[1] * c
	startz = startz + b.dot(-b) * c

	plt.plot([startx,endx], [starty,endy], [startz,endz],color='r')

	startx = xs[k]
	starty = ys[k]
	startz = d.dot(v.dot([startx,starty])**2)
	b = 2*(((-v.dot([startx,starty]))[:,np.newaxis] * v) * d[:,np.newaxis] ).sum(axis=0)

	b = b / np.abs(b).sum()

	endx = startx - b[0] * c
	endy = starty - b[1] * c
	endz = startz + b.dot(b) * c

	startx = startx + b[0] * c
	starty = starty + b[1] * c
	startz = startz + b.dot(-b) * c

	plt.plot([startx,endx], [starty,endy], [startz,endz],color='r')


plt.scatter([0], [0], [0])

angles = range(-50,50)
def update(i):
	ax.view_init(40, angles[i])
anim = FuncAnimation(fig, update, frames=np.arange(0, 20), interval=200)
anim.save('msve_td_sym.gif', dpi=80, writer='imagemagick')

plt.show()


