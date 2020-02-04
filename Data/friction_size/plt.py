import numpy as np
import matplotlib.pyplot as plt

"""Double peg with different friction factor"""
fig = plt.figure(figsize=(30, 15))

file_name = 'Double_norand_f0_s1_'
count = 1
plt_data = np.load(file_name + str(count) + '.npy')
# plot force
plt.subplot(231)
plt.plot(plt_data[12][0], plt_data[6][0])
plt.subplot(232)
plt.plot(plt_data[12][0], plt_data[7][0])
plt.subplot(233)
plt.plot(plt_data[12][0], plt_data[8][0], label='Friction factor = 0')
plt.subplot(234)
plt.plot(plt_data[12][0], plt_data[9][0])
plt.subplot(235)
plt.plot(plt_data[12][0], plt_data[10][0])
plt.subplot(236)
plt.plot(plt_data[12][0], plt_data[11][0])

file_name = 'Double_rand_f0.5_s1_'
count = 1
plt_data = np.load(file_name + str(count) + '.npy')
# plot force
plt.subplot(231)
plt.plot(plt_data[12][0], plt_data[6][0])
plt.subplot(232)
plt.plot(plt_data[12][0], plt_data[7][0])
plt.subplot(233)
plt.plot(plt_data[12][0], plt_data[8][0], label='Friction factor = 0.5')
plt.subplot(234)
plt.plot(plt_data[12][0], plt_data[9][0])
plt.subplot(235)
plt.plot(plt_data[12][0], plt_data[10][0])
plt.subplot(236)
plt.plot(plt_data[12][0], plt_data[11][0])

file_name = 'Double_norand_f1_s1_'
count = 1
plt_data = np.load(file_name + str(count) + '.npy')
# plot force
plt.subplot(231)
plt.plot(plt_data[12][0], plt_data[6][0])
plt.ylabel('Fx/N', fontsize=32)
plt.subplot(232)
plt.plot(plt_data[12][0], plt_data[7][0])
plt.ylabel('Fy/N', fontsize=32)
plt.subplot(233)
plt.plot(plt_data[12][0], plt_data[8][0], label='Friction factor = 1')
plt.legend(loc=1, fontsize=20)
plt.ylabel('Fz/N', fontsize=32)
plt.subplot(234)
plt.plot(plt_data[12][0], plt_data[9][0])
plt.ylabel('Tx/Nm', fontsize=32)
plt.subplot(235)
plt.plot(plt_data[12][0], plt_data[10][0])
plt.ylabel('Ty/Nm', fontsize=32)
plt.subplot(236)
plt.plot(plt_data[12][0], plt_data[11][0])
plt.ylabel('Tz/Nm', fontsize=32)

fig.suptitle('Double peg with different friction factor\nClearance = 1', fontsize=32)
plt.savefig('./Double peg with different friction factor.jpg')
plt.show()

"""Double peg with different clearance"""
fig = plt.figure(figsize=(30, 15))

file_name = 'Double_rand_f0.5_s0.5_'
count = 0
plt_data = np.load(file_name + str(count) + '.npy')
# plot force
plt.subplot(231)
plt.plot(plt_data[12][0], plt_data[6][0])
plt.subplot(232)
plt.plot(plt_data[12][0], plt_data[7][0])
plt.subplot(233)
plt.plot(plt_data[12][0], plt_data[8][0], label='Peg-hole Clearance = 0.5')
plt.subplot(234)
plt.plot(plt_data[12][0], plt_data[9][0])
plt.subplot(235)
plt.plot(plt_data[12][0], plt_data[10][0])
plt.subplot(236)
plt.plot(plt_data[12][0], plt_data[11][0])

file_name = 'Double_rand_f0.5_s1_'
count = 1
plt_data = np.load(file_name + str(count) + '.npy')
# plot force
plt.subplot(231)
plt.plot(plt_data[12][0], plt_data[6][0])
plt.subplot(232)
plt.plot(plt_data[12][0], plt_data[7][0])
plt.subplot(233)
plt.plot(plt_data[12][0], plt_data[8][0], label='Peg-hole Clearance = 1')
plt.subplot(234)
plt.plot(plt_data[12][0], plt_data[9][0])
plt.subplot(235)
plt.plot(plt_data[12][0], plt_data[10][0])
plt.subplot(236)
plt.plot(plt_data[12][0], plt_data[11][0])

file_name = 'Double_rand_f0.5_s1.5_'
count = 2
plt_data = np.load(file_name + str(count) + '.npy')
# plot force
plt.subplot(231)
plt.plot(plt_data[12][0], plt_data[6][0])
plt.ylabel('Fx/N', fontsize=32)
plt.subplot(232)
plt.plot(plt_data[12][0], plt_data[7][0])
plt.ylabel('Fy/N', fontsize=32)
plt.subplot(233)
plt.plot(plt_data[12][0], plt_data[8][0], label='Peg-hole Clearance = 1.5')
plt.legend(loc=1, fontsize=20)
plt.ylabel('Fz/N', fontsize=32)
plt.subplot(234)
plt.plot(plt_data[12][0], plt_data[9][0])
plt.ylabel('Tx/Nm', fontsize=32)
plt.subplot(235)
plt.plot(plt_data[12][0], plt_data[10][0])
plt.ylabel('Ty/Nm', fontsize=32)
plt.subplot(236)
plt.plot(plt_data[12][0], plt_data[11][0])
plt.ylabel('Tz/Nm', fontsize=32)

fig.suptitle('Double peg with different clearance\nFriction = 0.5', fontsize=32)
plt.savefig('./Double peg with different clearance.jpg')
plt.show()

"""Double peg with different hole position"""
fig = plt.figure(figsize=(30, 15))

file_name = 'Double_rand_f0.5_s0.5_'
count = 0
plt_data = np.load(file_name + str(count) + '.npy')
# plot force
plt.subplot(231)
plt.plot(plt_data[12][0], plt_data[6][0])
plt.subplot(232)
plt.plot(plt_data[12][0], plt_data[7][0])
plt.subplot(233)
plt.plot(plt_data[12][0], plt_data[8][0], label='Hole Position 1')
plt.subplot(234)
plt.plot(plt_data[12][0], plt_data[9][0])
plt.subplot(235)
plt.plot(plt_data[12][0], plt_data[10][0])
plt.subplot(236)
plt.plot(plt_data[12][0], plt_data[11][0])

file_name = 'Double_rand_f0.5_s0.5_'
count = 1
plt_data = np.load(file_name + str(count) + '.npy')
# plot force
plt.subplot(231)
plt.plot(plt_data[12][0], plt_data[6][0])
plt.subplot(232)
plt.plot(plt_data[12][0], plt_data[7][0])
plt.subplot(233)
plt.plot(plt_data[12][0], plt_data[8][0], label='Hole Position 2')
plt.subplot(234)
plt.plot(plt_data[12][0], plt_data[9][0])
plt.subplot(235)
plt.plot(plt_data[12][0], plt_data[10][0])
plt.subplot(236)
plt.plot(plt_data[12][0], plt_data[11][0])

file_name = 'Double_rand_f0.5_s0.5_'
count = 3
plt_data = np.load(file_name + str(count) + '.npy')
# plot force
plt.subplot(231)
plt.plot(plt_data[12][0], plt_data[6][0])
plt.ylabel('Fx/N', fontsize=32)
plt.subplot(232)
plt.plot(plt_data[12][0], plt_data[7][0])
plt.ylabel('Fy/N', fontsize=32)
plt.subplot(233)
plt.plot(plt_data[12][0], plt_data[8][0], label='Hole Position 3')
plt.legend(loc=1, fontsize=20)
plt.ylabel('Fz/N', fontsize=32)
plt.subplot(234)
plt.plot(plt_data[12][0], plt_data[9][0])
plt.ylabel('Tx/Nm', fontsize=32)
plt.subplot(235)
plt.plot(plt_data[12][0], plt_data[10][0])
plt.ylabel('Ty/Nm', fontsize=32)
plt.subplot(236)
plt.plot(plt_data[12][0], plt_data[11][0])
plt.ylabel('Tz/Nm', fontsize=32)

fig.suptitle('Double peg with different hole position\nClearance = 0.5; Friction = 0.5', fontsize=32)
plt.savefig('./Double peg with different hole position.jpg')
plt.show()