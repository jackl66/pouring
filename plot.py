import numpy as np
import matplotlib.pyplot as pt

def plotting(data=None,final_data_target=None):
    fig = pt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(331)
    ax1.set_title('Angle')
    ax2 = fig.add_subplot(332)
    ax2.set_title('Force')
    ax3 = fig.add_subplot(333)
    ax3.set_title('Initial')
    ax4 = fig.add_subplot(334)
    ax4.set_title('Target')
    ax5 = fig.add_subplot(335)
    ax5.set_title('Height')
    ax6 = fig.add_subplot(336)
    ax6.set_title('Kappa')
    ax7 = fig.add_subplot(313)
    ax7.set_title('Velocity')
    pt.tight_layout()

    pt.ion()

    data = np.load('./testing/number_of_blocks50_target2_result5_friction0.02_length0.025_spill3.npy')
    index=0
    t=0
    while index<len(data):
        # if index > 190:
        #     break
        inputs=data[index]
        # inputs = np.append(inputs, final_data_target[index,0])
        ax1.plot(t, inputs[0], 'k.', markersize=2)
        # t+=1
        ax2.plot(t, inputs[1], 'k.', markersize=2)
        # t += 1
        ax3.plot(t, inputs[2], 'k.', markersize=2)
        # t += 1
        ax4.plot(t, inputs[3], 'k.', markersize=2)
        # t += 1
        ax5.plot(t, inputs[4], 'k.', markersize=2)
        # t += 1
        ax6.plot(t, inputs[5], 'k.', markersize=2)
        # t += 1
        ax7.plot(t, inputs[6], 'k.', markersize=2)
        t += 1
        index+=1
        pt.draw()

    pt.ioff()
    pt.show()

def plot_ori(data):
    fig = pt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(331)
    ax1.set_title('Angle')
    ax2 = fig.add_subplot(332)
    ax2.set_title('Force')
    ax3 = fig.add_subplot(333)
    ax3.set_title('Initial')
    ax4 = fig.add_subplot(334)
    ax4.set_title('Target')
    ax5 = fig.add_subplot(335)
    ax5.set_title('Height')
    ax6 = fig.add_subplot(336)
    ax6.set_title('Kappa')
    ax7 = fig.add_subplot(313)
    ax7.set_title('Velocity')
    pt.tight_layout()

    pt.ion()


    index=0
    i=0
    while index<len(data):
        # if index > 190:
        #     break
        inputs=data[index]
        t=i/100
        ax1.plot(t, inputs[0], 'k.', markersize=2)
        # t+=1
        ax2.plot(t, inputs[1], 'k.', markersize=2)
        # t += 1
        ax3.plot(t, inputs[2], 'k.', markersize=2)
        # t += 1
        ax4.plot(t, inputs[3], 'k.', markersize=2)
        # t += 1
        ax5.plot(t, inputs[4], 'k.', markersize=2)
        # t += 1
        ax6.plot(t, inputs[5], 'k.', markersize=2)
        # t += 1
        ax7.plot(t, inputs[6], 'k.', markersize=2)
        i += 1
        index+=1
        pt.draw()

    pt.ioff()
    pt.show()

plotting()
# import time
#
# stat=time.perf_counter()
# time.sleep(1)
# check=time.perf_counter()
# print(check-stat)