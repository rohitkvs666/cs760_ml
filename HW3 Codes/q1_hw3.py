import matplotlib.pyplot as plt
import numpy as np

def label_1NN(f1, f2):
    data = np.loadtxt("C:\\Users\\kvsr\\Desktop\\ECE 760 ML\\hw3-1\\data\\D2z.txt")
    
    min_dist = 10000000
    for i in range(len(data)):
        curr_dist = ((f1 - data[i][0])**2 + (f2 - data[i][1])**2)**0.5
        if curr_dist < min_dist:
            min_dist = curr_dist
            label_to_return = data[i][2]
    
    return label_to_return

points_in_plane_and_their_1NN_label = []
for x in np.arange(-2, 2.1, 0.1):
    for y in np.arange(-2, 2.1, 0.1):
        label_of_xy = label_1NN(x, y)
        points_in_plane_and_their_1NN_label.append([x, y, label_of_xy])

points_in_plane_with_label0 = []
points_in_plane_with_label1 = []
for k in range(len(points_in_plane_and_their_1NN_label)):
    if(points_in_plane_and_their_1NN_label[k][2] == 0):
        points_in_plane_with_label0.append(points_in_plane_and_their_1NN_label[k])
    else:
        points_in_plane_with_label1.append(points_in_plane_and_their_1NN_label[k])

x_label0_list = [m[0] for m in points_in_plane_with_label0]
y_label0_list = [n[1] for n in points_in_plane_with_label0]

x_label1_list = [m[0] for m in points_in_plane_with_label1]
y_label1_list = [n[1] for n in points_in_plane_with_label1]

data_given = np.loadtxt("C:\\Users\\yashw\\Documents\\CS760\\P3\\hw3-1\\data\\D2z.txt")

x_plot = data_given[:, [0, 1]]
y_plot = data_given[:, -1].astype(int)

plt.scatter(x_label0_list, y_label0_list, s=1, c='b')
plt.scatter(x_label1_list, y_label1_list, s=1, c='r')

plt.scatter(x_plot[:,0][y_plot==0], x_plot[:,1][y_plot==0], s=7, marker="s", c='orange')
plt.scatter(x_plot[:,0][y_plot==1], x_plot[:,1][y_plot==1], s=7, marker="^", c='black')

plt.show()

