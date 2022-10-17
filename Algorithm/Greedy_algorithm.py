
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import scipy.interpolate
import itertools


class greedy_algorithm_main:
    def __init__(self, final_array, angle_constraint=10, step_back=3):
        self.prod_map = final_array
        self.angle_constraint = angle_constraint*step_back/(10)
        self.step_x = 1
        self.step_back = step_back

    def tangent_func(self, traj, xVal):
        x = np.arange(0, len(traj))
        interp = greedy_algorithm_main.traj_interpolation
        degrees = [3, 4, 5, 6]
        r2_score_base = 0
        for i in degrees:
            y = interp(traj,degree=i)
            r2_check = r2_score(traj, y)
            if r2_check > r2_score_base:
                r2_score_base = r2_check
                degr = i
        try:
            y = interp(traj, degree=degr)
        except:
            y = interp(traj, degree=4)
        # calculate gradient
        slope = np.gradient(y, x)

        # determine slope and intercept at xVal
        ind1 = (np.abs(x - xVal)).argmin()
        # case 1 the value is a data point
        if xVal == x[ind1]:
            yVal, slopeVal = y[ind1], slope[ind1]
            # case 2 the value lies between to data points
        # in which case we approximate linearly from the two nearest data points
        else:
            if xVal < x[ind1]:
                ind1, ind2 = ind1- 1, ind1
            else:
                ind1, ind2 = ind1, ind1 + 1
            yVal = y[ind1] + (y[ind2 ] - y[ind1]) * (xVal - x[ind1]) / (x[ind2 ] - x[ind1])
            slopeVal = slope[ind1] + (slope[ind2 ] - slope[ind1]) * (xVal - x[ind1]) / (x[ind2 ] - x[ind1])
        intercVal = yVal - slopeVal * xVal

        x_val = [x.min(), x.max()]
        y_val = [slopeVal * x.min( ) + intercVal, slopeVal * x.max( ) + intercVal]

        return x_val, y_val

    def deg_calc(self, traj):
        x_val_1 ,y_val_1 = self.tangent_func(traj, len(traj ) - 1)
        x_val_2 ,y_val_2 = self.tangent_func(traj, len(traj ) - self.step_back)

        vector_1 = np.array(y_val_1)
        vector_2 = np.array(y_val_2)

        dot_prod = np.dot(vector_1, vector_2)
        unit_vector_1 = np.linalg.norm(vector_1)
        unit_vector_2 = np.linalg.norm(vector_2)
        cos_angle = min(dot_prod/(unit_vector_1 * unit_vector_2), 1)
        deg = np.degrees(np.arccos(cos_angle))
        return deg


    def greedy(self, start_point = [0, 40], step_y = 10, steps_ahead = 5):
        traj = np.zeros_like(self.prod_map)

        traj[start_point[0], start_point[1]] = 1
        next_point = start_point
        traj_points = [start_point[1]]
        dH = 1
        obj_func_val = 0
        pass_count = 0
        k = 0
        items = [0, 1, -1]
        all_candidates = []
        for item in itertools.product(items, repeat=steps_ahead):
            all_candidates.append(list(item))


    #   obtain OFV of all possible candidates

        for i in range (1, self.prod_map.shape[0] - start_point[0]):

            # upper boundary constraint
            if next_point[1] >= self.prod_map.shape[1] - self.step_x:
                next_point = [next_point[0] + self.step_x, next_point[1] - step_y]
                traj_points.append(next_point[1])
                continue

            # lower boundary constraint
            if next_point[1] < step_y:
                next_point = [next_point[0] + self.step_x, next_point[1]]
                traj_points.append(next_point[1])
                continue

            best_candidate = all_candidates[0]
            cand_point = next_point
            OFV_best = 0
            if next_point[1] >= self.prod_map.shape[1] or next_point[0] >= self.prod_map.shape[0]:
                greedy_simple = True:
            else:
                for l in range(0, len(all_candidates) + 1):
                    OFV = 0
                    for v in range(0, len(all_candidates[1])):
                        cand_point = [cand_point[0] + 1, cand_point[1] + all_candidates[l][v]]
                        if l == 0:
                            OFV_best += self.prod_map[cand_point[0], cand_point[1]]
                        else:
                            OFV += self.prod_map[cand_point[0], cand_point[1]]

                    if OFV > OFV_best:
                        best_candidate = all_candidates[l]
                        OFV_best = OFV


            # calculation of angle constraint
            if i > 10:
                # print(next_point[0])
                deg = self.deg_calc(traj_points)

                if deg >= self.angle_constraint:
                    next_point = [next_point[0] + self.step_x, next_point[1]]
                    traj_points.append(next_point[1])
                    traj[next_point[0], next_point[1]] = 1
                    continue

            next_point = next_point[next_point[0] + self.step_x, next_point[1] + best_candidate[0]]

            #greedy main
            if greedy_simple:
                if (self.prod_map[next_point[0], next_point[1] + step_y] / (np.sqrt(2) * dH) > self.prod_map[
                    next_point[0], next_point[1] + 0]) and \
                        (self.prod_map[next_point[0], next_point[1] + step_y] / (np.sqrt(2) * dH) > self.prod_map[
                            next_point[0], next_point[1] - step_y] / (np.sqrt(2) * dH)):
                    if next_point[1] > self.prod_map.shape[1]:
                        next_point = [next_point[0] + self.step_x, next_point[1]]
                    next_point = [next_point[0] + self.step_x, next_point[1] + step_y]
                elif (self.prod_map[next_point[0], next_point[1] + 0] > self.prod_map[
                    next_point[0], next_point[1] + step_y] / (np.sqrt(2) * dH)) and \
                        (self.prod_map[next_point[0], next_point[1] + 0] > self.prod_map[
                            next_point[0], next_point[1] - step_y] / (np.sqrt(2) * dH)):
                    next_point = [next_point[0] + self.step_x, next_point[1] + 0]
                else:
                    next_point = [next_point[0] + self.step_x, next_point[1] - step_y]

            traj_points.append(next_point[1])
            # print(len(traj_points),i)
            obj_func_val += self.prod_map[next_point[0], next_point[1]]
            traj[next_point[0], next_point[1]] = 1

        print('OFV =', np.round(obj_func_val, 2))
        return traj_points

    @staticmethod
    def traj_interpolation(trajectory, degree=6):
        # curve interpolation
        d = np.polyfit(np.arange(0, len(trajectory)), trajectory, deg=degree)
        z = np.poly1d(d)
        dt = z(np.arange(0, len(trajectory)))
        return dt

    @staticmethod
    def visualization(array, *trajectory, interpolation=True, scale_x, scale_y):
        fig, ax = plt.subplots()
        fig.set_size_inches(20, 6)
        plt.gca().invert_yaxis()
        plt.title('Productivity potential map')
        plt.xlabel('Distance *%1.0f [m]' % scale_x)
        plt.ylabel('Depth 1/%1.0f [m]' % scale_y)

        p_map = plt.imshow(array, cmap='cividis_r', aspect='auto')
        bar = plt.colorbar(p_map)
        for traj in trajectory:
            plt.plot(traj, linewidth=3)
            if interpolation:
                dt = greedy_algorithm_main.traj_interpolation(traj)
                plt.plot(dt, linewidth=3)
        plt.show()

    @staticmethod
    def curve_length(curve):
        """ sum of Euclidean distances between points """
        return np.sum(np.sqrt(np.sum((curve[:-1] - curve[1:]) ** 2, axis=0)))



# constraint based on radius-vectors

#   vector_1 = np.array([traj_points[0],traj_points[i]])
#   vector_2 = np.array([traj_points[0],traj_points[i-1]])

#   dot_prod = np.dot(vector_1,vector_2)
#   unit_vector_1 = np.linalg.norm(vector_1)
#   unit_vector_2 = np.linalg.norm(vector_2)
#   cos_angle = min(dot_prod/(unit_vector_1*unit_vector_2),1)
#   deg = np.degrees(np.arccos(cos_angle))
#   if np.isnan(deg):

#     print(vector_1)
#     print(vector_2)
#     print(cos_angle)
#   print(deg)
#   if deg >= self.angle_constraint:
#     next_point = [next_point[0]+1,next_point[1]]
#     #print('angle constraint initiated')
#     traj_points.append(next_point[1])
#     continue