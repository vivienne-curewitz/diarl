import numpy as np
import math
from time import perf_counter
from os import listdir
from os.path import isfile, join

def get_data_indices(file_path):
    files = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    indices = []
    for file in files:
        if "pls" in file:
            name = file.split(".")[0]
            indices.append(int(name[3:]))
    indices.sort()
    return indices

def load_csv_data(file_path):
    with open(file_path, "r") as pls:
        line1 = pls.readline()
        data = pls.readline().replace("{", "").replace("}", "").split(",")
        points = []
        for i in range(0, len(data)-3, 3):
            points.append((float(data[i]), float(data[i+1])))
        return points

def gama_to_points(gama: str):
    data = gama.replace("{", "").replace("}", "").replace("[", "").replace("]", "").split(",")
    # print(data)
    points = []
    for i in range(0, len(data)-3, 3):
        points.append((float(data[i]), float(data[i+1])))
    return points


def points_to_heatmap(current_position, points):
    px = current_position[0]
    py = current_position[1]
    heatmap = np.zeros((11, 11))
    min_x = math.floor(px - 5)
    min_y = math.floor(py - 5)
    start = perf_counter()
    for p in points:
        if p[0] >= min_x and p[0] <= min_x+10 and p[1] >= min_y and p[1] <= min_y+10:
            hx = math.floor(p[0] - min_x)
            hy = math.floor(p[1] - min_y)
            heatmap[hx][hy] += 1
    # print(f"Heatmap gen time: {perf_counter() - start}s")
    return heatmap


class env_sim:
    def __init__(self, dir, tpos, cpos):
        self.dir = dir
        if dir != "":
            self.indices = get_data_indices(dir)
        self.step = 0
        self.fname = "pls"
        self.points = []
        self.tpos = tpos
        self.cpos = cpos
        self.all_previous_positions = []
        self.width = 100
        self.height = 100
        self.max_dist = math.hypot(50, 50)
        self.prev_action = (0, 0)
        self.num_people = 200
    
    def next(self, step=True):
        fname = self.fname + str(self.indices[self.step]) + ".csv"
        path = join(self.dir, fname)
        self.points = load_csv_data(path)
        hm = points_to_heatmap(self.cpos, self.points)/self.num_people
        if step:
            self.step += 1
        return hm

    def get_env(self, step=False):
        hm = self.next(step=step).flatten()
        # # hm = np.zeros((121,)).flatten() 
        self.all_previous_positions.append((self.cpos[0], self.cpos[1]))
        tx, ty = self.tpos
        cx, cy = self.cpos
        # dist to target state
        dx, dy = (tx - cx, ty - cy)
        d = math.hypot(dx, dy)
        d_pow = d/self.max_dist
        dx /= max(d, 1.0)
        dy /= max(d, 1.0)
        # boundary state
        dleft = cx/self.width
        dright = (self.width - cx)/self.width
        dup = (self.height - cy)/self.height
        ddown = cy/self.height
        px, py = self.prev_action
        state_vec = [dx, dy, d_pow, dleft, dright, dup, ddown, px, py]
        state_vec.extend(hm)
        return state_vec 

    def get_env_one_shot(self, tpos, cpos, gama_str):
        points = gama_to_points(gama_str)
        tx, ty = tpos
        cx, cy = cpos
        hm = points_to_heatmap(cpos, points).flatten()
        # dist to target state
        dx, dy = (tx - cx, ty - cy)
        d = math.hypot(dx, dy)
        d_pow = d/self.max_dist
        dx /= max(d, 1.0)
        dy /= max(d, 1.0)
        # boundary state
        dleft = cx/self.width
        dright = (self.width - cx)/self.width
        dup = (self.height - cy)/self.height
        ddown = cy/self.height
        px, py = self.prev_action
        state_vec = [dx, dy, d_pow, dleft, dright, dup, ddown, px, py]
        state_vec.extend(hm)
        return state_vec 


    def step_env(self, delta):
        reward, complete = self.reward_func(delta)
        self.cpos = (self.cpos[0] + delta[0], self.cpos[1] + delta[1])
        state_vec = self.get_env(step=True) 
        return reward, state_vec, complete

    def reward_func(self, delta):
        cx, cy = self.cpos
        tx, ty = self.tpos 
        nx = cx + delta[0]
        ny = cy + delta[1]

        # boundary
        if cx < 0 or cy < 0 or cx > 100 or cy > 100:
            return -10000.0, True 

        # hit the target
        t_dist = math.hypot(nx - tx, ny - ty)
        if t_dist < 2.0:
            return 300.0, True
        # t delta reward
        prev_t_dist = math.hypot(cx - tx, cy - ty)
        tdr = -1*(t_dist - prev_t_dist)

        # count people
        crowd_penalty = 0.0
        for px, py in self.points:
            d = math.hypot(cx - px, cy - py)
            if d < 2.0: # covid rules
                crowd_penalty -= 1 * (2.0 - d) / 2.0  # linear penalty
        # time penalty
        step_penalty = -0.01
        # return all
        reward = tdr + crowd_penalty + step_penalty
        return reward, False

 
if __name__ == "__main__":
    # points = load_csv_data("../dia-homeworks/project/models/pls0.csv")
    # current_pos = (5, 18)
    # hm = points_to_heatmap(current_pos, points)
    # print(hm)
    # indices = get_data_indices("../dia-homeworks/project/models/data")
    # print(indices)
    current_pos = (25, 25)
    rf = env_sim("../dia-homeworks/project/models/data", (100, 100), (1, 1))
    for _ in range(100):
        r, sv, c = rf.step_env((1, 1))
        print(sv)