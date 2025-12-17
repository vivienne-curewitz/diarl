import numpy as np
from read_csv import env_sim 
import math
from time import perf_counter
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from random import randint, random


import torch
import torch.nn as nn
import torch.nn.functional as F

from multiprocessing import Process, Queue

class RLPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(130, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, state):
        raw = self.net(state)
        return raw
        

class RLValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(130, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.net(state)


def load_model():
    agent = RLPolicy()
    checkpoint = torch.load("checkpoint_c2.pt")
    agent.load_state_dict(checkpoint["policy_state_dict"])
    return agent



def rand_unit_cirlce_pos(coords, distance):
    x, y = coords
    theta = random()*2*math.pi
    sx = distance*math.cos(theta) + x
    sy = distance*math.sin(theta) + y
    return (sx, sy)


def train_loop(data_queue: Queue, curriculum=1):
    wr_size = 200
    agent = RLPolicy()
    critic = RLValue()
    policy_opt = torch.optim.Adam(agent.parameters(), lr=1e-4)
    value_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
    total_steps = 0
    epoch = 0    
    last_n = np.zeros((wr_size,)).tolist() 
    current_start_dist = 3
    batch_size = 10
    try:
        checkpoint = torch.load("checkpoint.pt")
        agent.load_state_dict(checkpoint["policy_state_dict"])
        critic.load_state_dict(checkpoint["value_state_dict"])
        policy_opt.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        value_opt.load_state_dict(checkpoint["value_optimizer_state_dict"])
        total_steps = checkpoint["global_step"]
        epoch = checkpoint["epoch"]
        if "spawn_dist" in checkpoint.keys():
            current_start_dist = checkpoint["spawn_dist"]
        if current_start_dist * 1.1 > 50:
            current_start_dist = 10
    except Exception as e:
        print(f"Failed to load; restarting training {e}")
    states, actions, rewards, log_probs, values, entropies = [], [], [], [], [], []
    c3_epochs = 0
    # epoc init
    while(True):
        if c3_epochs > 2000 and curriculum == 3:
            break
        c3_epochs += 1
        if current_start_dist > 50:
            print(f"Training Complete -- Success Dist: {current_start_dist*(1/1.1)}")
            break
        total_reward = 0
        start = perf_counter()
        if curriculum == 1:
            tx, ty = (50, 50) # (randint(10, 90), randint(10, 90)) #target position
            cx, cy = rand_unit_cirlce_pos((tx, ty), current_start_dist) # current position
        elif curriculum == 2:
            tx, ty = (randint(10, 90), randint(10, 90))
            cx, cy = (randint(10, 90), randint(10, 90))
        elif curriculum == 3:
            tx, ty = (50, 50) # (randint(10, 90), randint(10, 90)) #target position
            current_start_dist = 45
            # cx, cy = (10, 65)
            cx, cy = rand_unit_cirlce_pos((tx, ty), current_start_dist) # current position
        if curriculum == 1 or curriculum == 2:
            folders = [
                "data_diagonal",
                "data_uniform_random"
            ]
        elif curriculum == 3:
            folders = [
                "data_circle",
                "data_diagonal",
                "data_uniform_random",
                "data_rectangle",
            ]
        fr = folders[randint(0, len(folders)-1)]
        rf = env_sim(f"../dia-homeworks/project/models/{fr}", (tx, ty), (cx, cy))
        state_vec = rf.get_env() 
        succeed = False
        sigma = 0.05
        gamma = 0.99
        entropy_coef = 0.01
        max_grad_norm = 0.5
        # buffers

        for i in range(2000):
            #loop
            state = torch.tensor(state_vec, dtype=torch.float32)

            # action = agent(state)
            # action += sigma * torch.randn_like(action)
            # action = F.normalize(action, dim=-1)

            mu = agent(state)
            dist = torch.distributions.Normal(mu, sigma)
            action = dist.sample()
            action = F.normalize(action, dim=-1)
            log_prob = dist.log_prob(action).sum()
            entropy = dist.entropy().sum() # <--- NEW: Calculate entropy
            # action here
            dx, dy = action.detach().numpy()
            # calculate step 
            reward, next_state_vec, complete = rf.step_env((dx, dy))
            if complete and reward > 0:
                succeed = True
            # get data for this step
            states.append(state.squeeze(0))
            actions.append(action.squeeze(0))
            rewards.append(reward)
            log_probs.append(dist.log_prob(action).sum())
            values.append(critic(state))
            entropies.append(dist.entropy().sum())

            state_vec = next_state_vec
            
            # do learning for this epoch
            if i > 0 and i%batch_size == 0:
                # data fixing
                states = torch.stack(states)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                log_probs = torch.stack(log_probs)
                values = torch.stack(values).squeeze()
                entropies = torch.stack(entropies)

                returns = []
                # Calculate discounted returns (R_t = r_t + gamma * R_{t+1})
                R = 0.0
                for r in reversed(rewards):
                    R = r + gamma * R
                    returns.insert(0, R)
                    
                returns = torch.tensor(returns, dtype=torch.float32)
                
                # generate advantage and normalzie
                advantage = returns - values.detach() # .detach() here is important
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                # Batch policy update
                # Policy Loss: Maximize returns and entropy (negative sign)
                policy_loss = - (log_probs * advantage.detach()).mean() - (entropy_coef * entropies.mean())
                
                policy_opt.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=max_grad_norm) # KEEP CLIPPING
                policy_opt.step()
                
                # Value Loss: Minimize mean squared error of the returns vs predicted values
                value_loss = (returns.detach() - values).pow(2).mean() # Use .mean() over the batch
                
                value_opt.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=max_grad_norm) # KEEP CLIPPING
                value_opt.step()
                # reset data
                states, actions, rewards, log_probs, values, entropies = [], [], [], [], [], []

            if complete:
                break

        rt = perf_counter() - start
        total_steps += i
        cx, cy = rf.cpos
        if succeed:
            last_n[epoch%wr_size] = 1
        else:
            last_n[epoch%wr_size] = 0
        print(f"TS: {total_steps} -- Epoch: {epoch} -- Average Reward: {total_reward/(i+1e-8):.4f} -- Runtime: {rt:.2f}s Win Rate {100*sum(last_n)/wr_size}% Spawn Distance {current_start_dist:.3f}")
        data_queue.put((rf.all_previous_positions, (tx, ty)))
        win_rate = sum(last_n)/wr_size
        if win_rate*100 > 90.0:
            current_start_dist *= 1.1 
            last_n = np.zeros((wr_size,)).tolist()
            torch.save({
                "policy_state_dict": agent.state_dict(),
                "value_state_dict": critic.state_dict(),
                "policy_optimizer_state_dict": policy_opt.state_dict(),
                "value_optimizer_state_dict": value_opt.state_dict(),
                "global_step": total_steps,
                "epoch": epoch,
                "spawn_dist": current_start_dist
            }, "checkpoint.pt")
        epoch += 1        


def run_train_and_view_procesess():
    dq = Queue()
    tp = Process(target=train_loop, args=(dq,3), daemon=True)
    tp.start()


    fig, ax = plt.subplots()
    line, = ax.plot([], [], linewidth=2)
    start_point = ax.scatter([], [], c='blue', label='start', s=100)
    end_point = ax.scatter([], [], c='red', label='end', s=100)
    t_line = ax.scatter([50], [50], c="orange", marker="x", s=100)
    
    # Set fixed axis limits
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("Agent Path")
    ax.legend()
    
    epoch = 0
    
    def update(frame):
        nonlocal epoch
        try:
            # Non-blocking check for new data
            if not dq.empty():
                app, (tx, ty) = dq.get_nowait()
                if app:
                    xs, ys = zip(*app)
                    line.set_data(xs, ys)
                    # t_line.set_offsets([[tx, ty]])
                    if len(xs) > 0:
                        start_point.set_offsets([[xs[0], ys[0]]])
                        end_point.set_offsets([[xs[-1], ys[-1]]])
                    
                    # Update title
                    ax.set_title(f"Agent Path - Epoch {epoch}")
                    epoch += 1
        except:
            pass  # No new data
        return line, start_point, end_point
    
    ani = animation.FuncAnimation(fig, update, interval=100, blit=True, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    # start_pos = (50, 50)
    # for i in range(10):
    #     x, y = rand_unit_cirlce_pos(start_pos, 20)
    #     ac_dist = math.hypot(x-50, y-50)
    #     print(f"({x:.2f}, {y:.2f}) -- Distance: {ac_dist}")
    run_train_and_view_procesess() 