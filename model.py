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


def rand_unit_cirlce_pos(coords, distance):
    x, y = coords
    theta = random()*2*math.pi
    sx = distance*math.cos(theta) + x
    sy = distance*math.sin(theta) + y
    return (sx, sy)


def train_loop(data_queue: Queue):
    agent = RLPolicy()
    critic = RLValue()
    policy_opt = torch.optim.Adam(agent.parameters(), lr=1e-4)
    value_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
    total_steps = 0
    epoch = 0    
    last_n = np.zeros((100,)).tolist() 
    current_start_dist = 3
    try:
        checkpoint = torch.load("checkpoint.pt")
        agent.load_state_dict(checkpoint["policy_state_dict"])
        critic.load_state_dict(checkpoint["value_state_dict"])
        policy_opt.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        value_opt.load_state_dict(checkpoint["value_optimizer_state_dict"])
        total_steps = checkpoint["global_step"]
        epoch = checkpoint["epoch"]
    except Exception as e:
        print(f"Failed to load; restarting training {e}")
    # epoc init
    while(True):
        total_reward = 0
        start = perf_counter()
        tx, ty = (50, 50) # (randint(10, 90), randint(10, 90)) #target position
        cx, cy = rand_unit_cirlce_pos((tx, ty), current_start_dist) # current position
        rf = env_sim("../dia-homeworks/project/models/data_uniform_random", (tx, ty), (cx, cy))
        state_vec = rf.get_env() 
        succeed = False
        sigma = 0.05
        gamma = 0.99
        entropy_coef = 0.01
        max_grad_norm = 0.5
        # buffers
        states, actions, rewards, log_probs, values, entropies = [], [], [], [], [], []

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
            # get data for this step
            states.append(state.squeeze(0))
            actions.append(action.squeeze(0))
            rewards.append(reward)
            log_probs.append(dist.log_prob(action).sum())
            values.append(critic(state))
            entropies.append(dist.entropy().sum())

            state_vec = next_state_vec
            
            if complete:
                break

        # do learning for this epoch
        # data fixing
        states = torch.stack(states)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        log_probs = torch.stack(log_probs)
        values = torch.stack(values).squeeze()
        entropies = torch.stack(entropies)

        # if we took too long
        if complete:
            # If the episode ended, the final return is just the final reward/terminal state value
            # The bootstrap term is 0 if it was a final success/failure.
            # We will use the last state in the buffer to determine the value of the next state
            # but this calculation needs to be more robust for full episodes.
            R = 0.0
        else:
            # If max_steps reached, bootstrap the value
            final_state = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
            R = critic(final_state).detach().squeeze().item()

        returns = []
        # Calculate discounted returns (R_t = r_t + gamma * R_{t+1})
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # Generalized Advantage Estimation (GAE) or simple TD-Advantage
        advantage = returns - values.detach() # .detach() here is important

        # --- RE-ENABLE Advantage Normalization Safely ---
        # Now that advantage is a batch of N values, this works perfectly!
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

        rt = perf_counter() - start
        total_steps += i
        cx, cy = rf.cpos
        if succeed:
            last_n[epoch%100] = 1
        else:
            last_n[epoch%100] = 0
        print(f"TS: {total_steps} -- Epoch: {epoch} -- Average Reward: {total_reward/(i+1e-8):.4f} -- Runtime: {rt:.2f}s Win Rate {sum(last_n)}% Spawn Distance {current_start_dist}")
        data_queue.put((rf.all_previous_positions, (tx, ty)))
        win_rate = sum(last_n)/100
        if win_rate*100 > 90.0:
            current_start_dist += 1 
            last_n = np.zeros((100,)).tolist()
            torch.save({
                "policy_state_dict": agent.state_dict(),
                "value_state_dict": critic.state_dict(),
                "policy_optimizer_state_dict": policy_opt.state_dict(),
                "value_optimizer_state_dict": value_opt.state_dict(),
                "global_step": total_steps,
                "epoch": epoch
            }, "checkpoint.pt")
        epoch += 1        


def run_train_and_view_procesess():
    dq = Queue()
    tp = Process(target=train_loop, args=(dq,), daemon=True)
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