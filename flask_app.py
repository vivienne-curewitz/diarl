from flask import Flask, request, jsonify
# from flask_cors import CORS
import json
from read_csv import env_sim
from model import load_model
import torch
import torch.nn.functional as F

app = Flask(__name__)

rf = env_sim("", (0,0), (0,0)) # args don't matter
agent = load_model()
sigma = 0.05

@app.route("/sync")
def sync():
    print("hit")
    return "ok"

@app.route('/update', methods=['POST'])
def update_location():
    data = request.data
    # print(data)
    sd = data.decode("utf-8")
    gx, gy, cx, cy, lt, pl = tuple(sd.split("\t"))
    print(f"Goal: {gx}, {gy} -- Current: {cx}, {cy}")
    
    # return jsonify(response)
    tdx, tdy = next_location((float(gx), float(gy)), (float(cx), float(cy)), pl)
    # ret = f"{float(cx)+1},{float(cy)+1}"
    ret = f"{tdx},{tdy}"
    return ret

def next_location(tpos, cpos, points):
    state_vec = rf.get_env_one_shot(tpos, cpos, points)
    print(f"Len state vec {len(state_vec)}")
    state = torch.tensor(state_vec, dtype=torch.float32)
    mu = agent(state)
    dist = torch.distributions.Normal(mu, sigma)
    action = dist.sample()
    action = F.normalize(action, dim=-1)
    dx, dy = action.detach().numpy()
    cx, cy = cpos
    tdx = cx + dx
    tdy = cy + dy
    return (tdx, tdy)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7654, debug=True)