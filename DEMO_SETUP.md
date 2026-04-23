# UGV Ball Push Demo — Classmate Setup

## What You Need
- Python 3.8 or newer
- A laptop with WiFi (Windows, Mac, or Linux all work)
- Git installed

---

## Step 1 — Clone the Repo

```bash
git clone -b pure-rl-fov https://github.com/lordoofring/rl_ugv_jetson.git
cd rl_ugv_jetson
```

## Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

> If you get a conflict with opencv, run this instead:
> ```bash
> pip install stable-baselines3 opencv-python pyyaml gymnasium shimmy numpy
> ```

---

## Step 3 — Connect to the Robot's WiFi

- Network name: **AccessPopup**
- Password: **1234567890**

> Make sure your laptop is connected to **AccessPopup** before continuing.  
> You will lose internet access while connected — that's normal.

---

## Step 4 — Start the Server on the Robot

Open a terminal and SSH into the Jetson:

```bash
ssh jetson@192.168.50.5
```

Password: **jetson**

Once logged in, start the server:

```bash
python3 ~/ugv_rl/run_server.py
```

Leave this terminal open. You should see `Server listening...` or similar.

---

## Step 5 — Run the Policy on Your Laptop

Open a **second terminal** on your laptop, navigate to the repo, and run:

```bash
cd rl_ugv_jetson
python run_ball_push.py --model ball_push_ppo_final --ip 192.168.50.5
```

A camera window will open showing what the robot sees. The robot will start moving automatically.

---

## Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `SPACE` | Pause / Resume |
| `R` | Reset step counter |

---

## Troubleshooting

**Can't connect to AccessPopup?**  
The robot may still be booting. Wait 30 seconds and try again.

**SSH connection refused?**  
Make sure the server isn't already running. Check with:
```bash
pgrep -a python3
```
If it's running, kill it with `kill <PID>` and restart.

**Camera window doesn't open?**  
Make sure you're running the command in a regular terminal (not over SSH).

**Robot not moving?**  
Check the first terminal — the server may have crashed. Press Ctrl+C and rerun `python3 ~/ugv_rl/run_server.py`.
