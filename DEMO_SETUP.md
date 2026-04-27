# UGV Ball Push UT EXPLORER day instructions

## What You Need
- Python 3.8 or newer
- A laptop with WiFi (Windows, Mac, or Linux all work)
- Git installed

---
*(If you have done these steps already, skip to step 3)*

## Step 1 — Clone the Repo

```bash
git clone -b pure-rl-fov https://github.com/lordoofring/rl_ugv_jetson.git
cd rl_ugv_jetson
```

## Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

> If you get a conflict with opencv, uninstall the existing version first:
> ```bash
> pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
> pip install -r requirements.txt
> ```

---
**note: run `git pull` to ensure your local repo is synchronized with the latest changes**

## Step 3 — Connect to the Robot's WiFi

- Network name: **AccessPopup**
- Password: **1234567890**

> Make sure your laptop is connected to **AccessPopup** before continuing.  
> You will lose internet access while connected — that's normal.

---

## Step 4 — Open Two Terminals

You need **two terminals open at the same time** for the rest of the steps.

- **Terminal 1** → talks to the robot (SSH)
- **Terminal 2** → runs the policy on your laptop

Open both now before continuing.

---

## Step 5 — Terminal 1: Start the Server on the Robot

In **Terminal 1**, SSH into the Jetson:

```bash
ssh jetson@192.168.50.5
```

Password: **jetson**

Once logged in, start the server:

```bash
python3 ~/CS2320/rl_ugv_jetson/run_server.py
```

You should see `Server listening...` or similar. **Leave Terminal 1 open and do not close it.**

---

## Step 6 — Terminal 2: Run the Policy on Your Laptop

In **Terminal 2** on your laptop (not SSH — your own machine), navigate to the repo and run:

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
Check Terminal 1 — the server may have crashed. Press Ctrl+C and rerun `python3 ~/ugv_rl/run_server.py`.
