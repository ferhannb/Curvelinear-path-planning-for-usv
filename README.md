# RotaOptimalds

A clothoid-based multi-waypoint MPC route optimization project.

This project generates trajectories between waypoints using receding-horizon MPC, jointly optimizing curvature state (`K`) and curvature command (`Kcmd`).

## Features

- Clothoid increment-based kinematic propagation
- Ramp-limited curvature transition via `K_next_fixed_ramp`
- Multi-waypoint receding-horizon flow
- Terminal / hit / tube cost structure
- Blocked controls support (`Kcmd`, `ds`)
- Warm start + solution shifting
- 2x2 analysis plots:
  - XY trajectory
  - Curvature profile (`K`, `Kcmd`)
  - Heading vs `s`
  - Step length (`ds`) profile
- Waypoint labeling:
  - Segment labels `WPi n=<point_count>`
  - Start-index labels `WPi`

## Project Structure

- `Fresnel/RotaOptimalds.py`
  - Legacy entry file (wrapper)
- `Fresnel/rota_optimalds/helpers.py`
  - Math and dynamics helper functions
- `Fresnel/rota_optimalds/mpc.py`
  - `MPCNumericClothoidCost` class
- `Fresnel/rota_optimalds/scenario.py`
  - Default parameters and scenario
- `Fresnel/rota_optimalds/main.py`
  - Modular entry point

## Requirements

- Python 3.10+
- `numpy`
- `matplotlib`
- `casadi` (with IPOPT recommended)

Suggested setup (venv):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib casadi
