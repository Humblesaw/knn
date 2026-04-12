# Waypoint Mission Curriculum Learning Guide

## Overview
This curriculum teaches aircraft to fly autonomous waypoint missions using reinforcement learning with JSBSim flight simulator. The approach uses progressive task complexity aligned with realistic flight capabilities.

## Curriculum Stages

### Stage 1: Level Flight + Heading Hold (Foundation)
**Goal**: Maintain level attitude while holding a constant heading  
**Difficulty**: Easy  
**Duration**: ~800K timesteps (PPO)  
**Learning Rate**: 3e-4  
**Key Skills**:
- Roll/pitch stability
- Heading tracking (single target)
- Basic attitude control

**Reward Components**:
- Roll alignment: 0.5 (exponential decay)
- Pitch alignment: 0.3
- Heading error: 0.2 (Gaussian penalty)

---

### Stage 2: Level Flight + Random Heading Navigation
**Goal**: Maintain level flight while navigating to randomly selected headings  
**Difficulty**: Medium  
**Duration**: ~1M timesteps (PPO)  
**Learning Rate**: 2.5e-4  
**Key Skills**:
- Heading tracking (variable targets)
- Bank angle management for turns
- Smooth attitude transitions

**Reward Components**:
- Heading navigation: 0.6
- Roll stability: 0.2
- Pitch stability: 0.2

---

### Stage 3: 2D Navigation (Altitude + Heading)
**Goal**: Maintain random altitude while navigating to random headings  
**Difficulty**: Hard  
**Duration**: ~1.5M timesteps (PPO)  
**Learning Rate**: 2e-4  
**Key Skills**:
- Altitude control (vertical navigation)
- Coordinated heading changes at altitude
- Multi-objective task management

**Reward Components**:
- Heading navigation: 0.4
- Altitude hold: 0.4
- Stability: 0.2

**New Physics**:
- Altitude targets: 500-3000 ft AGL
- Altitude tolerance: ±50 ft for rewards

---

### Stage 4: Waypoint Missions (Full Task)
**Goal**: Follow sequential waypoints (2-3 per mission) with altitude and heading constraints  
**Difficulty**: Very Hard  
**Duration**: ~2M timesteps (PPO or **better: SAC**)  
**Learning Rate**: 1.5e-4  
**Key Skills**:
- Sequential mission planning
- Waypoint acquisition
- Long-horizon task management
- Multi-step navigation

**Reward Components**:
- Heading navigation: 0.35
- Altitude hold: 0.35
- Stability: 0.15
- Waypoint bonus: 0.5 (when on waypoint) / 5.0 (mission complete)

**Mechanics**:
- Waypoints are (heading, altitude) tuples
- Episode generates 3 random waypoints
- Agent must reach each waypoint in sequence
- Waypoint completion requires 100 steps within 5° heading and 50 ft altitude

---

## Algorithm Recommendations

### PPO (Current - Good for Stages 1-3)
✅ **Strengths**:
- Very stable convergence
- Sample efficient for simple tasks
- Easy to tune hyperparameters
- On-policy: good for curriculum progression (clear task progress)

❌ **Limitations**:
- Requires many samples for complex exploration
- Less effective for sparse/delayed rewards
- May plateau on Stage 4

**Use**: Stages 1-3

**Hyperparameters** (current):
```python
batch_size=256
n_steps=2048
clip_range=0.2
gamma=0.99
```

---

### SAC (Recommended for Stage 4 - Better Exploration)
✅ **Strengths**:
- **Off-policy learning**: 2-3x more sample efficient than PPO
- Automatic entropy regularization: natural exploration control
- Better handling of multi-modal reward landscapes
- Superior for long-horizon planning (waypoint sequences)
- More sample efficient = faster training

❌ **Limitations**:
- Requires careful hyperparameter tuning
- More memory overhead (experience replay buffer)
- Slower wall-clock time per step

**Why SAC for Stage 4**:
1. **Sample efficiency**: Waypoint missions are expensive to simulate
2. **Exploration**: SAC's entropy bonus helps explore different waypoint approaches
3. **Multi-objective**: Better at balancing heading + altitude + progression
4. **Long-horizon**: Off-policy learning helps with sparse waypoint rewards

**Recommended Hyperparameters for Stage 4 SAC**:
```python
learning_rate=1.5e-4
gamma=0.99
tau=0.005  # soft update coefficient
ent_coef='auto'  # let SAC auto-tune entropy
batch_size=256
buffer_size=100_000
```

---

### Hybrid Strategy (BEST)
1. **Train Stages 1-3 with PPO** (proven stable)
2. **Fine-tune Stage 4 with SAC** (better exploration + sample efficiency)
3. Optionally transfer Stage 3 policy to Stage 4 as warmstart

**Benefits**:
- Stability of PPO for curriculum foundation
- Efficiency of SAC for complex task
- Faster overall training

---

## Training Commands

### Train Stage 1 (Level Heading Hold)
```bash
# In curriculum.py, set:
MODE = "train"
STAGE = 1
ALGORITHM = "PPO"

python curriculum.py
# Monitor: tensorboard --logdir logs/curriculum
```

### Train Stage 2 (Random Heading)
```bash
# In curriculum.py, set:
MODE = "train"
STAGE = 2
ALGORITHM = "PPO"

python curriculum.py
```

### Train Stage 3 (Altitude + Heading)
```bash
# In curriculum.py, set:
MODE = "train"
STAGE = 3
ALGORITHM = "PPO"

python curriculum.py
```

### Train Stage 4 (Waypoints with SAC - RECOMMENDED)
Option A: Continue with PPO
```bash
# In curriculum.py, set:
MODE = "train"
STAGE = 4
ALGORITHM = "PPO"

python curriculum.py
```

Option B: Use SAC (better for long-horizon)
```bash
# Modify curriculum.py: change train() function to use SAC instead
# Or create a separate curriculum_sac.py with SAC variant
MODE = "train"
STAGE = 4
ALGORITHM = "SAC"

# Then update make_env() and train() to create/use SAC model
```

---

## Visualization

```bash
# In curriculum.py, set:
MODE = "vis"
STAGE = 4

python curriculum.py
# Shows real-time roll/pitch/heading/altitude plots
```

---

## Key Hyperparameter Tuning Tips

### If training is unstable:
- Reduce learning rate by 0.5x
- Increase entropy coefficient (exploration)
- Reduce clip_range to 0.1 (PPO)

### If training is slow:
- Increase batch_size (if GPU memory allows)
- Increase learning rate by 1.5x
- Reduce n_steps (PPO) for more frequent updates

### For Stage 4 (Waypoint):
- Consider longer n_steps (4096) for better policy stability
- Increase entropy for better exploration of waypoint approaches
- Use SAC if PPO plateaus

---

## Expected Performance Progression

| Stage | Task | PPO Performance | Time (hours) |
|-------|------|-----------------|--------------|
| 1 | Level + Heading | >0.80 reward | 2-3 |
| 2 | Random Heading | >0.70 reward | 3-4 |
| 3 | 2D Navigation | >0.60 reward | 4-6 |
| 4a | Waypoints (PPO) | >0.50 reward | 6-10 |
| 4b | Waypoints (SAC) | >0.60 reward | 4-8 |

---

## Next Steps

1. **Start with Stage 1** using PPO (easiest to debug)
2. **Monitor TensorBoard** reward curves for convergence
3. **Validate each stage** with visualization before progressing
4. **For Stage 4**: Try PPO first, then experiment with SAC if needed
5. **Transfer learning**: Use Stage 3 weights to warmstart Stage 4

---

## File Structure
```
curriculum.py              # Main training entrypoint
curriculum_tasks.py        # Task wrapper definitions
models/                    # Saved models (stage_X_ppo.zip, etc)
logs/curriculum/           # TensorBoard logs
CURRICULUM_GUIDE.md        # This file
```
