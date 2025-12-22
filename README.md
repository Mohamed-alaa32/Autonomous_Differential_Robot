# Autonomous Differential Drive Robot

This repository contains the simulation, control, and description
packages for a custom autonomous differential drive robot for Autonomous Systems Course at Ain Shams University. It is
explicitly designed for **ROS 2 Jazzy** and utilizes **Gazebo
Harmonic** for high-fidelity physics simulation.


## 🛠 Prerequisites

Ensure you have the following installed:

-   **OS:** Ubuntu 24.04 (Noble Numbat)
-   **ROS Distro:** ROS 2 Jazzy
-   **Simulator:** Gazebo Harmonic

## 🚀 Installation

### 1. Setup Workspace

``` bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
```

### 2. Clone Repository

``` bash
git clone https://github.com/Mohamed-alaa32/Autonomous_Differential_Robot.git

```

### 3. Install Dependencies

``` bash
cd ~/ros2_ws
sudo apt update
rosdep install --from-paths src --ignore-src -r -y
```

Key dependencies: `ros-jazzy-ros-gz`, `ros-jazzy-xacro`,
`ros-jazzy-robot-state-publisher`.

### 4. Build

``` bash
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

## 🎮 Usage

### 1. Launch Simulation

``` bash
ros2 launch autonomous_differential_rob sim.launch.py
```

Arguments: - `use_rviz` (default: true) - `world` (default:
world.sdf)

### 2. Teleoperation

``` bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

### 3. View TF Frames

``` bash
ros2 run tf2_tools view_frames
```

## 📂 Directory Structure

    config/
      └── diff_drive.yaml
    launch/
      └── sim.launch.py
    scripts/
      └── control.py
      └── lane_keeping.py
    urdf/
      └── robot.xacro
    worlds/
      └── world.sdf
