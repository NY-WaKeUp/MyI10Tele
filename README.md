# MyI10Tele

This project is designed to collect demonstration data for the Aubo i10 robotic arm, supporting remote operation and recording of the process in both video and robot state formats.  
Main features include:

- Controlling the Aubo i10 robotic arm via a MyEnv-based environment for demonstration data collection
- Capturing data such as RGB observation images, wrist camera images, 7-DoF robot states, and operator actions
- Continuous encoded video storage with h264 support; automatic switching to VideoToolbox acceleration on macOS
- Ability to create or reuse data directories as needed, with options to overwrite existing ones
- Uses the lerobot dataset structure for seamless integration with LeRobotDataset

**Quick Start**
1. Set the `XML_PATH` and `ROOT` variables to point to your local resource paths
2. Run:  
   ```
   python src/core/0.tele.py
   ```
3. Follow the prompts to operate the robot arm and collect demonstration data

Default task: “Put cube on the black platform”

This tool is intended for data collection in imitation learning, robotic reinforcement learning, and similar algorithm research.  
To customize the collected elements or tasks, simply modify the relevant parameters or environment initialization in `src/core/0.tele.py`.
