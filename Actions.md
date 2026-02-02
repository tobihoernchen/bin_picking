## Actions and Training Procedures

### Pick from Box

- Programmierung: "Pick from Box at (x, y, z, a, b, c) | (w, l, h)"
- Components: 
    - Gripping Pose Estimation: <br/>
    Input: Dual Camera Image <br/> 
    Output: 6D-Gripping Pose 
    - Trajectory Net: <br/>
    Input: Dual Camera Image + Relevant Points + Axis Angles <br/> 
    Output: Axis Angles (within max. distance to prev.)

- Training
    - Gripping Pose Estimation: <br/>
    Dataset with STL Scene + High Res SDF within Box, Mujoco xml, Mujoco State, Mujoco Dual Camera Render, Metadata (Box Position, Camera Position etc.) <br/>
    Energy-Based (no target variable). Use Gripper and Mesh Data + SDF to compute grip optimality. For SDF: use torch.nn.functional.grid_sample <br/>



### Position Picked Part
- Programmierung: "Position like in this Picture"

