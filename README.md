# 🎯 Projectile Trajectory Predictor Under Wind Influence

**Author**: [Anurag Pathak](https://github.com/anuragpathak27)  
📧 Email: pathakanurag445@gmail.com

## 📌 Overview

This project simulates the trajectory of a projectile considering the influence of wind. It leverages fundamental physics equations to predict the projectile's path, accounting for parameters like initial velocity, launch angle, and wind speed. The simulation provides insights into how external factors, such as wind, affect projectile motion.

## 🚀 Features

- **Physics-Based Simulation**: Utilizes kinematic equations to model projectile motion under wind influence.
- **User Input**: Allows users to input initial conditions, including velocity, angle, and wind speed.
- **Data Logging**: Records simulation data into a CSV file (`projectile_dataset.csv`) for further analysis.
- **Modular Code Structure**: Organized codebase with clear separation of concerns, facilitating easy modifications and extensions.

## 🧠 Physics Principles

The simulation is grounded in classical mechanics, particularly the equations of motion for projectile trajectories. Key considerations include:

- **Horizontal Motion**:
  - Velocity: \( v_x = v_0 \cdot \cos(\theta) + v_{\text{wind}} \)
  - Displacement: \( x = v_x \cdot t \)

- **Vertical Motion**:
  - Velocity: \( v_y = v_0 \cdot \sin(\theta) - g \cdot t \)
  - Displacement: \( y = v_0 \cdot \sin(\theta) \cdot t - \frac{1}{2} g t^2 \)

Where:
- \( v_0 \) is the initial velocity,
- \( \theta \) is the launch angle,
- \( v_{\text{wind}} \) is the wind speed (positive for tailwind, negative for headwind),
- \( g \) is the acceleration due to gravity (9.81 m/s²),
- \( t \) is the time elapsed.


## 🛠️ Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/anuragpathak27/Projectile-Trajectory-Predictor-Under-Wind-Influence.git
   cd Projectile-Trajectory-Predictor-Under-Wind-Influence
    ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the simulation by executing the app.py script**:
   ```bash
   streamlit run app.py
   ```
   
Upon execution, the program will prompt you to input the following parameters:

- Initial Velocity (m/s): The speed at which the projectile is launched.
- Launch Angle (degrees): The angle above the horizontal at which the projectile is launched.
- Wind Speed (m/s): The speed of the wind affecting the projectile's horizontal motion.

After inputting the required parameters, the simulation will compute the projectile's trajectory and give the following parameters as output:

- Time (s): Time elapsed since launch.
- Horizontal Distance (m): Distance traveled along the horizontal axis.
- Vertical Height (m): Height above the launch point.
- Horizontal Velocity (m/s): Velocity component along the horizontal axis.
- Vertical Velocity (m/s): Velocity component along the vertical axis.
