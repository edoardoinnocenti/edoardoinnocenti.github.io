---
title: "Money Drone Game"
excerpt: "Python-based game where you can play against a PID and a Reinforcement Learning AI  <br><br><img src='/images/md_soloPlayer.png'>"
collection: portfolio
---

<br>
<p><a href="https://github.com/edoardoinnocenti/DroneGame" target="_blank">For the Full Code Visit the DroneGame GitHub Repository</a></p>

## 1. Introduction
This project involves developing a Python-based game, through the pygame library, that centers around piloting a simulated quadcopter drone. The primary objective of the game is for players to gather as much money as possible, which serves as their score, within a predetermined time limit. Players have the option to either play solo or compete against an AI-controlled drone.

This AI opponent can be directed through either a Proportional Integral Controller (PID) or a more advanced drone simulated using a Reinforcement Learning (RL) algorithm, specifically a Soft Actor-Critic (SAC) agent. The game offers an immersive experience that not only challenges players' strategic and navigational skills but also provides an opportunity to understand the intricacies of different drone control systems.

<div style="text-align:center">
  <img src="/images/md_soloPlayer.png" alt="alt text">
</div>
<br>

## 2. Physics Simulation and Replication in Pygame Environment

In the simulation, the quadcopter's dynamics are modeled to reflect real-world physics. This includes factors such as lift generated by the rotors and gravitational forces. The model ensures that the quadcopter's behavior in the simulation closely mirrors its behavior in a real environment. The 2D aspect of the simulation allows to simplify the complex dynamics of a quadcopter, the equation of motions are reported below:

<div style="text-align: center;">
  <p>\( m\dot{x} = -(T_l + T_r) \sin(\theta) \)</p>
  <p>\( m\dot{y} = -(T_l + T_r) \cos(\theta) + mg \)</p>
  <p>\( m\ddot{\theta} = (T_r - T_l) \cdot l \)</p>
</div>

The human participants can control the quadcopter within the 2D simulation environment. The interaction is likely structured to allow players to use input devices like a joystick or keyboard to maneuver the quadcopter, aiming to navigate and collect the money.

## 3. PID Control

A PID (Proportional-Integral-Derivative) controller is a control loop mechanism widely used in industrial control systems. It calculates an error value as the difference between a desired setpoint and a measured process variable. The controller attempts to minimize this error by adjusting the process control inputs.

- Proportional (P): This part of the controller reacts proportionally to the current error. The larger the error, the greater the proportional response.
- Integral (I): This component sums up past errors, addressing accumulated offset that the proportional part cannot eliminate.
- Derivative (D): This part predicts future errors based on the rate of change, introducing a damping effect to minimize overshooting.

In implementing a PID controller for quadcopter control, these components work together to adjust the rotor speeds, ensuring stable flight and precise navigation. The PID parameters (P, I, D) are tuned to match the quadcopter's dynamics for optimal performance.

<div style="text-align:center">
  <img src="/images/md_PID.png" alt="alt text">
</div>

## 4. Reinforcement Learning Control

The Soft Actor-Critic (SAC) agent is an advanced reinforcement learning algorithm used for controlling the quadcopter in the simulation. This algorithm is part of a family of actor-critic methods that learn both a policy (actor) and a value function (critic).

The SAC algorithm is particularly known for its stability and efficiency in learning. It incorporates three key components:

- Off-Policy Learning: This allows the SAC agent to learn from past experiences, improving efficiency and stability.
- Entropy Maximization: SAC seeks to maximize both the expected return and the entropy of the policy. This entropy term encourages the agent to explore more diverse strategies, leading to more robust learning.
- Actor-Critic Architecture: The actor updates the policy to maximize the expected return, while the critic estimates the value of state-action pairs, guiding the actor's updates.

In implementing the SAC algorithm for the quadcopter, these features enable the agent to effectively learn complex maneuvers and adapt to the dynamics of the quadcopter, outperforming traditional control methods in many aspects. The SAC agent's learning process involves interacting with the environment, receiving feedback, and iteratively improving its control policy.

<div style="text-align:center">
  <img src="/images/md_SAC.png" alt="alt text">
</div>

## 5. Results

A comparative gameplay session is reported below, where I, alongside a PID-controlled and a SAC-controlled agent, simultaneously pilot the three different drones in the simulation. Please forgive me for my bad performance &#128517;

<div style="text-align:center">
    <img src="/video/md_game.gif" alt="alt text">
</div>
<br>

*Disclaimer: this game is inspired by the Quadcopter-AI project by AlexandreSajus*