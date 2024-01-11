---
title: "Aero Quanser Control"
excerpt: "Project about the implementation of different types of control algotithms (PID, LQR) in MATLAB and Simulink<br><br><img src='/images/aq_portfolioPhoto.png'>"
collection: portfolio
---

## 1. Introduction
This project, conducted during my time at *Politecnico di Milano*, delves into the identification and control of an *Aero Quanser*, a device created for educational purposes that facilitates the implementation of control algorithms in a real physical system.

In the first part of the project, the necessary mechanical properties of the system are estimated; after validation, these properties can be used to create a mathematical digital twin of the real system that can simulate the Aero Quanser.

Various control methods, such as Proportional Integral Derivative (PID), pole placement, and Linear Quadratic Regulator (LQR), are employed to govern the 1-DOF system under diverse scenarios. As the research progresses, similar methodologies are extended to a more complex 2-DOF system, aiming to enhance our understanding of multi-degree-of-freedom dynamics and to implement robust control strategies capable of mitigating external disturbances.

<div style="text-align:center">
  <img src="/images/aq_aeroQuanser.png" alt="alt text">
</div>

## 2. Model Identification

Initially, only pitch rotation freedom is allowed and the structural mechanical properties of the system are assessed: moment of inertia *J*, stiffness *k* and damping coefficient *r*. The mass, thrust displacement, and center of mass position are instead given.

<div style="text-align:center">
  <img src="/images/aq_parametersEstimation1dof.png" alt="alt text">
</div>
<br>

To evaluate the mechanical properties the response of the free system was measured to estimate its natural frequency.
This was done by activating only one motor, then when a stable position was reached the motor was stopped letting the structure freely oscillate.

<div  style="display: flex; justify-content: space-between;">
  <img src="/video/aq_freeOscillationGIF.gif" width="220" height="400" alt="alt text">
  <img src="/images/aq_freeOscillation.png" alt="alt text">
</div>
<br>

With the free response of the structure, the adimensional damping ratio was evaluated with the logarithmic decrement method, consequentially the natural frequency and damping ratio was calculated. For the stiffness of the system, only the gravitational contribution is considered since any internal springback or internal friction are negligible terms of a higher order that give a small contribution to the dynamic of the system.

Finally, the thrust force was linearized in function of the applied voltage, the relation between them was verified by measuring the thrust force. The comparison between the model and the real system is shown in the figure below.

<div style="text-align:center">
  <img src="/images/aq_modelComparison.png" alt="alt text">
</div>

## 3. Control of 1-DOF system

Various control methods are examined for a 1-DOF system under two scenarios: one following a stable step position as input and another allowing the system to track a sine wave. The Proportional Integral Derivative (PID) control involves tuning the  gains to achieve desired system response. The proportional term responds to the current error, the integral term addresses accumulated past errors, and the derivative term anticipates future errors, collectively providing stability and performance. The Simulink model of the PID control for a step reference input is shown below.

<div style="text-align:center">
  <img src="/images/aq_PIDmodel.png" alt="alt text">
</div>

With pole placement technique, instead, the goal is to determine the system's closed-loop poles by strategically placing them to achieve desired dynamic characteristics. This technique allows to shape the system's response by selecting appropriate pole locations in the complex plane, influencing the system's stability, speed of response, and damping ratio.
Two distinct types of system responses to a step input are presented: one characterized by a short rise time, while the other prioritizes the minimization of overshoot.

<div style="text-align:center">
    <img src="/images/aq_response1.png" alt="alt text">
    <img src="/images/aq_response2.png" alt="alt text">
</div>

Finally, an infinite time Linear Quadratic Regulator (LQR) is employed for optimal control by minimizing a quadratic cost function over an infinite time horizon. It uses state feedback and weighting matrices to balance the trade-off between control effort and system performance, resulting in an optimal control law that ensures stability and optimal performance of the system. 
The response to a sinusoidal reference with a LQR control is shown below.

<div style="display: flex; justify-content: space-between;">
    <img src="/video/aq_sineResponseGIF.gif" width="220" height="400" alt="alt text">
    <img src="/images/aq_sineTrajectory.png" alt="alt text">
</div>

## 4. Control of 2-DOFs system

The same tasks shown above were done also for the 2-DOFs system. To keep this presentation short, all the steps and results for the 2-DOF are not reported, but for further information feel free to contact me.

Only the response of the 2-DOFs system to an external macroscopic distrubance is shown below. In this case a LQR control algorithm was implemented to have a better response to the distrurbances and maintaing a stable position.

<div style="text-align:center">
    <img src="/video/aq_distrurbancesRejectionGIF.gif" width="600" height="340" alt="alt text">
</div>
