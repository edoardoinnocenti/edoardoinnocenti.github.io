---
title: "Design of a Corking Machine"
excerpt: "Report of the design procedure of a corking machine using MATLAB and Adams View<br><br><img src='/images/cm_portfolioPhoto.png'>"
collection: portfolio
---

## 1. Introduction
This project report aims to review the design process of a corking machine, used to insert a cork in bottles of wine.
In the first part, the motion law for the plunger was chosen to follow a cycloidal profile, chosen for its smooth and efficient motion. Then the design process involves two phases: initial lengths definition of the machine's elements using MATLAB and subsequent kinematic and dynamic analysis in Adams View.

<div style="text-align:center">
  <img src="/images/cm_corkingMachine.png" alt="alt text">
</div>

## 2. Design and Motion Law definition

Starting from the design and productivity requirements, the proposed system operates by means of three four-bar mechanisms, all driven by the same cam. The plunger has to perform a total stroke of *8mm* to press the cap, assuring a force of *500N*. Moreover, the constraint on productivity requires the cam to perform three revolutions per second.

<div style="text-align:center">
  <img src="/images/cm_requirements.png" alt="alt text">
</div>

The motion law of the plunger is composed by four different phases. In particular, the effective motion occurs in 120° of rotation of the cam (45° forward and 75° backward), while the working phase (plunger in position near to the external dead point) has a duration of 80°. In the remaining 160°, the plunger remains in the retracted position (internal dead point), to allow the insertion of a new cap. After analyzing the different possible profiles for the motion law, a cycloidal one is chosen. This motion law is favored for its ability to provide smooth, precise, and efficient motion, reducing mechanical stress and wear.

<div style="text-align:center">
  <img src="/images/cm_motionLawInitial.png" alt="alt text">
</div>
<div style="text-align:center">
  <img src="/images/cm_motionLaw.png" alt="alt text">
</div>

## 3. Design of the mechanism

In order to synthesize the mechanism, the direct method has been employed. Defining the abscissa ε, this method requires the nullification of the structural error in a finite number of points, i.e. null e(ε) = f(ε)−g(ε) at each precision point. Furthermore, for each simulation, the motion of the mechanism between those points has been checked.
The design process has been divided in two main phases. The first aimed at addressing the kinematic of the mechanism through the use of MATLAB. At the end of this phase, an initial length of the links will be obtained. Starting from the results obtained from the previous step, during the second phase the kinematic and the dynamic behaviours of the mechanism have been studied employing Adams View.

<div style="text-align:center">
  <img src="/images/cm_scheme.png" alt="alt text">
</div>
<br>

# 3.1 Phase One: initial lengths definition

Focusing on the two equal slider-crank mechanisms OAB and O’A’B’, two variables were introduced to define the precision points: the characteristic ratio γ and the angle θ, defined as the angle among the link AB and the horizontal axis passing through O.

The goal of the first simulation was to evaluate the minimum length able to satisfy the imposed constraints. As a first step, the motion of the plunger was imposed and the motion of point A can consequentially be evaluated in order to use it as input for the internal slider crank design. The final result of the inverse kynematic analysis are reported below.

<div style="text-align:center">
  <img src="/images/cm_parameters.png" alt="alt text">
</div>
<div  style="display: flex; justify-content: space-between;">
  <img src="/images/cm_velocity.png" alt="alt text">
  <img src="/images/cm_acceleration.png" alt="alt text">
</div>
<br>

# 3.2 Phase Two: ADAMS simulations

As previously stated, once the geometric values of the mechanism had been defined, other quantities can be studied using ADAMS View, i.e. the dynamic behaviour of the mechanism. In order to properly account for the inertial forces, also the material had to be selected. Hence, all the components have been selected of steel, while the plunger has been chosen of stainless steel because of the contact with food and beverage products. After the links had been properly modelled on ADAMS View, as shown below.

<div style="text-align:center">
  <img src="/images/cm_adamsModel.png" alt="alt text">
</div>
<br>

Once all whole mechanism had been defined on ADAMS View, a kinematic simulation was done imposing the motion at the plunger. The results can be compared to the ones obtained with MATLAB: as expected, there is a perfect a matching among the shapes of the curves obtained with the two programs.

Then, a further step in the design process had been done: the cam design. For this project, a cam and roller mechanism has been chosen, thanks to its ability to reduce the friction and since allows not to use lubricants. In particular, this can be obtained by imposing a curve-to-curve contact between the roller and the cam profile.

<div  style="display: flex; justify-content: space-between;">
  <img src="/images/cm_camProfile.png" alt="alt text">
  <img src="/images/cm_camNoUndercut.png" alt="alt text">
</div>
<br>

Once the cam had been defined, the contact constraint between the roller and the cam was introduced. Moreover, a spring was employed to maintain the contact among these two elements.

Thus, a dynamic simulation was performed and, as can be clearly seen, the motion of the plunger is not coincident with the desired one. This is due to the fact that the spring introduces lots of oscillations that are not sufficiently damped.

<div style="text-align:center">
    <img src="/video/cm_dynamicsFirstIterstion.gif" width="600" height="340" alt="alt text">
</div>
<br>

After several simulations, aimed at properly tuning the stiffness of the spring, the best result in terms of following the imposed motion was found. However, this happens at the expense of the contact force, which resulted in being too high and would have led to a contact pressure impossible to be withstood by most commercial steels.

A possible solution was the one related to a change in the geometric parameters of the mechanism, with the goal of decreasing the maximum displacement of the follower. Indeed, this choice allows to increase the stiffness of the spring without increasing too much the elastic restoring force, one of the main parameters affecting the contact pressure.

Clearly, once the geometric characteristics had been modified, the whole kinematic of the mechanism changes, leading to a change in the required cam. Doing the very same procedure followed previously, the new cam has been obtained and reported below.

<div  style="display: flex; justify-content: space-between;">
  <img src="/images/cm_camProfileFinal.png" alt="alt text">
  <img src="/images/cm_camNoUndercutFinal.png" alt="alt text">
</div>
<br>

At this point, several dynamic simulations were done, were the only difference was the value of the spring stiffness. Moreover, the *500N* force required to force the cork in the bottle was introduced. Finally, the effect of the friction between the cam and the roller had been investigated. The final dynamic simulation is reported below.

<div style="text-align:center">
    <img src="/video/cm_dynamicsFinalIterstion.gif" width="600" height="340" alt="alt text">
</div>
<br>

## 4. 3D CAD Design

<div style="text-align:center">
  <img src="/images/cm_cadModel.png" alt="alt text">
</div>