---
title: "Noise and Vibration Engineering Projects"
excerpt: "Three small projects about the interaction of noise and vibrating structures<br/><img src='/images/500x300.png'>"
collection: portfolio
---

### Introduction

### 1. Bending waves in thin rectangular plates

In this project, a theoretical plate's vibration behavior is analyzed analytically and with a Finite Element Analysis made with a MATLAB script. Two different constraint conditions are considered, in the first part the plate is constrained on all four edges and later only on two opposite ones.

## Rectangular plate supported on the four edges

The mode shapes and natural frequencies of the plate are evaluated analytically and the first 6 mode shapes are shown in the figure below
![alt text](/images/nv_firstModeShapes.png)

The comparison of the analytical solution for the natural frequencies with the FEM solutions is shown in the following scatter plot.
Notably, the solutions are nearly identical up to a frequency of about 350 Hz, above that the differences in the results increase rapidly. As a reason for that observation can be stated, that in the mentioned frequency region the resolution of the FEM model is not any more accurate enough to properly describe the physical situation.
For the analytical model, on the other hand, the assumption to neglect the shear contribution becomes an issue for high frequencies, since the wavelength reduces to be comparable to the plate thickness.
![alt text](/images/nv_scatterPlot1.png)
![alt text](/images/nv_MAC1.png)