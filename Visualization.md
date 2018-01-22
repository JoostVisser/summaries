# Visualization summary

*2IMV20*

[TOC]

## Lecture 1 - Organization

### Introduction

**Aim:** Provide both *theoretical* and *practical* knowledge.

**Materials:** Slides on Canvas + selected papers.

**Practical Work**

- **Assignment S1:** Volume visualization, visualizing a 3D dataset.
  - Written report.
- **Assignment S2:** Information visualization + interaction + data analysis
  - Written report + Screencast

Grading: $\frac{S1 + S2} 2$ if $S1 \geq 5$ and $S2 \geq 5$.

### Overview

What is visualization?

- Provide *visual representations* of datasets, designed to help *peIople* carry out *tasks* more efficiently.

In fact, we're using visualization to augment human capabilities.





## Lecture 2 

### Recap of last lecture

There are a couple of different volume rendering method:

- Slice-by-slice
  - Very easy to calculate but a bit unclear.
- Isosurfaces 
- Direct volume rendering
  - For each pixel we have a ray that passes through the object.
    - This ray R is perpendicular to the view plane.
    - This vector R can be calculated by getting the first and the last intersection with the bounding box. Then using $t \in [0, 1]$, we get samples of the object with some function $f(q_t)$ to get some scalar value (red dots).
  - Then we have some function $F$ on all the scalar values $s_t$, such as the maximum.
  - What if the pixel isn't there? Then we can apply **linear interpolation**.

Linear interpolation:

- Suppose we want a value $x$ on a line between $x_0$ and $x_1$. The distance between $x_0$ and $x$ is $\alpha$ and the distance between $x$ and $x_1$ is $1-\alpha$. We consider the values between $x_0$ and $x_1$ a linear function, then we can calculate value $v$ of $x$:

$$
\alpha = \frac{x-x_0}{x_1-x_0} \\
v=(1-\alpha)v_0 + \alpha v_1
$$

- Want this in more dimensions? Then we can apply it for each dimensino separately.

**Maximum intensity projection:** Essentially, we apply the function $I(p) = \max_t s_t$ as $F$ on the scalar values.

### Add colouring or shading

Iso-surface or Iso-contour: think of the pressure map.

- We draw some particular values, say $h=100, 104,108$, to get a sense of how the function looks like. (With some form of interpolation.)
  - Marching Squares Algorithm: perform a case distinction on how the contour passes the cell, by checking each of the corner vertices and see which value is higher or lower.
    - Results in $2^4=16$ cases. 
  - Then perform linear interpolation to calculate the exact intersection point (using linear algebra).
  - There are two cases where there is ambiguity, by extending or breaking. But if we assume that the scalar value are linearly interpolated in the grid cells, then we can calculate the value of the scalar, which follows a hyperbolic function. but then we only have to compare the isovalue against the value of the interpolant at intersection of asymptotes!
- What if we want this in 3D? We just perform a case distinction on 256 cases. This has a lot more ambiguous cases.

What does it look like? We try to select the value of the skin, then we can get an image of the foot.

We also need some shading to see the shape of the figure. We can use Phong shading model for this:
$$
I = \underbrace{I_ak_\texttt{ambient}}_{\texttt{Ambient light}} + \underbrace{I_lk_\texttt{diff}  (L\cdot N)}_{...} + \underbrace{I_l k_\texttt{spec}(V \cdot R)^\alpha}_{\texttt{Shininess}}
$$

- We need to know the normals (perpendicular side) of the surface for the shading. We do so by computing a normalized local gradient vector. We can use central differencing instead for forward differencing for better average results.

Downsides of Isosurfacing:

- Provides only approximation of a surface.
  - Only works if the object-to-visualize has a similar function value.
- Amorphous phenomena have no surfaces. 

### Direct volume rendering

There is a pretty difficult volume rendering equation, but a linear implementation of a simplified and discretized version of the formula is:

$C_i = \tau_i c_i + (1-\tau_i) C_{i-1}$ 

- $\tau_i$ is the opacity. 
- $c_i$ is the colour.

First tactic:

- Trial-and-error colouring using values with the histogram function.
- Is there another solution? Well, we can use data-driven transer function design. 
  - A boundary exists where the maximum difference happens in terms of value. So we want to find the maximum in the first-order derivative.