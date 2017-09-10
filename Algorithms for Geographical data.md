# Algorithms for Geo data

[TOC]

*Course code:* 2IMG15

$$
\texttt{LaTeX commands}
\newcommand{\O}{\mathcal O}
$$

## Lecture 1

### Geographic data

Development of techniques and tools that helps process geographic datas:

- Can be abstract polygons
- Can be a planar subdivision
- Can be points on a map [earthquake epicentre map]

Grading scheme, 50/50:

- 4 homework assignments $\leftarrow$ Both individual and group
- One big assignment $\leftarrow$ You can choose this.

**Academic Dishonesty:** When stealing other people's stuff, try to change it so that they don't recognise it.

#### Scales

Various scales that we can use:

- **Nominal** scale: No kind of order.
- **Ordinal** scale: There is some kind of order; questionnaire-answers or school-type
- **Interval** scale: degrees celsius or fahrenheit. Ratios make no sense, but differences do.
- **Ratio** scale: Equivalence, order, difference and ratio all make sense. (*Natural zero*)

There are other scales, however:

- **Angle**
- **Vector**
- **Categorical** scales with partial membership: fuzzy lines between categories.

#### Classification schemes

How to aggregate interval and scales? Such as 4, 5, 5, 8, 12, 14, 17, 23, 27.

- Fixed intervals: [1-10], [11-20], [21-30]
- Fixed intervals, *based on spread*.
- Quantiles, *equal representatives*.
- "Natural boundaries"
- ...

Choice of classification is important, as it influences *interpretation*.

**Object view:** Discrete objects in the real world.
*Example:* roads or lakes

**Field view:** Geographic variable has a value at every lcoation in the real word.
*Example:* Elevation, temperature.

Dependency of dimension: it can depend on how we represent objects depending on e.g. the scale.
Also the dimension/representation depends on the goal. 
Transport route? $\rightarrow$ 1D
Google maps? $\rightarrow$ 2D

Evelation can be considered on the ratio scale at $(x, y)$-coordinates. [If we agree that sea-level would be the natural zero and we consider ratios from this level meaningful.]

### Geographic Information System

What is **geographic information system (GIS)**? Well, it does the following parts:

1. Data correction / representation
2. Analysis / querying
3. *Automated cartography:* Visualisation of results

**Geometry:** Coordinates.
**Topology:** Adjecency relations of objects.
**Attributes:** Anything else. Properties, values.

There are two types of representation of geometry:

- Raster
- Vector

#### GIS problems

Input: generally easy.
Output: Pretty difficult.

> Example: River label placement.
>
> What should exactly be the output? The first step is to look in literature what good places are.
>
> According to Imhof:
>
> - Not too close and too far
> - Not too curved
> - ...
>
> So now we have some idea, but what exactly is "too close" and "too far"?
> We can use the *directed Hausdorff distance* for this.

Given two lines.
**Hausdorff distance:** If I stand on one of the lines, how far should I walk to the other line in the *worst-case*. 

#### Useful geometric tools

Sometimes we can use some geometric tools as a blackbox, which has been solved in other particular fields.

Point location, check if point $p$ is within a polygon:

- *Naive:* $\mathcal O(n)$
- After some preprocessing: $\mathcal O(\log n)$.

A suitable triangulation of a planar point set can be computed in $\O(n \log n)$.
Computing the Voronoi diagram takes $\O(n \log n)$ time. [Nearest neighbour graph / proximity]

#### Trajectories

Model for the movement of a points / object:

- $f (\text{time})\mapsto $ 2D or 3D point.

There are various tracking technologies.

**GPS**

- Range: Whole world
- Precision: 2-10 meters in *lat-lon*.
- Sampling rate: Depends on device.

Typical assumption: *constant velocity*. Can even use statistical methods.

There are various kinds of questions that we can asks with trajectory. This also depends on the type:

- Single trajectory
- Two trajectories
- Multiple trajectories

##### Solving trajectory problems

Research involges 2+ stages:

1. Proper formalization of the problem
2. Development of useful, efficient algorithms
   - Also involves implementation, verification, feedback, ...


## Lecture 2 - Similarity

We'll only do *linear features* in this course.

### Trajectories

As a reminder, we can model movement by: $f (\text{time})\mapsto $ 2D or 3D point.
Data from GPS comes in a sequence of triples: $(x_i, y_i, t_i)$.

- We assume that velocity / speed is a piecewise constant function.

*Similarity* is the building block for many different trajectory algorithms.

There is **spatial similarity** and **temporal similarity**.

Measures of similarity:

- *Location* $\leftarrow$ Focus of this lectures
- Direction
- Turning Angles

### Various location similarities

#### Projection into high-dimensional space

Disadvantage: trajectories must have the same point and focus on the vertices only.

#### Hausdorff distance

Hausdorff distance: The longest *shortest distance*.

$$H(P \rightarrow Q) = \max_{p \in P} \min_{q \in Q} |p-q|$$

$d_H(P,Q) = \max [H(P \rightarrow Q), H(Q \rightarrow P)]$

Disadvantage: Fails to capture distance between the curves.

#### Aligning sequences via Time Series

**Dynamic Time Warping (DTW):** Change the allignment such that the trajectories lineup better.

**Euclidean Distance:** One-to-one alignments. Just straight line between trajectories.
**Time Warping Distance:** Non-linear alignments is allowed to lineup the trajectories better.

- Consider the *distance matrix*: a matrix with the differences between each point in time between two trajectory. 
  The lower left corner is the distance between first points and the upper right point is the distance between the last point. Instead of going diagonally by the one-to-one allignments (euclidean distance), we want to align the paths as much as possible.
- In particular, we can select a warping path $w$, we want to minimize the *Dynamic Time Warping*:
  - $DTW(Q, C) = \min \sqrt{\sum_{k=1}^K w_k}$

How to DP for DTW?

- Fill in the table and consider the three subproblems and grab the shortest dinstance from this.

Still, we also want to consider **temporal similarity**, as the DP solution can remove time.

- We can add time to the weight calculation, by e.g. as an extra dimension.
- Measure the distance between the warping path and the diagonal path, or to *limit warping* to points with similar time stamp.
  - Sakoe-Chiba Band: Straight band, the time cannot change more than a constant value.
  - Itakura Parallelogram: Sort of parallelogram, difference in time is not constant but changes over time.

*Drawback* of all temporal methods: non-metrics.

- Triangle inequality does not hold $\rightarrow$ Making clustering complicated

#### Fréchet distance

Fréchet distance measures the similarity of two curves.

Analogy

- Person (curve 1) is walking his dog (curve 2)
- Can vary their speeds [dynamic timewalking]
- Fréchet distance: **minimal leash length** necessary for both to walk the curves from beginning to end.

*Key differences* between this and timewarp:

- Fréchet distance uses maximum vs sum of DTW.
- Fréchet distance uses continuous distance vs discrete of DTW. 
  Although there is also a **discrete Fréchet distance**, but this may give large errors.

$$
\delta_F(P,Q) = \inf_{\begin{aligned}{\alpha:[0,1]\rightarrow P} \\{\beta:[0,1]\rightarrow Q}\end{aligned}}
$$

How to compute the continuous Fréchet distance? Consider the problem of the distance of two lines.

- **Decision problem:** $\delta_F(P,Q) \leq r$.
- Consider the continuous distance matrix $P$ and $Q$. Via circles we can compute for each point whether these are inside or the outside of the range. If we colour those white and the others red, then we create the **Free-Space diagram**. 
  - We can look if there is a path from the lower left corner to the upper right corner.

