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
  - We can look if there is a path from the lower left corner to the upper right corner. This needs to be *continuous* and *monotone*.
  - $q_1 \leftrightarrow p_1$ and $q_m \leftrightarrow p_n$


What if there more than two lines? Well, then we can duplicate such block *for each line*! 

- Suppose $P$ has $k$ line segments and $Q$ has $l$ line segments, then there will be $k \times l$ blocks.

> **Algorithm - Fréchet distance decision problem**
>
> 1. Compute free-space diagram in $\O(mn)$ time.
>
>    - One cell can be constructed in $\O(1)$ time
>
>
>    - Square and ellipsoid intersect in at most 8 critical points, we only need those points.
>
> 2. Compute a monotone path from $(q_1, p_1)$ to $(q_m, p_n)$ in $\O(mn)$ time.
>
>    - Idea: find the *earliest reachable point* that can be reached with a monotone path from $(q_1, p_1)$. There are 3 cases:
>      - All free space on the right side and higher, then indicate 2 points.
>      - Free space is both higher and lower than current space, then the lowest points should be at least on the same hight.
>      - Free space is lower than current space, then it's impossible to reach this part.

With the above algorithm, how do we compute the exact Fréchet distance?

- In *practice:* determine $r$ bit by bit via binary search.
- In *theory*: there are three cases that define the minimum distance of $r$:
    1. $r$ is minimal with $(q_1, p_1)$ and $(q_m, p_n)$
       - Constant time
       2. $r$ is minimal when a new vertical or horizontal passage opens up between two adjacent cells in free space.
      - $\O(mn)$ events
        3. $r$ is minimal when a new vertical or horizontal passage opens up between two non-adjacent cells in the free space. (Measure where vertical value of point $t$ .
      - $\O(m^2n + mn^2)$ possible events. Via parametric search $O(mn \log mn)$.

## Lecture 3 - Simplification

**Observation:** We can simplify many trajectories without losing much information.

- Depends on application, of course.

We want a trajectory with the smallest number of vertices and with error at most $\epsilon$. We cannot use simple line-simplification though, because that makes the constant velocity assumption invalid.

- Various error measurements. We'll start with *curve simplification*.

There are three algorithms we will discuss for **simplifying polygonal curves**.

### Curve simplification

#### Ramer-Douglas-Peucker

**Input:** polygonial path $p=\langle p_1, \dots, p_n\rangle $ and threshold $\epsilon$.

> **Algorithm**
>
> ```python
> DP(P, i, j):
>   Find the vertex v_f farthest from (p_i,p_j)
>   dist := distance between v_f and p_i, p_j
>
>   if dist > epsilon then
>       DP(P, i, f)
>       DP(P, f, j)
>   else
>       Output v_i, v_j
> ```

Worst-case complexity: $\O(n^2)$, but can use $\O(n \log n)$ or even $\O(n \log^* n)$ in non-intersecting cases.

Con: does not give a bound on the complexity of the simplificaiton.

#### Imai-Iri

**Input:** polygonial path $p=\langle p_1, \dots, p_n\rangle $ and threshold $\epsilon$.

> **Algorithm**
>
> 1. Build a graph G and find all possible shortcuts.
>    - Check for all points if they are are above the threshold $\epsilon$.
> 2. Find shortest path using these shortcuts via BFS.

Brute force running time? $\O(n^3)$ as there are $\O(n^2)$ possible shortcuts and $\O(n)$ per shortcut.

- There are improvements for $\O(n^2)$ running time.
- Idea: test all shortcuts from a single vertex in linear time, by creating circles around the next points and the lines have to go through all the circles.
  - We're done with a point once we find a line between two circles.
    - We should check a ray both from $p_i$ to $p_j$ and vice versa! (Because otherwise if a route goes first to the right and then to the left it can skip some points as some disks are intersected by the ray but just one-way.)
  - Checking if a **ray intersects all disks** can be done efficiently.
    - The algorithm works by looking at planes, rays and wedges. Each time we wedge, we can reduce $W_j$ and compute the wedge. If the area becomes empty, then we are finished as there are no more rays we can shoot that intersects the rays. 
- We can't do better than $\Omega(n^2)$. 
- Pro: output with the *minimum number of edges*.

#### Argarwel et al.

Running time: $\O(n \log n)$.
Measure: Fréchet distance.
Output: path has at most the same complexity as a minimum link $(\epsilon / 2)$-simplification.

**Input:** polygonial path $p=\langle p_1, \dots, p_n\rangle $ and threshold $\epsilon$.

Let $\delta(p_ip_j)$ be the Fréchet distance between the line $(p_i, p_j)$ and the subpath $\pi(p_i, p_j)$.

> **Algorithm**
>
> Find *any* $j > i$ such that $\delta(p_i, p_j) \leq \epsilon$ and $\delta(p_i, p_{j+1}) > \epsilon$.
>
> - Exponential search: Test $p_{i+1}, p_{i+2}, p_{i+4}, p_{i+8}$ until found $p_{i+2^k}$ such that $\delta(p_i, p_{i+2^k}) > \epsilon$.
> - Binary search: Search for $p_j$ between $p_{i+2^{k-1}}$ and $p_{i+2^k}$ such that $\delta(p_i, p_j) \leq \epsilon$ and $\delta(p_i, p_{j+1}) > \epsilon$.
>   - This must me the case, because at the exponential search such jump happened.

If we do a lot of searching, then it must be because of a large jump so we perform a large shortcut. This helps with the running time. What's the running time? 

- The searching takes $\O(\log n) + \O(\log n)$ time.
- After some math, sums up to $\O(n \log n)$.

Prove of the approximation of the simplification is given in the slides. 

#### Adding time

So... how can we add time? 

- Just add a third temporal-dimension and check if the beams go through spacial disks or diamonds.

## Lecture 4 - Segmentation

### What is segmentation?

**Segmentation:** Split into segments that have meaning.

- Different modes of transportation of a puny human.

**Input:** Trajectory with geometric attributes.
**Criteria:** Bounded variance in speed, curvature, direction, distance, ...
**Aim:** Partition $T$ into a minimum number of subjtrajectories (*segments*) such that each segment fulfils the criteria.

**Decreasing monotone criteria:** If the criterio holds on a segments, then it holds on any subsegments. 

- Examples: bounded speed, bounded angular range (*heading*).

Note that a greedy algorithm works for any decreasing monotone criteria.
Designing greedy algorithms: See the slides for more information.

If we have a constant-update criteria which we can check, such as speed, then we can obtain the segments in $\O(n)$ time. 

What if there is no constant update criteria, but we have to check for each $k$ points to see if it a valid segment in $\O(\log k)$ time? Well, first idea might be binary search, but this takes $\O( n^2 \log n)$ time.

**Observation:** *Iterative double & search* is faster, which is an exponential search followed by a binary search. This restricts the search area and results in an overall running time of $\O(n \log n)$.

### How can we use segmentation?

We can search some trajectory with several criteria to look for certain properties, i.e. stopover of geese by looking at *Radius within 30km* + *Within 48h*.

**Observation:** For a combination between decreasing monotone & increasing monotone criteria then we cannot always use a greedy algorithm.

Non-decreasing monotone:

- Minimum time
- Standard deviation

Solution: **start-stop diagram**.

Given a trajectory $T$ over time interval $I = \{t_0, \ldots, t_\tau\}$and criterion $C$.

The **start-stop diagram** $D$ is the upper diagonal half of the $n \times n$ grid where each point $(i, j)$ is associated to segment $[t_i, t_j]$ with:

- $(i, j)$ is in free space if $C$ holds on $[t_i, t_j]$
- $(i, j)$ is in forbidden space if $C$ does not hold on $[t_i, t_j]$. 
- A minimal segmentation of $T$ corresponds to a (min-link) staircase in $D$.

We can solve this by *dynamic penetration*! 
$$
L(j) = \min _{\substack{i < j \\ \texttt{with $j$ reachable from $i$}}} (L(i)) + 1
$$
Results in a $\O(n^2)$ algorithm. Can we do better?

**Stable criteria:** A criterion is *stable* iff $\sum_{i=0}^n v(i) = \O(n)$ where $v(i) = $ # of changes of validity on segments $[0,i], [1, i], \ldots, [i-1, i]$.

- Increasing monotone and decreasing monotone are *stable criteria*.

- For stable criteria the start-stop diagram can be compressed by applying *run-length encoding*.

- The encoding can be calculated by a a greedy algorithm

  - This is done by two points and finding the staircase.

  1. First we try to extend the segment. 
  2. Then we try to check if it's a valid segment.
  3. And we need a *shorten* operations.

  - A balanced binary search tree can be used. $\O(\log n)$ is efficient enough.

- The start-stop diagram of two criteria is their intersection.

How to improve the $\O(n^2)$ time? We want to handle a couple of white cells per block. We can store these blocks in an **augmented binary search tree**.

1. Choose an underlying data structure. (RB-tree)
2. Determine additional information to maintain. (min-count[x])
3. Verify that we can maintain additional information for existing data structures.
   - **Theorem:** if the additional information only depends on left[x] and right[x], then we can augment this in $\O(\log n)$ time.

This algorithm runs in $\O(n \log n)$ time.

### Lecture 5 - Static Map Labelling



There are various guidelines that have been found after experience with manual map labelling.

- Automatic label placement can be easily modelled as a computational geometry problem, as it just a sum of various features of the guidelines of the readability of the map.
- We assume that the type of labels are given to us and that we just have to position them. (*Geometric placement*)

**Input:** Set of $n$ points in the place, for each point a label represented by a rectangle. (*Bounding box*)
**Goal:** There are various different objectives:

- Find a valid[^1] labelling for a maximum size subset of the points such that no two lables intersect. (`MaxNumber`)
- Find a valid[^1] labelling with all labels such that no two labels intersect + font size is maximized. (`MaxSize`)

#### Part I - Complexity

**Goal:** show that MaxNumber is NP-complete in the 4-slider model.
**Idea:** We can reduce the 4-slider problem into `Planar 3-SAT`.

**Clause graph:** each clause connects all the variables in terms of edges.

- Each variable is a vertex and each clause is a vertex.
- The clauses are connected with edges to the vertices of the variables (which are in the middle).

#### Part II - Sliding vs fixed positions

How many more points can be potentially labelled with sliding vs fixed?

For unit-size squares, the 2-slides model somtimes allows $\leq 2 \times \#\text{labels of 4-position model}$.

- Proof is done by intersection of odd and even lines of size slightly bigger than unit-size squares. (Showing *never more than twice*)
- This is also tight, as wel can show a 2s solution that has twice as many solutions as 4p model.

#### Part III - Approximation Algorithms

Approximation algorith for 4-position:

- Want to grab the maximum subset such that they don't intersect.


- In other words:We want the maximum independent set in the *rectangle intersecting graph*.
  - We're making a node for each label candidate and connect the node if the rectangles intersect. 
    - Also, for each point, we draw a complete $K_4$ for each node such that only one can be added.

Let's make the problem easier:

- Assume labels have equal height but varying width.

##### Equal height, varying width

Heuristic 1: randomly grab 1 and remove the rest, then do this again.

- Approximation ratio: $\Theta(1/n)$.

Heuristic 2: Choose the shortest label, eliminate the intersecting candidates and repeat.

- Approximation ratio: $\Theta(1/4)$, since any intersection must intersect one of the corner.

Heuristic 3: Greedy, left-to-right placement. [Leftmost right side]

- Approxmation ratio: $\Theta(1/2)$
  - Everything has to intersect the right two corners. 
- **Reference point:** lower left corner of label.
  - Mini-heap to find leftmost candidate
  - Priority search tree to find all reference points of candidates (*useless candidates*) 
  - $\O(n \log n)$

(Does not work for varying height.)

In fact, there exists a PTAS such that, for any $k > 1$, a $\frac k {k+1}$-approximation exist in $\O(n \log n + n^{2k-1})$.

Another idea, assuming labels have $1$ height:

- Draw horizontal lines such that
  - Separation between any two lines $> 1$
  - Each line intersects at least one rectangle.
  - Each rectangle is intersected by some line.
  - This can be done greedy top-down by looking at the first ending which has not been intersected yet. 
- Compute MIS for the rectangles for each line.
- Return the max of the MIS of $L_1, L_3, L_5, \cdots$  and MIS of $L_2, L_4, L_6, \cdots$.
- 1/2-approximation

What, instead of per 1 line, we solve everything per two lines?

- $L_1 \cup L_2, L_2 \cup L_3, L_3 \cup L_4, \cdots$.
- Note that $L_1 \cup L_2$ and $L_4 \cup L_5$ cannot have intersections, so we only remove every third line.
  - In the end, take the maximum of the three different cases.

#### Lecture 5 - Footnotes

[^1]: This depends on the model chosen. Are we using only the upper right part? Or perhaps all 4 corners. Or pehaps even a slider model where it can be everywhere above. The former is called 4P, the latter 4S.

### Lecture 6