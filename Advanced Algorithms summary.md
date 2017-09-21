# Advanced Algorithms

[TOC]



*Course code:* 2IMA10
$$
\texttt{LaTeX commands}
\newcommand{\O}{\mathcal O}
\newcommand{\txt}{\texttt}
\newcommand{\alg}{\txt{ALG}}
\newcommand{\opt}{\txt{OPT}}
\newcommand{\lb}{\txt{LB}}
\newcommand{\eps}{\epsilon}
$$

## Lecture 1

### Course overview

The course consists of three parts:

- Approximation algorithms
  For some problems, computing exact solutions is (NP-)hard.  Approximate solution might be much easier.
- I/O-efficient algorithms
  Running time is generally analyzed in terms of number of computational steps. However, what if the main constraint is memory or disks (I/O)?
- Streaming algorithms
  This is useful for when data comes in, in large amounts, on-the-fly.

### What is an approximation algorithm?

**Load-Balancing problem:** How can we compute an optimal workload distribution over all servers?

(Some stuff that I already know so I am not writing down)

Let $\text{ALG}$ be an algorithm for a given minimization problem.
$\alg(I)$ Output of algorithm
$\opt(I)$ = Optimal output of the problem.

For minimization problems:

- $\txt {ALG}$ is a $\rho$-approximation algorithm, for some $\rho>1$, if:

$$
\txt{ALG}(I) \leq \rho \cdot \txt{OPT}(I) \qquad \forall I \in \text{Inputs}
$$


For maximization problems:

- $\txt {ALG}$ is a $\rho$-approximation algorithm, for some $\rho<1$, if:

$$
\txt{ALG}(I) \geq \rho \cdot \txt{OPT}(I) \qquad \forall I \in \text{Inputs}
$$

### Load-balancing problem

Problem statement:

- Various jobs with each having different processing time.
- Number of machines, such that the *makespan* (= maximum load on any machine) is minimized.

Input:

- Set $\{ t_1, t_2, \ldots, t_n\}$ of processing times.
- Integer $m = $ number of machines.

This is **NP-Hard** even for two machines.

Greedy algorithm: go over jobs one by one, assigning job to machine of smallest load.
$\O(n \log m)$ if we store the loads in a priority queue.

#### Proving approximation - Algorithm 1

How to prove that $\alg(I) \leq \rho \cdot \opt (I)$ when we know $\opt(I)$?

- Find lowerbound $\lb(I)$ such that $\opt(I) \geq \lb(I)$.
- Prove that $\alg(I) \leq \rho \cdot \lb(I)$.

At least one machine has:

- $Load(M_i) \geq \frac 1 m \cdot \sum_{j=1}^n t_j$, since the maximum must at least be the average.
- $$Load(M_i) \geq \max_{1 \leq j \leq n} t_j$$, since we need to assign the maximum task.

Let us define some variables:

- $M_{i^*} :=$ Machine with the largest load. Therefore: $\text{makespan}=Load(M_{i^*})$.
- $J_{j^*} := $ Last job assignment to $M_{i^*}$.
- $Load^*(M_{i^*}) = $ Load on $M_{i^*}$ just before $J_{j^*}$ was applied.

Using some mathematics and the fact that according to the algorithm the last job that is assignment will have a smaller load than the average node, as we put the task at the last machine.

As shown in the slides, we can prove that this is a 2-approximation for the job.

---

**Theorem**

Greedy-scheduling is a 2-approximation algorithm.

-----

How to improve on this?

1. Use same algorithm and same lower-bound, but better proof.
2. Use same algorithm and different louwer-bound and perhaps a better proof.
3. New algorithm / lower-bound.

If we proof more carefully, we can prove that greedy-scheduling is a $(2- \frac 1 m)$-approximation, which we can also prove is *tight*.

**The** approximation ration = $$\min_\rho \rho-$$approximiation for $\alg$= $$\max_{\text{Inputs } I} \frac{\alg(I)}{\opt(I)}$$.

#### Proving approximation - Algorithm 2

What if we first sort on the largest jobs and then the smallest job? Intuitively, we get a much better result. In fact, we can prove that it's a $(3/2)$-approximation algorithm.

This is done by the fact that a machine has at least two jobs, which both are greater than the final job assigned.

## Lecture 2 - (Weighted) Vertex Cover + LP Relaxation

*This will come in the exam!*

### Vertex-Cover problem

Given, a graph $G = (V, E)$.
Goal: Minimize $|C|$ where $C \subseteq V$ such that each edge $(u, v) \in E$ we have $u \in C \vee v \in C$.

Algorithm for vertex cover, called `VertexCoverAlg`: 

- Uncovered edges? Add an enpoint of an uncovered edge to the cover.
- Runs in $\O(|V| + |E|)$ if implemented correctly.
- Worst Case scenario: can be an $(|V|-1)$-approximation.

#### 2-approximation

Idea: pick the vertex with the highest degree.

- Results in $(\log |V| - 1)$-approximation

Why not pick both? This is called `ApproxVertexCover`.

> `ApproxVertexCover`
>
> 1. Select an edge $(u, v)$.
> 2. Add $u$ and $v$ to the vertex cover and remove all edges adjacent to both $u$ and $v$.
>
> We can, in fact, prove that this is a 2-approximation!

Lower bound: consider the **disjoint edges**, there are edges that do not share an endpoint. (*Maximal matching*). 
Let $E^*\subseteq E$ be any set of pairwise disjoint edges. $\implies$ $\opt(G) \geq |E^*|$.

Notice that all selected edges which are removed *are* pairwise disjoint!
Since we are removing all edges incident to two vertices, we are removing at least two edges. Therefore, we know that $\alg(G) \leq 2 |E^*| $.

By the way, Vertex Cover is APX-hard, meaning that it's NP-hard to approximate lower than a certain constant.

### Brief introduction to linear programming

Example:

Minimize $3x_1 - 2x_2$

Subject to:
$$
\begin{align*}
2x_1 - x_2 &\leq 4 \\
x_1 + x_2 & \geq 3 \\
-0.5 x_1 + x_2 & \geq 3 \\
x_2 & \geq 0
\end{align*}
$$
Notice that the cost function and the constraint function must be linear.

We can actually rewrite it as matrices! 

Minimize $\bf\vec c ^T \cdot \vec x$
Subject to $A \cdot {\bf \vec x} \leq {\bf \vec b}$

----

**Theorem:** LP problems can be solved in polynomial time.

(Unless it's a 0/1-LP problem)

---



Suppose $P := NP$.

Then we can derive from this that $P == NP$.
Therefore, I have proven that $P = NP$.

### Weighted vertex cover

#### Solving via 0/1-LP

**Input:** Graph $G=(V,E)$, $weight() \mapsto V \rightarrow \mathbb R$
Each vertex now has a *cost*. 
**Goal:** Select minimum-weight $C \subseteq V$ such that for each edge $(u,v) \in E$ we have $u \in C \vee v \in C$.

Can we use our previous `ApproxVertexCover` algorithm? No.
Can we model this as an LP? Yes.

> `WeightedVC-LP`
>
> Introduce a decision variable $x_v$ where:
>
> - $x_v = 1 \Leftrightarrow$ we put $v$ into the cover 
> - $x_v = 0 \Leftrightarrow$ we do not put $v$ into the cover
>
> Computing an optimal result for LP $\equiv$ Best Vertex Cover
>
> Minimize: Total weight of vertices in $C$.
> Subject to: Vertices in $C$ form a cover.
>
> Total weight of vertices in $C = \sum_{v \in  V} weight(v) \cdot x_v$
> Vertices in $C$ form a cover: $\forall_{(u, v) \in E} :x_u + x_v \geq 1$
>
> However, we also need to add a 0/1-constraint for the variables! Therefore: $\forall v \in V: x_v \in \{0,1\}$.
>
> The cover can then be found by selecting all vertices where $x_v = 1$: $C \leftarrow \{ v \in V : x_v = 1 \}$

However, this is not normal linear programming due to the constraint, but instead is 0/1 Linear Programming. This is a problem, though...

---

**Theorem:** 0/1 Linear Programming is NP-hard.

---

#### LP Relaxation

*Problem:*  $x_v \in \{0, 1\}$ is not a linear solution.**
*Solution:* What if you instead use $0 \leq x_v \leq 1$?

But then, which vertices do we put into the cover? Just pick a threshold $T$. :D

Then: $C = \{v \in V: x_v \geq T\}$.

> **Lemma:** If we pick $T = 0.5$ then $C$ will be a valid cover.
>
> **Proof:** Since we have that $x_u + x_v \geq 1$, we have that either $x_u \geq 0.5$ or $x_v \geq 0.5$.
> If $T = 0.5$, then we put at least one vertex into $C$ for every edge.

#### Analysis of the Approximation Ratio

We need a lowerbound, but this is always the same for the LP relaxation. 

The optimal solution for the relaxed LP is at least as good as the optimal solution for the 0/1-LP, because we have more options at the relaxed LP as the constraints are less rigid.
$$
\opt_\text{relaxed} \leq \opt_\text{0/1}
$$
Notice that $\opt_\text{0/1} = \opt_\text{VC}$, therefore we have: $\opt_\text{VC}  \geq\opt_\text{relaxed}$  

> **Lemma:** `WeightedVC-LP-Relaxation` is a 2-approximtion.
>
> **Proof:**
> $$
> \begin{align*}
> \texttt{WeightedVC-LG-Relaxation}(G)
> &= \sum_{v \in C} weight(v) \\
> & \leq \sum_{v \in C} weight(v) \cdot 2 x_v & \text{[Since $C = \{v \in V: x_v \geq 0.5\}$]} \\
> &= 2 \cdot \sum_{v \in C} weight(v) \cdot x_v  \\
> &\leq 2 \sum_{v \in V} weight(v) \cdot x_v \\
> &= 2 \cdot \opt_\text{relaxed} \\
> &\leq 2 \cdot \opt_\text{VC}
> \end{align*}
> $$
>

### Guide for solving Problems using LP-relaxation

1. Formulate the problem as a 0/1-LP
   - Suitable decision variables
   - Define cost function + constraints.
2. Relax the 0/1 -LP
3. ...
4. ...
5. Profit!


## Intermezzo - exercise tips

What to do with hyperedges? Add a constraint, say for 3, of:
$$
x_u + x_v + x_w \geq 1
$$
Then the constant / threshold should be $1/3$.

What if I want at least 2 of the 3 in the cover?
$$
x_u + x_v + x_w \geq 2
$$
But now the rounding is a little more tricky. 

- What about $2/3$? Well, what about $x_u = 1/2$, $x_v = 1/2$ and $x_w=1$ situation? Then we only add 1 in cover. So what about $1/2$? Well, the aforementioned is the worst-case situation, thus this should work. 

We now know that at least one of the variables

## Lecture 3 - Polynomial-time approximation schemes

### Polynomial-Time approximation schemes (PTAS) definition

Is it possible to get a $\rho$ that's really close to 1?

Algorithm $\alg$ with input:

- Problem instance $I$
- Parameter $\epsilon > 0$

$\alg$ is a polynomial-time approximation scheme (PTAS) if:

- $\alg(I, \epsilon) \leq (1 + \epsilon) \cdot \opt(I)$ (for all inputs $I$, $\epsilon$.
- Running time is polynomial in size of input instance $I$, but also depends on parameter $\epsilon$.
- Example: $\O(n^{2/\epsilon})$ or $\O((1/\epsilon)^4n^2)$. Preferably we want the latter; polynomial in $1 / \epsilon$. (Fully PTAS, aka FPTAS.)
- Maximization? $\alg(I, \epsilon) \leq (1 - \epsilon) \cdot \opt(I)$.

### Knapsack

Yeah, I know what the problem is by now. :3

Problem: Get subset $S \subseteq X$ such that $value(S)$ is maximal while $weight(S) \leq W$.

Global strategy:

1. Replace the value of each item $x_i$ by a new value $value^*(x_i)$, which is a "small" integer.
2. Solve the problem optimally for the new values.
   - **Optional:** Solved by DP. Generally given as an algorithm.
3. Return the subset computed in Step 2 as a solution for the original problem.

#### Optional: Solving step 2 by DP

We can compute an optimal solution in $\O(nV_\texttt{tot})$ time.

Alternative formulation: compute largest $j \in \{1, \ldots, V_\texttt{tot}\}$ such that there exists a subset $S \subseteq X$ with $value(S) = j$ and $weight(S) \leq W$.

**Subproblems:**

For each $0 \leq i \leq n$ and each $0 \leq j \leq V_\texttt{tot}$, compute the minimum weight of any subset $S \subseteq S_i$ with $value(S) = j$.

- Suppose we know this, can we compute the next step?

> **Dynamic Programming tip**: do we have overlapping subproblems and can we define a recursion that makes use of these overlapping subproblems?

#### Solving step 1

Idea 1: if I want integers, let's just round to the nearest value. 

- $value^*(x_i) := \lceil value(x_i) \rceil$
- Does not work, because resulting values can be very large and no control over approximation ratio $\rho$.

Idea 2: round each value to  next multiple of $\Delta$. 

- We can then round each value by $\Delta$ to get small images.


- $value^*(x_i) := \left\lceil \frac{value(x_i)}\Delta\right\rceil$

For a PTAS, notice that: $\alg \geq (1- \epsilon) \cdot \opt = \opt - \epsilon \cdot \opt$, therefore $\epsilon \cdot \opt$ is the total error we can make, thus $\text{Total error} \leq \epsilon \cdot \opt$.

Note: error on each item value is at most $\Delta$.

We have that:
$$
\left.
\begin{aligned}
\text{Error in each item}\leq \Delta \\
\text{At most $n$ items in any subset}
\end{aligned}
\right\rbrace
\text{Total error} \leq n \Delta
$$
Thus, suppose we pick $\Delta = \frac \epsilon n \cdot \opt$, then we have that $\text{Total error} \leq n \cdot  \frac \epsilon n \cdot \opt = \epsilon \cdot \opt$, which is what we wanted!

However, we don't know $\opt$, so lets find a lower bound $\lb\ \leq \opt$.

- We know that $\opt \geq $ max item weight value. Thus let's select $\lb = \max(value(x_i))$. (Assuming that we throw away all $x_i$ with weights $> W$. 

Therefore: $\Delta = \frac \epsilon n \cdot \lb$.

##### Proof

To prove that an algorithm is a PTAS, we need to show that:

1. The running time is $\geq (1-\epsilon) \cdot \opt$.
2. Running time is polynomial in $n$.
3. Output is *valid*.

Let us define $S$ as our computed solution and $S_\opt$ the optimal subset. Thus, we want to show that: $value(S) \geq (1-\epsilon) \cdot value(S_\opt) $.

- Notice that $value^*(S) \geq value^*(S_\opt)$, because $S$ is an optimal solution for $value^*$. [1]
- Furthermore, note that: $\frac{value(x_i)}\Delta \leq value^*(x_i) \leq \frac{value(x_i)}\Delta  + 1$, since by definition $value(x_i) = \left\lceil \frac{value(x_i)} \Delta \right\rceil$. [2]

Derivation:
$$
\begin{align*}
value(S)
&= \sum_{x_i \in S} value(x_i) \\
& \geq \sum_{x_i \in S} \Delta \cdot (value^*(x_i) -1) & \text{By [2]}\\
& = \Delta\cdot \left(\sum_{x_i \in S} value^*(x_i)\right) - |S| \cdot \Delta\\
& \geq \Delta\cdot \left(\sum_{x_i \in S_\opt} value^*(x_i)\right) - n \cdot \Delta & \text{By [1] and $|S| \leq n$}\\
& \geq  \sum_{x_i \in S_\opt} value(x_i) - n \cdot \Delta &\text{By [2]}\\
&= value(S_\opt) - n \cdot \Delta \\
&=  value(S_\opt) - \epsilon \cdot \lb & \text{By definition of $\Delta$} \\
& \geq value(S_\opt) - \epsilon \cdot \opt & \text{Since $\lb \leq \opt$}
\end{align*}
$$
Idea behind derivation:

1. You first start to look from $value(S)$ to $value^*(S)$. 
2. Then change $S \rightarrow S_\opt$.
3. Then we change $value^*(S_\opt)$ to $value(S_\opt)$ again.

##### Running time

For the second step, how large can $V_{tot}$ be?

- After some math: $\O(\frac{n^2}\eps)$.

We execute this a total at most $n$ times, resulting in a running time of $\O(\frac{n^3}\eps)$. 

## Lecture 4 - Exercises

**Exercise 3.4:** Show that for any $n>1$ there is a graph $G$ with $n$ vertices such that the integrality gap of the LP for Vertex Cover is $2 - 2/n$.
**Solution:** Consider $K_n$. 
Then the ideal solution for the 0/1 LP is $n-1$, because if we exclude two vertices from the vertex cover then there $\exists$ an edge which is not covered. $\opt_{0/1} = n-1$.
However, putting every vertex on $1/2$ is also a valid solution. This results in $\opt_\texttt{relaxed} \leq \frac n 2$. Then:
$$
I.G. \geq \frac{n-1}{\frac n 2} = 2 - \frac 2 n
$$


**Exercise 4.5:** Euclidean TSP
**Input:** $n$ points in the plane: $(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)$.
**Output:** Shortest path over all vertices such that we end up on the same vertex.

Approach: 

1. Round coordinates to next multiple of $\Delta$, for a suitable $\Delta$.
2. We divide all coordinates by $\Delta$.
3. Run `IntegerTSP` on new coordinates.

What is a suitable $\Delta$?

- Note that: Total Error $\leq \eps \cdot \opt$.
- What is the maxium change in length per edge? This is the *error per edge* and this is $\leq \sqrt 2 \cdot \Delta$.
- Number of edges in a TSP is $n$, therefore: Total Error $\geq n\sqrt 2 \cdot \Delta$. 

Thus, we know that $n \sqrt 2 \cdot \Delta \leq \epsilon \cdot \opt$, thus, $\Delta \leq \frac \eps {n \sqrt 2} \cdot \opt$.
Now we need to obtain a $\lb$ such that $\opt \leq \lb$.

Now there are two different kind of lengths:

- $length(p_i, p_j)$, which is the length of edge $(p_i, p_j)$
- $length^*(p_i, p_j)$, which is the length of edge $(p^*_i, p_j^*)$.

Note that $length(p_ip_j) - \sqrt 2\cdot \Delta \leq \Delta \cdot length^*(p_ip_j ) \leq length(p_ip_j) + \sqrt 2 \cdot \Delta$.

Now, our goal is to find a derivation such that:

$length(T^*)\leq \cdots \leq length(T_\opt) + \eps \cdot \lb \leq (1+\eps) \cdot \opt$.

We do so as follows:
$$
\begin{align*}
length(T^*)
&= \sum_{(p_i, p_j) \in T^*} length(p_ip_j) \\
&\leq \sum_{(p_i, p_j) \in T^*} \Delta length^*(p_ip_j) + \sqrt 2 \cdot \Delta \\
&= \Delta \left(\sum_{(p_i, p_j) \in T^*} length^*(p_ip_j)\right) + n\sqrt 2 \cdot \Delta \\
&\leq \Delta \left(\sum_{(p_i, p_j) \in T_\opt} length^*(p_ip_j)\right) + n\sqrt 2 \cdot \Delta \\
& \leq \sum_{(p_i, p_j) \in T_\opt} \left(length(p_ip_j) + \sqrt 2 \cdot \Delta\right) + n \sqrt 2 \Delta \\
& = \sum_{(p_i, p_j) \in T_\opt} \left(length(p_ip_j)\right) + 2 n \sqrt 2 \Delta \\
& \leq length(T_\opt) + 2 \eps \cdot LB
\end{align*}
$$
Hmm, not exactly what we hoped for. What if we take $\Delta \gets 2 \Delta$? Then we get 