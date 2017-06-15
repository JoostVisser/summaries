# Research methods

[TOC]

## Lecture 1

### Computer science

What exactly is **computer science**? It actually is a real broad topic that touches on many different areas:

1. Problem solving / computation
2. (Electronic) hardware and software
3. Information / data
4. Life

Another nice summary would be from the Wikipedia definition:

- Theory
- Experimentation
- Engineering
- Art

It differs in various topic from other sciences and mathematics, but it actually is a unique combination of multiple questions.

### Research questions

So, we want to do some research in Computer Science. A research paper contains a question that we want to answer.

These are some properties that the research question should adhere to:

1. Falsifiable
   - "Does God exists?" is a bad research question as this can be neither proven nor be disproved. 
2. Concrete
   - Put numbers on it, tell which constraints there are under which conditions, etc.
3. Measurable

There are many statistical traps when doing research. As an example, one can do 20 different studies and most likely one study will contain a link with $(p < 0.05)$. There are some things that can help with this:

- Formulate the research question in advance
- Try to disprove your proven theory
- Replication studies

## Lecture 2

**Goal:** $\Pr[\text{My claim is true | observation}]$, or in another way: $1- \Pr[\underbrace{\text{opposite}}_{H_0} \text{ is true | observation}]$

### Overview

Source of materials for papers:

- Books
  - Text books
  - Monographs
- Wikipedia
- Papers
  - Surveys (Useful for a summary of the current techniques.)
  - Original Research
    - Journals (Useful for the state-of-the-art)
      - Quite some time to review a paper
    - Conference proceedings $\leftarrow$ Peer reviewed
      - Friendly request to fix it, but nobody checks if it's fixed.
      - People only have little time to check the paper.
    - Workshops "proceedings" $\leftarrow$ Not/hardly peer reviewed
  - White papers
- Popular media
- Lecture notes + Slides
- PhD + Master + Bachelor theses
- Course work
- Technical reports (e.g. arXiv)


### Reliability of the various sources

#### Journals and conference papers

Journals: there are two experts that check your work over a long period of time. Then if it's accepted, there probably have to be some fixes via the review. Once this has been fixed, it needs to be fixed again.

Conference proceedings are too time constrained, as people only have little time to check the paper. Furthermore, there is only a friendly request to fix it, but nobody checks if it's fixed.

- More focussed on interestingness and relevancy, not so much as it is correct.

If you can and found a conference proceedings, try to find a journal with the same topic as it might be better.

#### Wikipedia

Wikipedia

- Pros: has many different users, as well as details and references 
- Cons: There are some sketchy wikipedia pages and the page itself can be changed, so try to find the reference itself.


#### Thesis and monographs

Thesis:

- Good thing: It has been thoroughly checked by the teacher.
  PhD also checks of whether it's research worthy.


- Bad thing: It might have gotten a lower grade which you don't know. 
  The job is not to check whether it's worthy of publishing.
  PhD is hundreds of pages and you cannot count that all details are fixed. (Be careful!)

Monographs are like PhD thesis but without the professor reviewing it thoroughly. It might be reviewed but this can be not independent. Don't count too much on too much back and forth reviewing. 

#### Books, lecture notes, and technical reports

For books - Check for the 2nd edition or something like that.

Lecture notes: sometimes helpful to get some nice figures and references. But might contain mistakes, can disappear so are not the most stable sources.

Technical reports: contains more information than journals/proceedings, but it is not peer-reviewed, although in arXiv there are endorsements.

#### Blog and youtube

Blog & Youtube: Look at the author. If the author has a good track record, then it might be considered reliable. Otherwise don't trust it too much.

- DBLP bibliography

#### Proceedings, course work, and the rest 

Workship "proceedings": look at the number of pages. If there are little pages (4), then there is probably no peer review, but if it was say 10 pages then it might've been properly peer-reviewed.

Course work, never rely on this because:

1. You don't know what grade it got.
2. You don't know the criteria.

Papers from different language: very unreliable if you cannot read the paper and verify what's in there.

### Signs of reliable work

Signs of reliable work

1. References
2. Detail
3. Number of users
4. TeX

## Lecture 3 - Experiments

> Small note
>
> This does not hold for all Computer Science projects, but just for the typical ones.
>
> An example of where this does *not* hold is a better worse-case bound for $k$-path algorithm. This does not follow the framework.

### The Theory-Hypothesis-Experiments framework

Most projects involve a mix of the following three elements:

1. Hypothesis
2. Theory
3. Experiments

In general, the order is as follows:
Theory $\longrightarrow$ Hypothesis $\longrightarrow$ Experiments $\longrightarrow$ Theory $\longrightarrow \cdots$

But where do we start? This actually can go in all three of these elements of the framework.

### Experiments

How do we perform an experiment, given an hypothesis? There are roughly four steps in creating an performing an experiment:

1. Design
2. Implementation
3. Data Acquisition / execution
4. Aggregation / analysis / evaluation

From steps 2, 3, and 4, something can go wrong / overlooked so then we can go back to designing our experiment again. This usually happens.

*Side note:* Do not underestimate the time it takes for the experiment.

#### Useful tips for experiments

It's useful to have a *baseline* to compare to. Usually, this should be the state-of-the-art solution. Preferably, we want the code for the state-of-the-art solution. 

- The state-of-the-art solution needs to be *available* and takes time to get.

Furthermore, we need to be able to collect and retrieve the *data*.

- The data must also be *available*.
- Think of stress test inputs or edge cases.
- Random input. Distribution should be similar to the scenario we care about.

##### Implementation tips

The experiments should be **reproducible**, **feasible**, and should be *interesting*, *new*, and *relevant*.

Another tip, suppose you have two implementations and you want to test there, then:

1. When you generate random data, **save** that data for reproducibility.
2. Use the **same data** for both the implementations.
3. Think about what you want to measure. Worst-case query time or median query time?

Think of the performance of the measurement. Only false positives? Or perhaps the *F-score*.

##### Measurement tips

- Check that your performance measure is the actual measure that you want to test and is not underwhelmed by other factors. 
  Especially important for *usability* measurements. (Measure users, don't ask.)
- Measure one thing at a time.

##### Verification

It's important to **verify** if the output of your algorithm is correct. 

##### Questionnaire tips

You want to make the response as high as possible to get a better representation.

1. Make it short.
2. No leading questions.
3. No ambiguous or vague questions.
4. Make sure that people can give any possible answer in case of non-grading.

Also, sometimes people don't want to answer "delicate questions", such as someone's income. 

It helps to tell what the questionnaire is used for to get a better response.

Finally, design, implement and test your questionnaire to make sure that you get the answers that you want to have. :)

:dead::do_not_litter::bomb:

## Lecture 4 - Statistics

Suppose we have hundreds of experiments. It might not be a good idea to include all of them in the master thesis, so perhaps we can just post statistics in the master thesis.

When we only give some statistics such as *mean*, *variance*, and *correlation coefficient*, then we still don't know that much about the underlying distribution, as shown with the Anscombe's Quartet. 

- Statistics do not tell the whole story

Statistics 101:

- **Population** - The whole population that ideally, but non-realistically, 
- **Sample** - Subset of the population for which we test something.
- **Probability Distribution**
- **Random variable** - A non-deterministic variable. This variable can take up various values, this depends on the distribution of the random variable.

Suppose we have two algorithms which we rank. For 50% of the cases algorithm A beats algorithm B and vice versa. Does that mean that they're equally good?

- We cannot state that. Perhaps algorithm A performs almost as good as algorithm B for 50% of those cases, but for the other tests algorithm B is much worse than A, than algorithm A would be a better choice than algorithm B.

**Categorical data:** Just labels
**Ordinal data:** Ranking of data: $1 < 2 < 3 < 4$, but the difference have no meaning.
**Interval data:** Difference between ranks is meaningful: $2-1 = 3-2$. Ratios, however, are not meaningful. Example: 23 May, 24 May, 25 May.
**Ratio data:** Same as interval data but ratios are always meaningful.

Useful things to know about a distribution:

- Measure of location [Always]
  - Mean (for Interval data + Ratio data) || median (for ordinal, interval or ratio)
- Measure of dispersion [Always]
  - Variance
- Measure of shape of distributions [Sometimes]
  - Kurtosis + Skewness - Although it might be better to just draw the distributions.

*Reminder:* It doesn't make much sense to report more digits than indicated of the standard deviation. Rule of thumb: first significant digit of standard deviation = last of the mean.

*Another note:* Remember that some numbers have to be comparable. If you test with different input sizes, then it doesn't make sense to take the mean running time over all different input sizes.

## Lecture 5 - Showing statistical data

### Design principles for good visualisation

Graphics displays should adhere to the following characteristics:

- Show the data (duh)
- Viewer thinks about the data itself, rather then the methodology / graphic design.
- Avoid distorting what the data wants to say
  - No cheating, e.g. removing removing the scales.
- Present many numbers in a small space. && Make large datasets coherent.
- Encourage the eye to compare the different pieces of data.
- Reveal the data at *several levels* of detail. [Big picture + more detail]
- Should have a clear **purpose** - Is it for exploration? How about description?
- Write in the text something about it.

There are some actual data principles:

1. Maximize data-ink ratio - As much data shown with as little ink as possible.
2. No chart junk (3D stuff etc)
   - No unjustified 3D
     - Research has shown that for planes, we can compare positions on common scales really well, unaligned scale and length decently well, angles a bit hard.
       Really difficult is Area (2D size) and Depth (3D position).	

Other important characteristics:

1. Focus on the *core message* - Don't show things that aren't important for the diagram.

**Lie factor:** $\frac {\text{Size of effect shown in data}} {\text{Size of effect in real world}}$

## Lecture 6 - How to write a Master-like thesis

### My first Master thesis!

**Goal:** We want the readers to read our Master thesis. If we can **entertain** the reader to read our research, then it perhaps becomes more popular & more impact to society.

How to do so?

1. Sense of **purpose** - Is it relevant to the research question? 
2. **Compelling** problem
   - Challenging *puzzle* - Some readers like this.
   - *Societal* relevance
3. Solid connection to the **literature** and to the **recent developments**.

Master thesis: "Telling your story 5 - 6 times with increasing level of detail."

- Title $\leq 12$ words
- Abstract $\leq \frac 1 2 $ words
- Introduction $\leq 3$ pages
- Preliminaries
- "Body"
- Conclusion $\leq 3$ pages

|               | Content | Problem | State-of-the-art | Solution | Results |
| ------------- | :-----: | :-----: | :--------------: | :------: | :-----: |
| Title         |         |   ++    |                  |    +     |    +    |
| Abstract      |   ++    |   ++    |        ++        |    ++    |   ++    |
| Introduction  |   ++    |   ++    |        ++        |    ++    |   ++    |
| Preliminaries |  (++)   |  (++)   |       (++)       |          |         |
| Conclusion    |   ++    |    +    |        +         |    ++    |   ++    |

#### Title

Consider the following title: 
"On an in-depth study of a hybrid approach to *water management in the Dommel basin*."

- "Water management in the Dommel basin" - Vague and too specific
- "Hybrid approach" - Not really anything novel.
- "On an in-depth study" - Remove "on" and also "in-depth study" (of course it is!).

#### Abstract

Has to be very concrete, with 1 - 2 sentences about each topic.

#### Introduction

Has to be **self-contained** and discuss all the things in a bit more detail.

- 1-2 paragraphs each.

In the **solution** paragraph(s) in your introduction, you will need to talk about *what's new*.

#### Preliminaries

More details about one-or-more of these things.

#### Body

This should follow a different structure and really depends on the research done.

Should contain **enough detail** for somebody else, someone of the same skill-level, to **reproduce** the experiment.  

#### Conclusion

Personal preference of the teacher: go to the elements in the reversed order.
Results $\rightarrow$ Solution $\rightarrow$ State-of-the-art $\rightarrow$ Problem $\rightarrow$ Context

- This is **not** a summary of the rest of the paper.
- We want to **reflect** on the results. Also contains *future work*.

#### Master-thesis only

What is the difference between the Master-thesis and the research paper?
Purpose of research paper: Demonstrate results.
Purpose of Master-thesis: Demonstrate you're a competent student.

- Even without results you can still give an in-depth discussion about the research.

## Lecture 7 - Ethics

**Kant** 

- If you apply the action in general, then it shouldn't lead to inconsistencies.
- Do to others as you would like them to do to you.

**Utilitarianism**

- Strife for maximum utility / maximum happiness in the world
- Has some limitations, with regards to e.g. gladiator fighting or the fact that there is no self-focus on happiness, but everything for the greater good.

**Virtue ethics**

- Strive for a better you

### TU/e code of scientific conduct

#### Trustworthiness

Students + Staff ground their views on *scientific evidence*.

Kinda looks *Kant*-ish.

#### Intellectual Honesty

