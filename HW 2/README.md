# CMPE260 Assignmnet 2
In this assignment you will implement policy and 
value iteration to optimally solve a simple maze. 
Then, you will use Q-learning to solve the same maze from samples.

### Goals
* implement value iteration
* implement policy iteration
* implement Q-learning with epsilon-greedy policy


### What to submit
* your notebooks
* a pdf with your results


### Activities

Part 1 (use a2p1.ipynb):
1. Add a wall with length of 3 cells at an arbitrary  position either vertically or horizontally, or both. Make sure there is still a path from the START to the GOAL.<br>
[DONE] 
2. Visualize the maze at the beginning verify your wall cells are where you want.<br>
[DONE] 
3. Implement Policy Iteration.<br>[DONE] 
4. Implement Value Iteration.<br>[DONE] 
5. Test if your implementations. Policy iteration takes some time to run. Its a good idea to reduce the number of iterations and evaluations to 5-10 to see if your implementation is working.<br>[DONE] 
6. Run Policy Iteration and Value Iteration with 0 noise, record your output to the pdf. Visualize 5 iterations of value and policy for both algorithms showing thier progress.
7. For each algorithm, explain the difference between iterations. <br>[DONE] 
8. Run Policy and Value Iteration with noise, record valua and policy from 5 iterations to the pdf.<br>
[DONE] 
9. Write down your observations regarding the difference between running with noise and without noise.<br>
[DONE] 
 
Part 2 (use a2p2.ipynb):
1. Add a wall with length of 3 cells at an arbitrary position either vertically or horizontally, or both. Make sure there is still a path from the START to the GOAL.<br>[DONE] 
2. Visualize the maze at the beginning verify your wall cells are where you want.<br>[DONE] 
3. Finish the step function.<br>[DONE] 
4. Finish the choose_action_epsilon function.<br>[DONE] 
5. Implement the Q-learning update.<br>[DONE] 
6. Save five images showing the training progress of the q-learning agent. Discuss the differences between the images.<br>[DONE] 

```python

```
