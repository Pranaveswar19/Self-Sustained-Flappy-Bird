# Self-Sustained-Flappy-Bird
Without any human interaction, the machine plays the popular game "Flappy Bird" with the help of NEAT Algorithm. We delve deeper into the fundamentals of NEAT in this project.

## Description
In this project we are going to create a program that will run the Flappy Bird game and not only that but it will also play by itself. 

With the help of NEAT Algorithm which is a method used in machine learning for evolving artificial neural networks, the machine creates number of networks and evolves by selecting the best performing network which is then mutated to perform even better in each iteration.

Python language is primarily used in this project. By using Pygame we create a realistic environment and simulate realistic physics.

## Getting Started
### Installing required libraries
In the latest version of Python(3.12.5) you need to install these following dependencices:
```python
pip install pygame neat-python visualize --quiet
```
### Required Files
Now that we have installed the libraries, we requre the two following files which are important:

1. `Flappy.py`: This file consists of main code which includes the game and NEAT Algorithm integrated. Install this file in the same location as other files are to keep everything organized.
2. `configure-feedforward.txt`: This file contains the configurations for the NEAT Algorithm. It is important to note that these files must be easy to locate in the directory.
3. `png`: Install all of the png files in the same directory which include `Flappy.py` and `configure-feedforward.txt`.

These all are the requried setups that you need to do before running the actual program.

## Running the code
After installation of the above files, open the `Flappy.py` in your prefered text editor to run the program and insert paths of the requried files in "line 20" and "line 389".

Now just RUN the program and you are good to go!

## **Understanding `Flappy Bird.py`**

The `Flappy Bird.py` file is the main Python script responsible for running the **Flappy Bird** game simulation. Its primary purpose is to manage game mechanics, such as the movement of the bird, pipes, and base, while incorporating the **NEAT** (NeuroEvolution of Augmenting Topologies) algorithm to evolve the bird's neural network over successive generations. The script can train the bird to learn how to play the game autonomously using reinforcement learning principles. Below are the main steps and components of the script:

---

### **Purpose of the File:**
The primary purpose of the `Flappy Bird.py` file is to combine the **Flappy Bird game mechanics** with the **NEAT algorithm** to evolve and optimize the bird's ability to navigate through pipes over time. The game involves:
1. Displaying the bird, pipes, and base on the screen.
2. Allowing the bird to "jump" or flap to avoid obstacles (pipes).
3. Using NEAT to train the bird's AI by adjusting its neural network based on its fitness, which is evaluated by the distance traveled and pipes passed.

### **Main Steps Involved:**

1. **Initialization of Game Components:**
   - The game starts by importing necessary libraries like `pygame`, `os`, and `neat`. 
   - Essential game variables such as `WIN_WIDTH`, `WIN_HEIGHT`, and various images (bird, pipe, background, and base) are loaded and scaled to fit the display window.
   - Fonts and game window captions are initialized using `pygame`.

2. **Defining the Bird Class:**
   - The `Bird` class represents the player’s bird, responsible for the bird's movement, image animation, and jumping functionality.
   - **Key methods** include:
     - `jump()`: Moves the bird upward.
     - `move()`: Controls the bird’s vertical movement based on gravity.
     - `draw()`: Animates the bird flapping its wings as it moves.

3. **Defining the Pipe and Base Classes:**
   - The `Pipe` class manages the position and movement of pipes. It also detects collisions between the bird and pipes.
   - The `Base` class handles the floor/base scrolling to give the effect of forward movement.

4. **Game Loop and Rendering:**
   - The `draw_window()` function is used to render the bird, pipes, base, and the score onto the screen. It also displays the number of birds alive and the generation of the simulation.

5. **Neural Network Integration:**
   - The **NEAT algorithm** is integrated through the `eval_genomes()` function, which runs the game for each bird (genome) in the population.
   - The fitness of each bird is evaluated based on how far it travels and how many pipes it passes.
   - The bird's neural network receives inputs such as the bird’s position and the pipes' positions and decides whether the bird should jump.

6. **Training with NEAT:**
   - The `run()` function executes the NEAT algorithm to train the bird over multiple generations. It utilizes a configuration file (`config-feedforward.txt`) that defines NEAT-specific parameters such as population size, mutation rates, and fitness thresholds.

7. **Game Control:**
   - In the main loop, the game updates every frame, checking for user input, updating the bird’s position, and adding/removing pipes as they move off-screen. If the bird hits a pipe or the ground, it is removed from the current generation.

---

By evolving the bird’s neural network through NEAT, the goal is to maximize its ability to survive and navigate through pipes. Over time, the bird learns from the environment and improves its gameplay autonomously.

## **Importance of `config-feedforward.txt`**

The `config-feedforward.txt` file is important in determining how the **NEAT algorithm** evolves the neural networks in the **Flappy Bird** game. It outlines the various configuration parameters that directly affect the training and optimization process for the bird’s AI. Here's why this file is essential:

---

### **Purpose of the Configuration File:**

1. **Neural Network Evolution Control:**
   - The file defines all the key parameters for the neuroevolution process. These include population size, mutation rates, and the fitness criteria used to judge the bird’s performance. Without these parameters, the NEAT algorithm wouldn't have a structured way to evolve neural networks.

2. **Customizability and Fine-Tuning:**
   - The configuration file allows developers to tweak and experiment with different settings without modifying the main Python script. This flexibility makes it easier to optimize the training process and find the best combination of parameters for successful evolution. For example, adjusting the population size or mutation rate can drastically affect the bird’s ability to learn.

3. **Separation of Logic and Configuration:**
   - By keeping these parameters in a separate file, the code becomes cleaner and more modular. The Python script remains focused on the game mechanics and the evolutionary process, while the configuration file manages all the hyperparameters related to NEAT.

4. **Key Parameters in NEAT:**
   - **Population and Fitness Settings**: Parameters such as `pop_size`, `fitness_criterion`, and `fitness_threshold` dictate how many birds are trained in each generation and what constitutes success. This is crucial for ensuring the AI improves over time.
   - **Mutation and Reproduction**: Parameters like `conn_add_prob`, `conn_delete_prob`, `node_add_prob`, and `node_delete_prob` manage how the neural networks mutate over generations, introducing the variation needed for evolution.
   - **Species Management**: The `compatibility_threshold` controls how genomes are grouped into species, ensuring genetic diversity and avoiding premature convergence.

5. **Neat Algorithm Guidance:**
   - It outlines how the neural networks should behave, including the structure of the genome (i.e., the number of inputs, outputs, and hidden layers) and mutation rates. These guide how the neural network should respond and evolve over time to become better at the game.

---

### **Impact on Training and Evolution:**


The settings in `config-feedforward.txt` are pivotal in shaping the AI’s learning curve. For instance:
- The **mutation rates** influence how much the neural network changes from one generation to the next, determining whether the changes are incremental or drastic.
- The **fitness criterion** ensures that only the best-performing birds are preserved and used to generate the next population.

By adjusting these parameters over time, the file plays a crucial role in optimizing the performance of the AI, helping the birds "learn" how to play Flappy Bird more effectively.

# Knowledge Sake

## NEAT

NeuroEvolution of Augmenting Topologies (NEAT) is an algorithm used in machine learning for evolving artificial neural networks. It was developed by **Kenneth Stanley** and **Risto Miikkulainen** in the year **2002**. NEAT is an evolutionary algorithm that evolves neural networks through mutation and selection. Unlike traditional methods, NEAT starts with simple networks and gradually adds complexity, optimizing both the network's structure and weights. It's used in projects like self-playing games to develop adaptive and efficient solutions without pre-defined architectures.

## Why Python?

Python was chosen for the main program due to its simplicity, readability, and extensive library support. Python’s rich ecosystem, including libraries like NumPy and TensorFlow, facilitates efficient development and integration of complex algorithms, such as NEAT. Additionally, Python’s strong community support and ease of debugging make it an ideal choice for rapid prototyping and experimentation.
