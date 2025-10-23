# Nithin Ravikumar
# nithinravikumar101@gmail.com
# RL Maze Solver

import time
import tkinter as tk
import numpy as np
import random

class MazeSolverGUI:
    def __init__(self, master, maze):
        self.master = master
        self.maze = maze
        self.rows, self.cols = maze.shape
        self.cell_size = 40
        self.actions = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
        self.learning_rate = 0.1
        self.discount_factor = 0.99
        self.epsilon = 0.1
        self.q_table = np.zeros((self.maze.size, 4))
        self.speed = 1000
        self.create_widgets()
        self.draw_maze()
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01

    def create_widgets(self):
        control_frame = tk.Frame(self.master)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.canvas = tk.Canvas(self.master, width=self.cols*self.cell_size, height=self.rows*self.cell_size)
        self.canvas.pack(side=tk.LEFT)

        tk.Label(control_frame, text="Controls & Stats").pack()

        self.start_button = tk.Button(control_frame, text="Start Training", command=self.start_training)
        self.start_button.pack()

        self.speed_up_button = tk.Button(control_frame, text="Speed Up", command=lambda: self.adjust_speed(-100))
        self.speed_up_button.pack()

        self.slow_down_button = tk.Button(control_frame, text="Slow Down", command=lambda: self.adjust_speed(100))
        self.slow_down_button.pack()

        self.stats_label = tk.Label(control_frame, text="Time Elapsed: 0s\nEpisode: 0\nStep: 0")
        self.stats_label.pack()

    def adjust_speed(self, adjustment):
        self.speed = max(10, self.speed + adjustment)

    def start_training(self):
        start_time = time.time()
        start_position = np.argwhere(self.maze == 2)[0]
        goal_position = np.argwhere(self.maze == 3)[0]
        success_rate_threshold = 0.95
        rolling_window_size = 20
        success_count = 0
        steps_in_recent_episodes = []
        previous_avg_steps = float('inf')

        episode = 0
        final_path = None

        while True:
            episode += 1
            state = self.position_to_state(start_position)
            done = False
            steps = 0
            current_path = [start_position.tolist()]

            while not done:
                steps += 1
                action = self.choose_action(state)
                position = np.array([state // self.cols, state % self.cols])
                move = self.actions[action]
                new_position = position + np.array(move)

                if not self.is_valid_position(new_position):
                    reward = -1
                    next_state = state
                elif np.array_equal(new_position, goal_position):
                    reward = 100
                    done = True
                    success_count += 1
                    next_state = self.position_to_state(new_position)
                    current_path.append(new_position.tolist())
                else:
                    reward = -0.1
                    next_state = self.position_to_state(new_position)
                    current_path.append(new_position.tolist())

                best_future_reward = np.max(self.q_table[next_state])
                self.q_table[state, list(self.actions.keys()).index(action)] += \
                    self.learning_rate * (reward + self.discount_factor * best_future_reward - self.q_table[
                        state, list(self.actions.keys()).index(action)])

                state = next_state
                self.update_gui(new_position)
                self.update_stats(time.time() - start_time, episode, steps)
                self.master.update()
                self.master.after(self.speed)

            steps_in_recent_episodes.append(steps)
            if len(steps_in_recent_episodes) > rolling_window_size:
                steps_in_recent_episodes.pop(0)

            if episode >= rolling_window_size:
                success_rate = success_count / rolling_window_size
                avg_steps = sum(steps_in_recent_episodes) / rolling_window_size
                if success_rate >= success_rate_threshold and previous_avg_steps >= avg_steps:
                    final_path = current_path
                    print(f"Maze solved efficiently after {episode} episodes.")
                    break
                success_count -= 1 if steps_in_recent_episodes[0] == steps_in_recent_episodes[-1] else 0
                previous_avg_steps = avg_steps

        if final_path:
            self.display_final_path(final_path)

    def display_final_path(self, path):
        self.draw_maze()
        for position in path:
            x0, y0 = position[1] * self.cell_size, position[0] * self.cell_size
            x1, y1 = x0 + self.cell_size, y0 + self.cell_size
            self.canvas.create_rectangle(x0, y0, x1, y1, fill="cyan",
                                         outline="gray")
        self.master.update()

    def update_gui(self, position):
        self.draw_maze()
        x0, y0 = position[1] * self.cell_size, position[0] * self.cell_size
        x1, y1 = x0 + self.cell_size, y0 + self.cell_size
        self.canvas.create_rectangle(x0, y0, x1, y1, fill="red", outline="gray")

    def update_stats(self, time_elapsed, episode, step):
        self.stats_label.config(text=f"Time Elapsed: {time_elapsed:.2f}s\nEpisode: {episode}\nStep: {step}")

    def draw_maze(self):
        self.canvas.delete("all")
        for i in range(self.rows):
            for j in range(self.cols):
                x0, y0 = j * self.cell_size, i * self.cell_size
                x1, y1 = x0 + self.cell_size, y0 + self.cell_size
                fill = "white"
                if self.maze[i, j] == 1:
                    fill = "black"
                elif self.maze[i, j] == 2:
                    fill = "blue"
                elif self.maze[i, j] == 3:
                    fill = "green"
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline="gray")

    def position_to_state(self, position):
        return position[0] * self.cols + position[1]

    def is_valid_position(self, position):
        if position[0] < 0 or position[0] >= self.rows or position[1] < 0 or position[1] >= self.cols:
            return False
        if self.maze[position[0], position[1]] == 1:
            return False
        return True

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(list(self.actions.keys()))
        else:
            action = list(self.actions.keys())[np.argmax(self.q_table[state])]
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        return action


# maze_layout = np.array([
#     [0, 0, 1, 1, 1],
#     [1, 0, 0, 1, 0],
#     [2, 0, 1, 0, 0],
#     [1, 0, 0, 0, 1],
#     [0, 0, 1, 0, 3]
# ])

def generate_large_maze(rows, cols):
    # seed 42 used for testing currently random seed
    np.random.seed()

    rows = max(rows, 5)
    cols = max(cols, 5)

    maze = np.zeros((rows, cols), dtype=int)

    wall_positions = np.random.choice(range(1, rows*cols-2), replace=False,
                                      size=int(rows*cols*0.25))
    maze[np.unravel_index(wall_positions, (rows, cols))] = 1

    entrance_position = (np.random.randint(0, rows), 0)
    exit_position = (np.random.randint(0, rows), cols-1)

    maze[entrance_position] = 2
    maze[exit_position] = 3

    if entrance_position[1] + 1 < cols:
        maze[entrance_position[0], entrance_position[1] + 1] = 0
    if exit_position[1] - 1 >= 0:
        maze[exit_position[0], exit_position[1] - 1] = 0

    return maze


def main():
    # best if set to 10 or less
    maze_layout = generate_large_maze(7,7)
    root = tk.Tk()
    root.title("Maze Solver")
    app = MazeSolverGUI(root, maze_layout)
    root.mainloop()


if __name__ == "__main__":
    main()