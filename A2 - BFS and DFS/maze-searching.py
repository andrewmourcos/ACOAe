# Maze class to help with printing functions
class Maze():
    def __init__(self, maze):
        self.maze = maze # 2D Array
        self.colored = []
    
    def print(self):
        for i, row in enumerate(reversed(self.maze)):
            for j, col in enumerate(row):
                if [j, len(self.maze)-1-i] in self.colored:
                    print(". ", end="")
                else:
                    print(col, end=" ")
            print("")

    def color(self, path):
        self.colored = path


"""
    Uninformed grid-search algorithm. Given maze (2d array), traverses open cells (marked with 0) in
    either breadth-first or depth-first fashion until a goal node is found.
"""
def uninformed_search(maze, start, goals, algorithm="bfs", actions=[[0,1],[1,0],[0,-1],[-1,0]]):
    numrows = len(maze)
    numcols = len(maze[0])

    if algorithm != "bfs" and algorithm != "dfs":
        print("Invalid input, please select either bfs or dfs as the search mode")
        return
    
    if algorithm == "dfs":
        actions.reverse()

    visited = {(start[0], start[1])}
    open_deque = [[start]]

    num_explored_nodes=0
    while(open_deque):
        if algorithm == "bfs":
            # Use open list as a FIFO queue in BFS
            path = open_deque.pop(0)
        if algorithm == "dfs":
            # Use open list as a LIFO stack in DFS
            path = open_deque.pop(-1)

        num_explored_nodes+=1
        curr_node = [col, row] = path[-1]
        if curr_node in goals:
            print("Found exit using ", algorithm, " at ", curr_node, " after exploring ", num_explored_nodes, " nodes.")
            print("Path: ", path)
            print("Cost: ", len(path))
            return path

        if algorithm == "dfs":
            visited.add((col,row))
        
        # Add all valid (unvisited, unblocked) children nodes to deque
        for dx, dy in actions:
            r, c = row + dy, col + dx

            if (r in range(numrows) and
                c in range(numcols) and
                maze[r][c] == 0 and
                (c,r) not in visited):

                open_deque.append( path + [[c,r]] )
                if algorithm == "bfs":
                    visited.add((c,r))

    print("Did not find the goal node")
    return []

maze = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]]



# Initialize given maze
print("Input maze:")
m=Maze(maze)
m.print()

print("\n***\nExperiment 1: BFS with LEFT-UP-RIGHT-DOWN tie-breaking")
p = uninformed_search(maze=maze, start=[2,11], goals=[[23,19], [2,21]], 
                        algorithm="bfs", actions=[[-1,0],[0,1],[1,0],[0,-1]])
m.color(p)
m.print()

print("\n***\nExperiment 2: BFS with RIGHT-UP-LEFT-DOWN tie-breaking")
p = uninformed_search(maze=maze, start=[2,11], goals=[[23,19], [2,21]],
                        algorithm="bfs", actions=[[1,0],[0,1],[-1,0],[0,-1]])
m.color(p)
m.print()

print("\n***\nExperiment 3: DFS with LEFT-UP-RIGHT-DOWN tie-breaking")
p = uninformed_search(maze=maze, start=[2,11], goals=[[23,19], [2,21]], 
                        algorithm="dfs", actions=[[-1,0],[0,1],[1,0],[0,-1]])
m.color(p)
m.print()

print("\n***\nExperiment 4: DFS with RIGHT-UP-LEFT-DOWN tie-breaking")
p = uninformed_search(maze=maze, start=[2,11], goals=[[23,19], [2,21]],
                        algorithm="dfs", actions=[[1,0],[0,1],[-1,0],[0,-1]])
m.color(p)
m.print()

