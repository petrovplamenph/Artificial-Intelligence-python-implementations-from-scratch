"""
Find path between 2 points, and plot the path.
The barriers(#) positions are randomly chosen.
"""
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import math
import re


def main():
    board_width = 90
    board_hight = 40
    
    end_row_cord = 35
    end_col_cord = 2

    start_col = 80
    start_row = 2
    # random_bariers is the number of barriers to be placed at random positions
    random_bariers = 40
    # for user input uncomment lines 26 and 44
    """
    board_width = input("Enter board width: ")
    board_width = int(re.findall('[0-9]+',board_width)[0])
    board_hight = input("Enter board hight: ")
    board_hight = int(re.findall('[0-9]+',board_hight)[0])
    
    end_row_cord = input("Enter row of the cell to reach: ")
    end_row_cord = int(re.findall('[0-9]+',end_row_cord)[0])
    end_col_cord = input("Enter column of the the cell to reach: ")
    end_col_cord = int(re.findall('[0-9]+',end_col_cord)[0])
    
    start_row = input("Enter row of the cell from which the serach begings: ")
    start_row = int(re.findall('[0-9]+',start_row)[0])
    start_col = input("Enter column of the cell from which the serach begings: ")
    start_col = int(re.findall('[0-9]+',start_col)[0])
    
    random_bariers = input("Enter number of random barriers: ")
    random_bariers = int(re.findall('[0-9]+',random_bariers)[0])
    """
    def compute_dist(x,y):
        """
        Array  to store the manhattan distance
        """
        return abs(x - end_row_cord)+ abs(y - end_col_cord)
    #initialize board with the manhattan distances
    man_dist_matrix = np.fromfunction(compute_dist,(board_hight, board_width), dtype=int)
    man_dist_matrix *= 10
    # Create padding on the array ,in orrder to simplify the function - update
    man_dist_matrix = np.lib.pad(man_dist_matrix, pad_width=1, mode='constant')
    dist_matrix = np.zeros((board_hight, board_width ), dtype=int)
    eq_dist_matrix = np.zeros((board_hight + 2 , board_width + 2), dtype=int)


    #create an array with bariers at randm places
    visited=np.zeros(board_hight*board_width, dtype=bool)
    
    visited[:random_bariers] = True
    np.random.shuffle(visited)
    visited=visited.reshape(board_hight, board_width)
    visited[start_row, start_col] = True
    visited = np.lib.pad(visited, pad_width=1, mode='constant', constant_values=1)
    dist_matrix[end_row_cord, end_col_cord] = man_dist_matrix[end_row_cord, end_col_cord]
    visited_invert = np.invert(np.copy(visited))
    current_point = (start_row+1, start_col+1)
    # store the front in a dictinary with key-the coordinates of the points 
    #and value = Manhattan distance+ Euclidean distance
    front = {}
    def update(current_point):
        """
        Updates the front and visited cells
        """
        row, col=current_point
        point_value = eq_dist_matrix[row, col]
        if (not visited[row - 1, col] and (row - 1, col) not in front):
            eq_dist_matrix[row - 1, col] = (10 + point_value)
            front[(row - 1, col)] = eq_dist_matrix[row - 1, col]+man_dist_matrix[row - 1, col]
        elif (row - 1, col) in front:
            if  eq_dist_matrix[row - 1, col] > 10 + point_value:
                eq_dist_matrix[row - 1, col] = 10 + point_value
                front[(row - 1, col)] = eq_dist_matrix[row - 1, col] + man_dist_matrix[row - 1, col]

                
        if (not visited[row + 1, col] and (row + 1, col) not in front):
            eq_dist_matrix[row + 1, col] = (10 + point_value)
            front[(row + 1, col)] = eq_dist_matrix[row + 1, col] + man_dist_matrix[row + 1, col]
        elif (row + 1, col) in front:
            if  eq_dist_matrix[row + 1, col] > 10 + point_value:
                eq_dist_matrix[row + 1, col] = 10 + point_value
                front[(row + 1, col)] = eq_dist_matrix[row + 1, col] + man_dist_matrix[row + 1, col]
                
        if (not visited[row, col - 1] and (row, col - 1) not in front):
            eq_dist_matrix[row, col - 1] = (10 + point_value )
            front[(row, col - 1)] = eq_dist_matrix[row, col - 1] + man_dist_matrix[row, col - 1]
        elif (row, col-1) in front:
            if  eq_dist_matrix[row, col-1] > 10 + point_value:
                eq_dist_matrix[row, col-1] = 10 + point_value
                front[(row, col-1)] = eq_dist_matrix[row, col-1] + man_dist_matrix[row, col-1]
                
        if (not visited[row, col + 1] and (row, col + 1)  not in front):
            eq_dist_matrix[row, col + 1] = (10 + point_value)
            front[(row, col + 1)] = eq_dist_matrix[row, col + 1] + man_dist_matrix[row, col + 1]
        elif (row, col+1) in front:
            if  eq_dist_matrix[row, col+1] > 10 + point_value:
                eq_dist_matrix[row, col+1] = 10 + point_value
                front[(row, col+1)] = eq_dist_matrix[row, col+1] + man_dist_matrix[row, col+1]
        if (not visited[row - 1, col - 1] and (row - 1, col - 1)  not in front):
            eq_dist_matrix[row - 1, col - 1] =(14 + point_value)
            front[(row - 1, col - 1)] = eq_dist_matrix[row - 1, col - 1] + man_dist_matrix[row - 1, col - 1]
        elif (row - 1, col - 1) in front:
            if  eq_dist_matrix[row - 1, col - 1] > 14 + point_value:
                eq_dist_matrix[row - 1, col - 1] = 14 + point_value
                front[(row - 1, col - 1)] = eq_dist_matrix[row - 1, col - 1] + man_dist_matrix[row - 1, col - 1]

        if (not visited[row - 1, col + 1] and (row - 1, col + 1) not in front):
            eq_dist_matrix[row - 1, col + 1] = (14 + point_value )
            front[(row - 1, col + 1)] = eq_dist_matrix[row - 1, col + 1] + man_dist_matrix[row - 1, col + 1]
        elif (row - 1, col + 1) in front:
            if  eq_dist_matrix[row - 1, col + 1] > 14 + point_value:
                eq_dist_matrix[row - 1, col + 1] = 14 + point_value
                front[(row - 1, col + 1)] = eq_dist_matrix[row - 1, col + 1] + man_dist_matrix[row - 1, col + 1]

        if (not visited[row + 1, col - 1] and (row + 1, col - 1) not in front):
            eq_dist_matrix[row + 1, col - 1] = (14 + point_value )
            front[(row + 1, col - 1)] = eq_dist_matrix[row + 1, col - 1] + man_dist_matrix[row + 1, col - 1]
        elif (row + 1, col - 1) in front:
            if  eq_dist_matrix[row + 1, col - 1] > 14 + point_value:
                eq_dist_matrix[row + 1, col - 1] = 14 + point_value
                front[(row + 1, col - 1)] = eq_dist_matrix[row + 1, col - 1] + man_dist_matrix[row + 1, col - 1]

        if (not visited[row + 1, col + 1] and (row + 1, col + 1) not in front):
            eq_dist_matrix[row + 1, col + 1] = (14 + point_value)
            front[(row + 1, col + 1)] = eq_dist_matrix[row + 1, col + 1] + man_dist_matrix[row + 1, col + 1]
        elif (row + 1, col + 1) in front:
                if  eq_dist_matrix[row + 1, col + 1] > 14 + point_value:
                    eq_dist_matrix[row + 1, col + 1] = 14 + point_value
                    front[(row + 1, col + 1)] = eq_dist_matrix[row + 1, col + 1] + man_dist_matrix[row + 1, col + 1]

    update(current_point)
    end_point = (end_row_cord+1, end_col_cord+1)
    while (current_point != end_point):
        #untill the point of interest is reached call the update function
        if not front:
            print('There is no way')
            return ("There is no way")
        # get point from the front with minimum value
        minValKey = min(front, key=front.get)
        current_point = minValKey
        update(current_point)
        visited[current_point[0],  current_point[1]]=True
        #remove the visited point from the front
        del front[current_point]
    visited = np.multiply(visited_invert, visited)
    visited[start_row+1, start_col+1]=True
    start_point = (start_row+1, start_col+1)
    # create array to store the way between the 2 point
    way = np.zeros((board_hight, board_width ), dtype=int)
    way[start_row, start_col]=2
    way[end_row_cord, end_col_cord]=2
    #By steping only on visited cells backtrack the fastaets found way, based on the calculated distances
    while (current_point != start_point):
        row, col=current_point
        visited[row, col]=False
        next_val = math.inf
        
        if (visited[row - 1, col]):
            next_val = eq_dist_matrix[row - 1, col]
            next_point = (row - 1, col)
            
        if (visited[row + 1, col]):
            if eq_dist_matrix[row + 1, col] < next_val:
                next_val = eq_dist_matrix[row + 1, col]
                next_point = (row + 1, col)
                
        if (visited[row, col - 1]):
            if eq_dist_matrix[row, col - 1] < next_val:
                next_val = eq_dist_matrix[row, col - 1]
                next_point = (row, col - 1)
                
        if (visited[row, col + 1]):
            if eq_dist_matrix[row, col + 1] < next_val:
                next_val = eq_dist_matrix[row, col + 1]
                next_point = (row, col + 1)
                
        if (visited[row - 1, col - 1]):
            if eq_dist_matrix[row - 1, col - 1] < next_val:
                next_val = eq_dist_matrix[row - 1, col - 1]
                next_point = (row - 1, col - 1)
                
        if (visited[row - 1, col + 1]):
            if eq_dist_matrix[row - 1, col + 1] < next_val:
                next_val = eq_dist_matrix[row - 1, col + 1]
                next_point = (row - 1, col + 1)
                
        if (visited[row + 1, col - 1]):
            if eq_dist_matrix[row + 1, col - 1] < next_val:
                next_val = eq_dist_matrix[row + 1, col - 1]
                next_point=(row + 1, col - 1)
                
        if (visited[row + 1, col + 1]):
            if eq_dist_matrix[row + 1, col + 1] < next_val:
                next_val = eq_dist_matrix[row + 1, col + 1]
                next_point = (row + 1, col + 1)
        way[next_point[0]-1, next_point[1]-1] = 2
        current_point = next_point
        
    #use the inverted initial visited array where only cells which were barrieres have False valuses
    obst = visited_invert[1:-1,1:-1].astype(int)
    obst[start_row, start_col] = 0
    obst[end_row_cord, end_col_cord]=0
    #  sum the array with the path and the one with barriers and plot the summed array`
    img = obst+way
    values = [0,1,2,3]
    labels = ['barrier','empty cell','end point','path']
    im = plt.matshow(img,cmap=plt.cm.gray, interpolation='nearest')
    colors = [ im.cmap(im.norm(value)) for value in values]
    patches = [ mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels)) ]
    plt.legend(handles=patches, bbox_to_anchor=(1, 1), loc=2, borderaxespad=0. )
    plt.show()

    
if __name__ == '__main__':
    main()
