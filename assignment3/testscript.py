import numpy as np
from functools import cache
from queue import Queue

def setup():
    big_matrix = np.arange(36).reshape((6,6))
    small_matrix_size = int(np.sqrt(np.size(big_matrix)/4))
    
    small_matrix = np.arange(small_matrix_size**2).reshape((small_matrix_size,small_matrix_size))
    return big_matrix
    



def recursive_search(big_matrix):
    
    n = big_matrix.shape[0]
    m = big_matrix.shape[1]
    
    ToFind = 4
    StartIndex  = 26
    checked = {StartIndex}
    CheckQueue = Queue()
    steps = [-1,1,-n,n]
    NextCheck = 0
    for i in steps:
        CheckQueue.put(i)
    
    
    NotAllChecked = True
    
    print(big_matrix)
    while NotAllChecked:

        NextToCheck = CheckQueue.get()
        if NextToCheck == ToFind:
            return [NextToCheck.x,NextToCheck.y]
        if NextCheck not in checked:
            for i in steps:
                CheckQueue.put(NextToCheck+i)
                
        


    
recursive_search(setup())
        index_start = Node(0,0)
    relative_subpixel_position = Node(0, 0)
    # TODO: Project the point as per the exercise description.
    ToFind = Node(px_coordinate_in_down[0],px_coordinate_in_down[1])
    # StartIndex  = 26
    checked = [index_start]
    CheckQueue = Queue()
    steps = [Node(-1,0),Node(1,0),Node(0,-1),Node(0,1)]
    NextCheck = 0
    for i in steps:
        CheckQueue.put(i)
    NotAllChecked = True

    while NotAllChecked:

        NextToCheck = CheckQueue.get()
        if NextToCheck == ToFind:
            return [NextToCheck.x,NextToCheck.y]
        if NextToCheck not in checked:
            for i in steps:
                CheckQueue.put(i + NextToCheck)
                