"""
Authors: Gayane Vardanyan (gayane.vardanyan@student-cs.fr) and Akash Malhotra (akash.malhotra@student-cs.fr)
For Github repo, visit: https://github.com/akashjorss/Decision_Modeling_Assignments
"""

#All imports here
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

#converting an excel to a .csv 
#def convert_excel_to_csv():
#    data = pd.read_excel("matrix.xlsx", header=0)
#    data.to_csv("test_data/matrix.csv", index = None, header=True)
#convert_excel_to_csv()

#Read the dataset
matrix = np.array(pd.read_csv("test_data/matrix.csv"))
#matrix = np.delete(matrix, 0, 1) #->?
print(matrix)


#Vizualize the matrix as a directed graph
def show_graph_with_labels(adjacency_matrix):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.DiGraph()
    gr.add_edges_from(edges)
    nx.draw(gr, node_size=1000, with_labels=True)
    plt.show()

show_graph_with_labels(matrix)


#Condition for the relation not to be complete (for further reusability)
def _not_complete(matrix, i, j): 
    return matrix[i][j] == matrix[j][i] == 0

#Check if the relation is Complete 
def complete_check(matrix):
    rows = matrix.shape[0]
    cols = matrix.shape[1]
    for i in range(rows):
        for j in range(cols):
            if _not_complete(matrix, i, j):
                return False
    return True
            
if complete_check(matrix):
    print("The relation is complete.")
else:
    print("The relation is not complete.")


#Check if the relation is reflexive
def reflexive_check(matrix):
    rows = matrix.shape[0]
    cols = matrix.shape[1]
    for i in range(rows):
            if matrix[i][i] == 0:
                return False
    return True

if reflexive_check(matrix):
    print("The relation is reflexive.")
else:
    print("The relation is not reflexive.")


#Condition for the relation not to be asymmetric (for further reusability)
def _not_asymetric(matrix, i, j): 
    return matrix[i][j] == 1 and matrix[j][i] == 1

#Check if the relation is asymmetric
def asymetric_check(matrix):
    rows = matrix.shape[0]
    cols = matrix.shape[1]
    for i in range(rows):
        for j in range(cols):
            if _not_asymetric(matrix, i, j): 
                return False
    return True

if asymetric_check(matrix):
    print("The relation is asymmetric.")
else:
    print("The relation is not asymmetric.")


#Check if the relation is symmetric
def symmetric_check(matrix):
    rows = matrix.shape[0]
    cols = matrix.shape[1]
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] != matrix[j][i]:
                return False
    return True

if symmetric_check(matrix):
    print("The relation is symmetric.")
else:
    print("The relation is not symmetric.")


#Condition for the relation not to be asymmetric (for further reusability)
def _not_antisymmetric(matrix, i, j): 
    return i!=j and matrix[i][j] == 1 and matrix[j][i] == 1

#Check if the relation is antisymmetric
def antisymmetric_check(matrix):
    rows = matrix.shape[0]
    cols = matrix.shape[1]
    for i in range(rows):
        for j in range(cols):
            if _not_antisymmetric(matrix, i, j): 
                return False
    return True

if antisymmetric_check:
    print("The relation is antisymmetric.")
else:
    print("The relation is not antisymmetric.")


#Condition for the relation not to be transitive (for further reusability)
def _not_transitive(matrix, i, j, k): 
    return matrix[i][j]==1 and matrix[j][k]==1 and matrix[i][k]==0

#Check if the relation is transitive
def transitive_check(matrix):
    rows = matrix.shape[0]
    cols = matrix.shape[1]
    for i in range(rows):
        for j in range(cols):
            for k in range(cols):
                if _not_transitive(matrix, i, j, k):
                    return False
    return True

if transitive_check:
    print("The relation is transitive.")
else:
    print("The relation is not transitive.")


#Check if the relation is negative transitive
def negative_transitive_check(matrix):
    rows = matrix.shape[0]
    cols = matrix.shape[1]
    for i in range(rows):
        for j in range(cols):
            for k in range(cols):
                if matrix[i][j]==0 and matrix[j][k]==0 and matrix[i][k]==1:
                    return False
    return True

if transitive_check:
    print("The relation is negative transitive.")
else:
    print("The relation is not negative transitive.")

# Check if the relation is a total order. 
# We don't chain the functions defined previously but use their condition as we can identify and terminate earlier in case of conditions are not met we don't 
# have to analyze the whole matrix for completeness after the first failed antisymmetricity check. 
def complete_order_check(matrix):
    rows = matrix.shape[0]
    cols = matrix.shape[1]

    for i in range(rows):
        for j in range(cols):
    # if it's not complete return false
            if _not_complete(matrix, i, j):
                return False
    # if it's not antisymmetric return false
            if _not_antisymmetric(matrix, i, j):
                    return False
    # if it's not transitive return false
            for k in range(cols):
                if _not_transitive(matrix, i, j, k):
                    return False
                
    return True

if transitive_check:
    print("The relation is a total order.")
else:
    print("The relation is not a total order.")


# Check if the relation is a complete preorder. 
# We don't chain the functions defined previously but use their condition as we can identify and terminate earlier in case of conditions are not met we don't 
# have to analyze the whole matrix for completeness after the first failed transitivity check.
def complete_preorder_check(matrix):
    rows = matrix.shape[0]
    cols = matrix.shape[1]
    for i in range(rows):
        for j in range(cols):
    # if it's not complete return false
            if _not_complete(matrix, i, j):
                return False
    # if it's not transitive return false
            for k in range(cols):
                if _not_transitive(matrix, i, j, k):
                    return False
                
    return True

if transitive_check:
    print("The relation is a complete preorder.")
else:
    print("The relation is not a compete preorder.")


#Return the strincness relation over the given relation
def strictness_relationship(matrix):
    rows = matrix.shape[0]
    cols = matrix.shape[1]
    outputmatrix = np.zeros((rows, cols), dtype=int)
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == 1 and matrix[j][i] == 0:
                 outputmatrix[i][j] = matrix[i][j]
    return outputmatrix
print("The strictness relationship over the given relation is: ")
print(strictness_relationship(matrix))


#Return the indifference relation over the given relation
def indifference_relationship(matrix):
    rows = matrix.shape[0]
    cols = matrix.shape[1]
    outputmatrix = np.zeros((rows, cols), dtype=int)
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == 1 and matrix[j][i] == 1:
                 outputmatrix[i][j] = matrix[i][j]
    return outputmatrix

print("The indifference relationship over the given relation is: ")
print(indifference_relationship(matrix))


#Return a topological sorting of the relation
def topological_sorting(matrix):
    rows = matrix.shape[0]
    cols = matrix.shape[1]
    #find the number of in-degrees for each vertex (the sum of elements per column)
    in_degrees = [0] * cols
    for i in range(rows):
        for j in range(cols):
            in_degrees[j]+=matrix[i][j]
    #number of already visited nodes
    visited_nodes=0
    #queue of the visited nodes
    queue = []
    #queue for the output nodes
    output_queue = []
    #add the nodes with in-degree 0 to the queue
    for i in range(cols):
        if in_degrees[i]==0:
            queue.append(i)
    #for each element visited Increment count of visited nodes by 1. Then decrease in-degree by 1 for all its neighboring nodes.
    while queue!=[]:
        element_being_visited=queue.pop(0)
        output_queue.append(element_being_visited)
        visited_nodes+=1
        for j in range(cols):
            if matrix[element_being_visited][j]==1:
                in_degrees[j]-=1
    # If the in-degree of a neighboring nodes is reduced to zero, then add it to the queue
                if in_degrees[j]==0:
                    queue.append(j)
    #If count of visited nodes is not equal to the number of nodes in the graph then the topological sort is not possible 
    #for the given graph.
    if visited_nodes!=cols:
        print("Topological ordering isn't possible for this graph")
    else: 
        print("The topological sorting for the following relation would be: ")
        print(output_queue)

#Following should not be possible to do topological sorting
topological_sorting(matrix)
DAG = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0] ])
show_graph_with_labels(DAG)
#Following should be possible to do topological sorting
topological_sorting(DAG)
