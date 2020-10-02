"""Assignment 2 of Decision Modelling
@Author: Akash Malhotra
Github repo: https://github.com/akashjorss/Decision_Modeling_Assignments
"""

from pulp import *
import pandas as pd


prob = LpProblem("The_Miracle_Worker", LpMaximize)

x = LpVariable("Medicine_1_units", 0, None, LpInteger)
y = LpVariable("Medicine_2_units", 0, None, LpInteger)
prob += 25*x + 20*y, "Health restored; to be maximized"
prob += 3*x + 4*y <= 25, "Herb A constraint"
prob += 2*x + y <= 10, "Herb B constraint"

prob.writeLP("MiracleWorker.lp")

prob.solve()

print("Status:", LpStatus[prob.status])

for v in prob.variables():
    print(v.name, "=", v.varValue)
    
print("Total Health that can be restored = ", value(prob.objective))


#Toy problem
prob = LpProblem("Maximise_the_profit", LpMaximize)
x = LpVariable("Toy_A", 0, None, LpInteger)
y = LpVariable("Toy_B", 0, None, LpInteger)
prob += 25*x + 20*y, "Profit to be maximized"
prob += 20*x +12*y <= 2000, "Total available units"
prob += 5*x + 5*y <= 540, "Total available minutes"

prob.writeLP("Profit_Maximisation.lp")
prob.solve()
print("Status:", LpStatus[prob.status])

for v in prob.variables():
    print(v.name, "=", v.varValue)
    
print("Total profit that can be made = ", value(prob.objective))


places = ['TE', 'ML', 'AT', 'MO', 'JT', 'CA', 'CP', 'CN', 'BS', 'SC', 'PC', 'TM', 'AC']
sites = LpVariable.dicts("visit", [place for place in places], cat='Binary')
sites.values()
price = [15.5, 12, 9.5, 11, 0, 10, 10, 5, 8, 8.5, 0, 15, 0]
time = [4.5, 3, 1, 2, 1.5, 2, 2.5, 2, 2, 1.5, 3/4, 2, 3/2]
appreciation = [5, 4, 3, 2, 3, 4, 1, 5, 4, 1, 3, 2, 5]
dist = pd.read_csv("test_data/walking_data.csv", header = None)
#make the matrix bidirectional
dist = dist + dist.T #make it bidirectional


#ListVisit1
 
def list_visit1(prob, disp=True):
    prob += lpSum(sites.values())
    prob += lpDot(sites.values(), price) <= 65
    prob += lpDot(sites.values(), time) <= 12

    prob.writeLP("ListVisit1.lp")
    prob.solve()
    print("Status:", LpStatus[prob.status])

    result = []
    for v in prob.variables():
        if disp == True:
            print(v.name, "=", v.varValue)
        result.append(v.varValue)
        
    print("List Visit 1 optimum = ", value(prob.objective))
    return result


prob = LpProblem("Optimise visit to Paris", LpMaximize)
list_visit1(prob)



#Preference 1: If two sites are geographically very close (within a radius of 1 km of walking), he will prefer to visit these
#two sites instead of visiting only one.

def set_pref1(prob, disp = True):  
    site_variables = list(sites.values())
    for row in range(dist.shape[0]):
        for col in range(row, dist.shape[1]):
            #print(dist.iloc[row][col], end=" ")
            if row != col and dist[row][col] <= 1:
                #add constraint to equation
                prob += site_variables[row] == site_variables[col]


    prob.writeLP("Preference_1.lp")
    prob.solve()
    print("Status:", LpStatus[prob.status])

    result = []
    for v in prob.variables():
        if disp:
            print(v.name, "=", v.varValue)
        result.append(v.varValue)
        
    print("Preference 1 optimum = ", value(prob.objective))
    return result

#Preference 2 : He absolutely wants to visit the Eiffel Tower (TE) and Catacombes (CA).
def set_pref2(prob, disp=True):
    prob += sites['TE'] == 1
    prob += sites['CA'] == 1
    prob.writeLP("List_Preference2.lp")
    prob.solve()
    print("Status:", LpStatus[prob.status])
    
    result = []
    for v in prob.variables():
        if disp:
            print(v.name, "=", v.varValue)
        result.append(v.varValue)

    print("Preference 2 optimum = ", value(prob.objective))
    return result


#Preference 3 : If he visits Notre Dame Cathedral (CN) then he will not visit the Sainte Chapelle (SC).
def set_pref3(prob, disp=True):
    prob += sites['CN'] + sites['SC'] <= 1
    prob.writeLP("List_Preference3.lp")
    prob.solve()
    print("Status:", LpStatus[prob.status])
    
    result = []
    for v in prob.variables():
        if disp:
            print(v.name, "=", v.varValue)
        result.append(v.varValue)

    print("Preference 3 optimum = ", value(prob.objective))
    return result


#Preference 4 : He absolutely wants to visit Tour Montparnasse (TM).
def set_pref4(prob, disp=True):
    prob += sites['TM'] == 1
    prob.writeLP("List_Preference4.lp")
    prob.solve()
    print("Status:", LpStatus[prob.status])
    
    result = []
    for v in prob.variables():
        if disp:
            print(v.name, "=", v.varValue)
        result.append(v.varValue)

    print("Preference 4 optimum = ", value(prob.objective))
    return result

#Preference 5 : If he visits the Louvre (ML) Museum then he must visit the Pompidou Center (CP).
def set_pref5(prob, disp=True):
    prob += (sites['ML'] - sites['CP']) <= 0
    prob.writeLP("List_Preference5.lp")
    prob.solve()
    print("Status:", LpStatus[prob.status])
    
    result = []
    for v in prob.variables():
        if disp:
            print(v.name, "=", v.varValue)
        result.append(v.varValue)

    print("Preference 5 optimum = ", value(prob.objective))
    return result

def compare_lists(A, B):
    for i in range(len(A)):
        if A[i] != B[i]:
            return False
    
    return True


print("------- Preference 1 -----")
#set up
prob = LpProblem("Optimise visit to Paris", LpMaximize)
# list_visit1
result_list_visit1 = list_visit1(prob, False)
result1 = set_pref1(prob)
print(compare_lists(result_list_visit1, result1))


print("------- Preference 2 -----")
#set up
prob = LpProblem("Optimise visit to Paris", LpMaximize)
# list_visit1
result_list_visit1 = list_visit1(prob, False)
result2 = set_pref2(prob)
print(compare_lists(result_list_visit1, result2))


#set up
prob = LpProblem("Optimise visit to Paris", LpMaximize)
# list_visit1
result_list_visit1 = list_visit1(prob, False)
print("------- Preference 3 -----")
result3 = set_pref3(prob)
print(compare_lists(result_list_visit1, result3))


print("------- Preference 4 -----")
#set up
prob = LpProblem("Optimise visit to Paris", LpMaximize)
# list_visit1
result_list_visit1 = list_visit1(prob, False)
result4 = set_pref4(prob)
print(compare_lists(result_list_visit1, result4))


print("------- Preference 5 -----")
#set up
prob = LpProblem("Optimise visit to Paris", LpMaximize)
# list_visit1
result_list_visit1 = list_visit1(prob, False)
result5 = set_pref5(prob)
print(compare_lists(result_list_visit1, result5))


#If Mr. Doe wishes, at the same time, to take into account Preference 1 and Preference 2, which list(s) would you recommend
#to him ?
prob = LpProblem("Optimise visit to Paris", LpMaximize)
result_list_visit1 = list_visit1(prob, False)
result1 = set_pref1(prob, False)
result_1_2 = set_pref2(prob)
print(compare_lists(result_list_visit1, result_1_2))

#If Mr. Doe wishes, at the same time, to take into account Preference 1 and Preference 3, which list(s) would you recommend
#to him ?
prob = LpProblem("Optimise visit to Paris", LpMaximize)
result_list_visit1 = list_visit1(prob, False)
result1 = set_pref1(prob, False)
result_1_3 = set_pref3(prob)
print(compare_lists(result_list_visit1, result_1_3))


#If Mr. Doe wishes, at the same time, to take into account Preference 1 and Preference 4, which list(s) would you recommend
#to him ?
prob = LpProblem("Optimise visit to Paris", LpMaximize)
result_list_visit1 = list_visit1(prob, False)
result1 = set_pref1(prob, False)
result_1_4 = set_pref4(prob)
print(compare_lists(result_list_visit1, result_1_4))




#If Mr. Doe wishes, at the same time, to take into account Preference 2 and Preference 5, which list(s) would you recommend
#to him ?
prob = LpProblem("Optimise visit to Paris", LpMaximize)
result_list_visit1 = list_visit1(prob, False)
result2 = set_pref2(prob, False)
result_2_5 = set_pref5(prob)
print(compare_lists(result_list_visit1, result_2_5))




#If Mr. Doe wishes, at the same time, to take into account Preference 3 and Preference 4, which list(s) would you recommend
#to him ?
prob = LpProblem("Optimise visit to Paris", LpMaximize)
result_list_visit1 = list_visit1(prob, False)
result3 = set_pref3(prob, False)
result_3_4 = set_pref4(prob)
print(compare_lists(result_list_visit1, result_3_4))


#If Mr. Doe wishes, at the same time, to take into account Preference 4 and Preference 5, which list(s) would you recommend
#to him ?
prob = LpProblem("Optimise visit to Paris", LpMaximize)
result_list_visit1 = list_visit1(prob, False)
result4 = set_pref4(prob, False)
result_4_5 = set_pref5(prob)
print(compare_lists(result_list_visit1, result_4_5))


#If Mr. Doe wishes, at the same time, to take into account Preference 1, Preference 2 and Preference 4, which list(s) would
#you recommend to him ?
prob = LpProblem("Optimise visit to Paris", LpMaximize)
result_list_visit1 = list_visit1(prob, False)
result1 = set_pref1(prob, False)
result_1_2 = set_pref2(prob, False)
result_1_2_4 = set_pref4(prob)
print(compare_lists(result_list_visit1, result_1_2_4))


#If Mr. Doe wishes, at the same time, to take into account Preference 2, Preference 3 and Preference 5, which list(s) would
#you recommend to him ?
prob = LpProblem("Optimise visit to Paris", LpMaximize)
result_list_visit1 = list_visit1(prob, False)
result2 = set_pref2(prob, False)
result_2_3 = set_pref3(prob, False)
result_2_3_5 = set_pref5(prob)
print(compare_lists(result_list_visit1, result_2_3_5))


#If Mr. Doe wishes, at the same time, to take into account Preference 2, Preference 3, Preference 4 and Preference 5, which
#list(s) would you recommend to him ?
prob = LpProblem("Optimise visit to Paris", LpMaximize)
result_list_visit1 = list_visit1(prob, False)
result2 = set_pref2(prob, False)
result_2_3 = set_pref3(prob, False)
result_2_3_4 = set_pref4(prob, False)
result_2_3_4_5 = set_pref5(prob)
print(compare_lists(result_list_visit1, result_2_3_4_5))


#If Mr. Doe wishes, at the same time, to take into account Preference 1, Preference 2, Preference 4 and Preference 5, which
#list(s) would you recommend to him ?
prob = LpProblem("Optimise visit to Paris", LpMaximize)
result_list_visit1 = list_visit1(prob, False)
result1 = set_pref1(prob, False)
result_1_2 = set_pref2(prob, False)
result_1_2_4 = set_pref4(prob, False)
result_1_2_4_5 = set_pref5(prob)
print(compare_lists(result_list_visit1, result_1_2_4_5))


#If Mr. Doe wishes, at the same time, to take into account Preference 1, Preference 2, Preference 3, Preference 4 and
#Preference 5, which list(s) would you recommend to him ?
prob = LpProblem("Optimise visit to Paris", LpMaximize)
result_list_visit1 = list_visit1(prob, False)
result1 = set_pref1(prob, False)
result_1_2 = set_pref2(prob, False)
result_1_2_3 = set_pref3(prob, False)
result_1_2_3_4 = set_pref4(prob, False)
result_1_2_3_4_5 = set_pref5(prob)
print(compare_lists(result_list_visit1, result_1_2_3_4_5))


#Is the solution ListVisit1 different to these solutions founded above (with the combination of preferences) ?
print("All the solutions with the above preferences are different to ListVisit1")

#Find rankings
import scipy.stats as stats
tau, p_value = stats.kendalltau(price, time)
print("Kendall Correlation for (price, time): ", tau)
tau, p_value = stats.kendalltau(price, appreciation)
print("Kendall Correlation for (price, ratings): ", tau)
tau, p_value = stats.kendalltau(appreciation, time)
print("Kendall Correlation for (ratings, time): ", tau)


rho, pval = stats.spearmanr(price, time)
print("Spearman Correlation for (price, time): ", rho)
rho, pval = stats.spearmanr(price, appreciation)
print("Spearman Correlation for (price, ratings): ", rho)
rho, pval = stats.spearmanr(appreciation, time)
print("Spearman Correlation for (ratings, time): ", rho)


print("The rankings are different if we sort by all 3 quantities. However, Time and Price have some significant correlation, or statistical dependence but other quantities seem independent")

