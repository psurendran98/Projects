import numpy as np
import pandas as pd
import sys
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('expand_frame_repr', False)
np.set_printoptions(linewidth=1000)

####################################################################################################################

#Please provide the Input in Standard Form
A = np.array([[1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
              [0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,0],
              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,1],
              [1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
              [0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
              [0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
              [0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
              [0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0],
              [0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0]])
b = np.array([110,75,95,125,60,140,200,90,40,45,110,70])
c = np.array([200,300,100,600,250,500,
              400,350,150,650,300,400,
              300,250,150,600,150,350,
              500,400,200,700,450,450,
              100,200,300,650,200,300,
              600,450,250,550,150,500,
              0,0,0,0,0,0])

#Mention True if the given problem is a Minimization Problem, else False
Minimization = True

####################################################################################################################


class SimplexAlgorithm:
    
    #Changing to a Min problem
    if Minimization==False:
        c=-c
        
    #Changing array elements into float for decimals values
    A=A.astype(np.float64)
    b=b.astype(np.float64)
    c=c.astype(np.float64)
    
    #Initializing m, n
    m=len(A)
    n=len(A[0])+m
    total_rows = m+1
    total_columns = n+1
    y=len(c)
    
    #Checking the condition for infeasibility
    if np.any(b<0):
        print("The given problem is infeasible (b<0)")
        sys.exit()
        
    #Creating Basis (Identity Matrix) and Adding Artificial Variables to A
    basis=np.identity(m)
    A=np.concatenate((A, basis.T), axis=1)
    
    #Initializing the variables in the basis
    basis_variables=[]
    basis_variables_copy=[]
    for i in range (0,m):
        basis_variables.append(i+n-m+1)
        basis_variables_copy.append(i+n-m+1)
    
    #Calculating the Cost
    current_cost=0
    for i in range(0,m):
        current_cost=current_cost-b[i]
    
    #Calculating the Reduced Cost
    reduced_cost=[]
    temp=0
    for i in range(0,n-m):
        for j in range(0,m):
            temp=temp-A[j][i]   
        reduced_cost.append(temp)
        temp=0
    
    #Adding zeroes to Cost vector since we added Artificial Variables
    c=np.ndarray.tolist(c)    
    for i in range(0,m):
        reduced_cost.append(0)
        c.append(0)    
    reduced_cost=np.asarray(reduced_cost)
         
    #Building the Tableau...
    #Creating Column names 
    column_name = []
    for i in range (0, n+1):
        if i==0:
            column_name.append('cost')
        else:
            column_name.append(i)
    #Row names        
    row_name = []
    for i in range (0, m+1):
        if i==0:
            row_name.append('cost')
        else:
            row_name.append(basis_variables[i-1])
    
    #Generating the Tableau
    tableau=np.insert(reduced_cost, 0, current_cost)
    for i in range(0,m):
        rows=np.insert(A[i], 0, b[i])
        tableau = np.hstack([tableau, rows])
    tableau= np.reshape(tableau, (-1, total_columns))
    
    #Creating a Tableau Dataframe
    tableau_df = pd.DataFrame(tableau, columns=column_name, index=row_name)
    
    #SIMPLEX ALGORITHM BEGINS
    pivot_index=[0,0]
    flag=np.any(reduced_cost<0)
    iteration=1
    print("PHASE I BEGINS \n")
    
    while(flag):
        print("Iteration", iteration)
        iteration=iteration+1
        print(np.around(tableau_df,2), "\n")
        ratio=[]
        lexicographic_rows=[]
        tableau[abs(tableau)<1e-10]=0.0
        for i in range(1, total_columns):
            if(tableau[0][i] <0):    #Bland's Rule
                pivot_index[0]=i
                for j in range(1, total_rows):
                    if tableau[j][i]<=0:
                        ratio.append(sys.maxsize)
                    else:
                        ratio.append(tableau[j][0]/tableau[j][i])            
                pivot_index[1]=np.argmin(ratio)+1
                for i in range(0, len(ratio)):
                    if ratio[pivot_index[1]-1] == ratio[i]:
                        lexicographic_rows.append(i)
                break
        
        #Lexicographic Pivoting rule
        if len(lexicographic_rows)>1:
            pivot_index[1]=lexicographic_rows[0]+1
            for i in range(0, len(lexicographic_rows)-1):
                for j in range(0, n):
                    row1_value=tableau[pivot_index[1]][j]/tableau[pivot_index[1]][pivot_index[0]]
                    row2_value=tableau[lexicographic_rows[i+1]+1][j]/tableau[lexicographic_rows[i+1]+1][pivot_index[0]]
                    if row1_value <0 or row2_value <0:
                        continue
                    if row1_value<row2_value:
                        pivot_index[1]=pivot_index[1]
                        break
                    elif row1_value>row2_value:
                        pivot_index[1]=lexicographic_rows[i+1]+1
                        break
            
        #Getting the column values of the variable exiting
        ref=basis_variables[pivot_index[1]-1]
        pivot_out_column = []
        for i in range(0, total_rows):
            pivot_out_column.append(tableau[i][ref])
            
        #Getting the column values of the variable entering
        ref=pivot_index[0]
        pivot_in_column = []
        count=0
        for i in range(0, total_rows):
            pivot_in_column.append(tableau[i][ref])
            if tableau[i][ref]<0:    #Condition for Unbounded LP
                count=count+1
        #When all the values are negative in pivot column
        if count == total_rows:
            print("Algorithm stopped...")
            print("The given LP is  Unbounded \n")
            print("The optimal cost is -Infinity")
            sys.exit()
        
        #Updating Basis variables
        row_name[pivot_index[1]] = pivot_index[0]
        tableau_df = pd.DataFrame(tableau, columns=column_name, index=row_name)      
        
        #Performing Row Operations (page 97)
        val=tableau[pivot_index[1]][pivot_index[0]]
        for i in range(0, total_rows):
            for j in range (0, total_columns):
                if pivot_out_column[i] == 0:
                    tableau[i][j] = tableau[i,j] - (pivot_in_column[i]/tableau[pivot_index[1]][pivot_index[0]])*tableau[pivot_index[1]][j]
                if i==pivot_index[1]:
                    tableau[i][j] = tableau[i,j]/val
        
        print("x" + str(basis_variables[pivot_index[1]-1]) + " leaves and x" + str(pivot_index[0]) + " enters \n")
        
        #Updating the Basis Variables
        basis_variables[pivot_index[1]-1] = pivot_index[0]
    
        #Updating Reduced Cost to check for loop termination condition
        for i in range (0, n):
            reduced_cost[i]=tableau[0][i+1]
        
        #Updating condition for while loop
        flag=np.any(reduced_cost<0)
        #End of while loop
        
    print("END OF PHASE I")
    print("Final resulting tableau is ")
    print(np.around(tableau_df,2), "\n") 
    
    #Finding out if there are any artificial variables still in the basis
    artificial_variable=[]
    for i in range (0,m):
        for j in range (0,m):
            if basis_variables[i]==basis_variables_copy[j]:
                artificial_variable.append(i)
                
    #Driving out the artificial variables (All conditions checked)
    for i in range (0, len(artificial_variable)):
        pivot_index[1]=artificial_variable[i]+1
        if tableau[0][0] == 0:
            print("The given LP is feasible with cost in the auxiliary problem as zero")
            if np.all(tableau[artificial_variable[i]+1][1:y]==0):
                print("Constraint", artificial_variable[i]+1, "is redundant\n")
                print("Removing constraint...")
                
                #Updating necessary data
                tableau = np.delete(tableau, artificial_variable[i]+1 , axis=0)
                basis_variables = np.delete(basis_variables, artificial_variable[i] , axis=0)
                row_name.pop(artificial_variable[i]+1)
                tableau_df = pd.DataFrame(tableau, columns=column_name, index=row_name)         
                m=m-1
                print(np.around(tableau_df,2), "\n") 
            
            else:
                print("Driving out Artificial variable...")
                for j in range(1, total_columns):
                    pivot_index[1]=artificial_variable[i]+1
                    tableau[artificial_variable[i]+1][0]=0
                    if(tableau[0][j]!=0):    
                        pivot_index[0]=j
                        break
                        
                #Getting the column values of the variable exiting
                ref=basis_variables[pivot_index[1]-1]
                pivot_out_column = []
                for k in range(0, total_rows):
                    pivot_out_column.append(tableau[k][ref])
                    
                #Getting the column values of the variable entering
                ref=pivot_index[0]
                pivot_in_column = []
                count=0
                for k in range(0, total_rows):
                    pivot_in_column.append(tableau[k][ref])
                
                #Updating Basis variables
                row_name[pivot_index[1]] = pivot_index[0]
                tableau_df = pd.DataFrame(tableau, columns=column_name, index=row_name)      
                
                #Performing Row Operations (page 97)
                val=tableau[pivot_index[1]][pivot_index[0]]
                for i in range(0, total_rows):
                    for j in range (0, total_columns):
                        if pivot_out_column[i] == 0:
                            tableau[i][j] = tableau[i,j] - (pivot_in_column[i]/tableau[pivot_index[1]][pivot_index[0]])*tableau[pivot_index[1]][j]
                        if i==pivot_index[1]:
                            tableau[i][j] = tableau[i,j]/val
                
                print("x" + str(basis_variables[pivot_index[1]-1]) + " leaves and x" + str(pivot_index[0]) + " enters \n")
                
                #Updating the Basis Variables
                basis_variables[pivot_index[1]-1] = pivot_index[0]
        
        else:
            print("The given LP is infeasible since the cost in the auxiliary problem is non-zero")
            sys.exit()
        
     
    print("The initial basis for PHASE II is:")
    print(row_name[1:], "\n") 
        
    print("PHASE II BEGINS")
        
    #Getting the values of x_b
    x_b=[0.0]*m
    for i in range (0,m):
        x_b[i]= tableau[i+1][0]
      
    #Deleting the Artificial Variables            
    for i in range(0,m):
        tableau = np.delete(tableau, n-i, axis=1)
    tableau_df = pd.DataFrame(tableau, columns=column_name[0:n-m+1], index=row_name)
    
    #Creating the new Basis for Phase 2
    B=np.empty((m, m), float)
    for i in range (0,m):
        for j in range(0,m):
            B[j][i]=A[j][basis_variables[i]-1]
    
    #Finding inverse of B
    B_inverse=np.linalg.inv(B)
    
    #Getting the coefficients of cost vector of all basic variables
    c_b=[0.0]*m
    for i in range(0,m):
        c_b[i] = c[basis_variables[i]-1]
        
    #Initializing the Updated Cost
    c_b=np.asarray(c_b)
    tableau[0][0]= -np.matmul(c_b.transpose(),x_b)
    
    #Getting the coefficients of cost vector of all non-basic variables and generating Aj vector
    c_j=[]
    non_basis_variables=[]
    temp=0
    Aj=np.empty((m, n-2*m), float)
    for i in range(1,n-m+1):
        if i not in basis_variables:
            c_j.append(c[i-1])
            non_basis_variables.append(i)
            for j in range(0,m):
                Aj[j][temp]=A[j][i-1]
            temp=temp+1
            
    #Computing the new Reduced cost 
    c_j_bar= c_j-np.matmul(np.matmul(c_b.transpose(),B_inverse),Aj)
    for i in range(0, len(non_basis_variables)):
        tableau[0][non_basis_variables[i]]=c_j_bar[i]
     
    #Updating while loop condition
    reduced_cost=tableau[0][1:]
    flag=np.any(reduced_cost<0)
    
    #Updating new m, n
    m=len(tableau)-1
    n=len(tableau[0])-1
    total_rows = m+1
    total_columns = n+1
    column_name=column_name[0:n+1]
    
    iteration=1
    while(flag):
        print("Iteration", iteration)
        iteration=iteration+1
        print(np.around(tableau_df,2), "\n")
        ratio=[]
        lexicographic_rows=[]
        tableau[abs(tableau)<1e-10]=0.0
        for i in range(1, total_columns):
            if(tableau[0][i] <0):    #Bland's Rule
                pivot_index[0]=i
                for j in range(1, total_rows):
                    if tableau[j][i]<=0:
                        ratio.append(sys.maxsize)
                    else:
                        ratio.append(tableau[j][0]/tableau[j][i])            
                pivot_index[1]=np.argmin(ratio)+1
                for i in range(0, len(ratio)):
                    if ratio[pivot_index[1]-1] == ratio[i]:
                        lexicographic_rows.append(i)
                break
        
        #Lexicographic Pivoting rule
        if len(lexicographic_rows)>1:
            pivot_index[1]=lexicographic_rows[0]+1
            for i in range(0, len(lexicographic_rows)-1):
                for j in range(0, n):
                    row1_value=tableau[pivot_index[1]][j]/tableau[pivot_index[1]][pivot_index[0]]
                    row2_value=tableau[lexicographic_rows[i+1]+1][j]/tableau[lexicographic_rows[i+1]+1][pivot_index[0]]
                    if row1_value <0 or row2_value <0:
                        continue
                    if row1_value<row2_value:
                        pivot_index[1]=pivot_index[1]
                        break
                    elif row1_value>row2_value:
                        pivot_index[1]=lexicographic_rows[i+1]+1
                        break
            
        #Getting the column values of the variable exiting
        ref=basis_variables[pivot_index[1]-1]
        pivot_out_column = []
        for i in range(0, total_rows):
            pivot_out_column.append(tableau[i][ref])
            
        #Getting the column values of the variable entering
        ref=pivot_index[0]
        pivot_in_column = []
        count=0
        for i in range(0, total_rows):
            pivot_in_column.append(tableau[i][ref])
            if tableau[i][ref]<0:    #Condition for Unbounded LP
                count=count+1
        #When all the values are negative in pivot column
        if count == total_rows:
            print("Algorithm stopped...")
            print("The given LP is  Unbounded \n")
            print("The optimal cost is -Infinity")
            sys.exit()
        
        #Updating Basis variables
        row_name[pivot_index[1]] = pivot_index[0]
        tableau_df = pd.DataFrame(tableau, columns=column_name, index=row_name)      
        
        #Performing Row Operations (page 97)
        val=tableau[pivot_index[1]][pivot_index[0]]
        for i in range(0, total_rows):
            for j in range (0, total_columns):
                if pivot_out_column[i] == 0:
                    tableau[i][j] = tableau[i,j] - (pivot_in_column[i]/tableau[pivot_index[1]][pivot_index[0]])*tableau[pivot_index[1]][j]
                if i==pivot_index[1]:
                    tableau[i][j] = tableau[i,j]/val
        
        print("x" + str(basis_variables[pivot_index[1]-1]) + " leaves and x" + str(pivot_index[0]) + " enters \n")
        
        #Updating the Basis Variables
        basis_variables[pivot_index[1]-1] = pivot_index[0]
    
        #Updating Reduced Cost to check for loop termination condition
        for i in range (0, n):
            reduced_cost[i]=tableau[0][i+1]
        
        #Updating condition for while loop
        flag=np.any(reduced_cost<0)
        #End of while loop
    
    print("END OF PHASE II")
    print("Final resulting tableau is")    
    print(np.around(tableau_df,2))
    print()
    cost=tableau[0][0]
    if Minimization==False:
        cost=-cost
    print("The optimal cost is", -np.around(cost,2))
    x=[0.0]*n
    for i in range (1,m+1):
        x[row_name[i]-1]= tableau[i][0]
    print("The BFS is : x* =" , np.around(x,2))