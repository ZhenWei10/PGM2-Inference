%BLOCKLOGDISTRIBUTION
%
%   LogBS = BlockLogDistribution(V, G, F, A) returns the log of a
%   block-sampling array (which contains the log-unnormalized-probabilities of
%   selecting each label for the block), given variables V to block-sample in
%   network G with factors F and current assignment A.  Note that the variables
%   in V must all have the same dimensionality.
%
%   Input arguments:
%   V -- an array of variable names
%   G -- the graph with the following fields:
%     .names - a cell array where names{i} = name of variable i in the graph 
%     .card - an array where card(i) is the cardinality of variable i
%     .edges - a matrix such that edges(i,j) shows if variables i and j 
%              have an edge between them (1 if so, 0 otherwise)
%     .var2factors - a cell array where var2factors{i} gives an array where the
%              entries are the indices of the factors including variable i
%   F -- a struct array of factors.  A factor has the following fields:
%       F(i).var - names of the variables in factor i
%       F(i).card - cardinalities of the variables in factor i
%       F(i).val - a vectorized version of the CPD for factor i (raw probability)
%   A -- an array with 1 entry for each variable in G s.t. A(i) is the current
%       assignment to variable i in G.
%
%   Each entry in LogBS is the log-probability that that value is selected.
%   LogBS is the P(V | X_{-v} = A_{-v}, all X_i in V have the same value), where
%   X_{-v} is the set of variables not in V and A_{-v} is the corresponding
%   assignment to these variables consistent with A.  In the case that |V| = 1,
%   this reduces to Gibbs Sampling.  NOTE that exp(LogBS) is not normalized to
%   sum to one at the end of this function (nor do you need to worry about that
%   in this function).
%
% Copyright (C) Daphne Koller, Stanford University, 2012

function LogBS = BlockLogDistribution(V, G, F, A)
if length(unique(G.card(V))) ~= 1
    disp('WARNING: trying to block sample invalid variable set');
    return;
end

% d is the dimensionality of all the variables we are extracting
d = G.card(V(1));

LogBS = zeros(1, d);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
% Compute LogBS by multiplying (adding in log-space) in the correct values from
% each factor that includes some variable in V.  
%
% NOTE: As this is called in the innermost loop of both Gibbs and Metropolis-
% Hastings, you should make this fast.  You may want to make use of
% G.var2factors, repmat,unique, and GetValueOfAssignment.
%
% Also you should have only ONE for-loop, as for-loops are VERY slow in matlab
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

All_Vars = unique([F(:).var]); #Retrieve all variables defined by our factor list_in_columns

[dummy, mapV] = ismember(V, All_Vars);
#dummy is a logical index indicating if V{i} is in All_Vars
#mapV is a matching index indicating which {j} V{1} is mapped to All_Vars
[others, mapOthers] = setdiff(All_Vars, V);
#others is the entries that only in All_Vars but not in V.
#mapOthers is the index of the entries that only in All_Vars but not in V.
ordering = [mapV mapOthers]; #This is a row vector that record the index of V and other than V in All_Vars.

firstMatrix = [1:d]'; #Creating a collumn vector equal to the 1:cardinality of variable V 1.

secondMatrix = repmat(firstMatrix, 1, length(V)); #The second matrix equals to the duplicated collumn vectors of first matrix by the number of variables in V. 

#The second entry equals to the number of rows of the output matrix
#The third entry is the number of collumns of the output matrix

thirdMatrix = repmat(A(others), d, 1); #Get the initial assignment of other variables, dublicate them into 2 rows

finalMatrix = [secondMatrix thirdMatrix]; #Combine the possible assignment of target and the initial assignment of the rest

finalMatrix(:, ordering) = finalMatrix; #Ensure the final matrix is sorted by the order of variables in its collumn, make the index # be the corresponding variable.

for i = 1:length(F),
  LogBS = LogBS + log(GetValueOfAssignment(F(i), finalMatrix(:, F(i).var))); #Factor product over all factors using the variable assignments over finalMatrix.
end

#The reason we need to do all these calculations, please refer to PPT 3.5.4 P5. 
#What we want here is only the numerator (unormalized measure).
#So for those variabiles, we do not fix their assignment to one value defined by G.
#We could have multiple Vs / random quantities, the returned probability disribution is their packed result on each dimension of the assignment.....But they must have same cardinalities.
#If we have only one random quantitie defined by V, we will have exactly the gibbs sampling defined in PPT.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Re-normalize to prevent underflow when you move back to probability space
LogBS = LogBS - min(LogBS);



