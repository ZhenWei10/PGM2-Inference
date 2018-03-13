% MHUNIFORMTRANS
%
%  MCMC Metropolis-Hastings transition function that
%  utilizes the uniform proposal distribution.
%  A - The current joint assignment.  This should be
%      updated to be the next assignment
%  G - The network
%  F - List of all factors
%
% Copyright (C) Daphne Koller, Stanford University, 2012

function A = MHUniformTrans(A, G, F)

% Draw proposed new state from uniform distribution
A_prop = ceil(rand(1, length(A)) .* G.card);

p_acceptance = 0.0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
% Compute acceptance probability
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

forward = 1.0;
backward = 1.0;

for i = 1:length(F),
  forward = forward * GetValueOfAssignment(F(i), A_prop(F(i).var) ); #Pi(Proposed) 
  backward = backward * GetValueOfAssignment(F(i), A(F(i).var) ); #Pi(Current)
end

#This is some how a chain rule toward the unormalized measure of joint probability, 
#since their partition function always equal, here we ignore normalization by their division.

p_acceptance = min(1,(forward / backward)); #likelihood of forward transision / likelihood of backward transision.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Accept or reject proposal
if rand() < p_acceptance
    % disp('Accepted');
    A = A_prop;
end