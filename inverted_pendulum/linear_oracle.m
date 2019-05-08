function [ g ] = linear_oracle( L,x,u )
%LINEAR_ORACLE Computes the output to a linear oracle
%   Computes g(x,u) = H*x + G*u, where H = L{1} and G = L{2} from the
%   parameter set L
    
    H = L{1};
    G = L{2};
    n = size(H,1);
    m = size(G,2);
    N = size(u,1)/m;
    H_tilde = [kron(eye(N),H),zeros(n*N,n)];	% augmented H matrix
    G_tilde = kron(eye(N),G);                   % augmented G matrix
    g = H_tilde*x + G_tilde*u;                  % linear oracle equation
    
end

