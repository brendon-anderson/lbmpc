function [ x_bar,x_tilde,u_check,c ] = lbmpc_opt_unconstrained( A,B,Q,R,P,K,N,x_t,oracle_t )
%LBMPC_OPT Solves lbmpc optimization over a single horizon
%   Inputs:
%       State matrix, A
%       Input matrix, B
%       State cost matrix, Q>=0
%       Input cost matrix, R>0
%       Terminal state cost matrix, P>0
%       State feedback gain matrix, K
%       Horizon length, N
%       Initial condition, x_t
%       Oracle function handle, oracle_t = @(x,u) g(x,u)
    
    % objective and constraint setup
    n = size(A,1);          % state dimension
    m = size(B,2);          % input dimension
    n_bar = n*(N+1);        % augmented state dimension
    m_check = m*N;          % augmented input dimension
    Q_tilde = blkdiag( kron(eye(N),Q) , P );    % augmented state cost
    R_check = kron(eye(N),R);                   % augmented input cost
    D_t = [eye(n),zeros(n,n*N)];	% initial condition extraction
    D = [zeros(n*N,n),eye(n*N)];    % differencing matrix
    A_tilde = [kron(eye(N),A),zeros(n*N,n)];	% augmented A matrix
    B_tilde = kron(eye(N),B);     	% augmented B matrix
    A_bar = A_tilde;                % augmented A matrix
    B_bar = B_tilde;                % augmented B matrix
    K_bar = [kron(eye(N),K),zeros(m*N,n)];
    
    % cvx solution
    cvx_begin
        cvx_quiet true
        
        variable x_bar(n_bar)
        variable x_tilde(n_bar)
        variable u_check(m_check)
        variable c(m_check)
        
        minimize( x_tilde'*Q_tilde*x_tilde + u_check'*R_check*u_check )
        
        subject to
            D_t*x_bar == x_t              	% initial condition
            D_t*x_tilde == x_t             	% initial condition
            D*x_tilde == A_tilde*x_tilde ...
                + B_tilde*u_check ...
                + oracle_t(x_tilde,u_check) % tilde dynamics
            D*x_bar == A_bar*x_bar ...
                + B_bar*u_check             % bar dynamics
            u_check == K_bar*x_bar + c;    	% feedback law
            
    cvx_end

end

