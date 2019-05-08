function [ x_bar,x_tilde,u_check,c ] = lbmpc_opt( A,B,Q,R,P,K,N,x_t,S,I,F_N,oracle_t )
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
%       (Robust) state constraint polytopes, S=[S_1,S_2,...,S_N]
%       (Robust) input constraint polytopes, I=[I_0,I_1,...,I_(N-1)]
%       (Robust) maximum invariant set, F_N
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
    
    % polytope conversions
    A_S = [];
    b_S = [];
    A_I = [];
    b_I = [];
    for k = 1:N
        A_S = blkdiag(A_S,S(k).A);      % concatenate A_S blocks
        b_S = [b_S;S(k).b];             % concatenate b_S blocks
        A_I = blkdiag(A_I,I(k).A);      % concatenate A_I blocks
        b_I = [b_I;I(k).b];             % concatenate b_I blocks
    end
    A_S = [zeros(size(A_S,1),n),A_S];   % eliminate x_bar_t
    A_F = [zeros(size(F_N.A,1),n*N),F_N.A];	% A_F matrix
    b_F = F_N.b;                            % b_F matrix
    
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
                + oracle_t(x_tilde,u_check)   % tilde dynamics
            D*x_bar == A_bar*x_bar ...
                + B_bar*u_check             % bar dynamics
            u_check == K_bar*x_bar + c;    	% feedback law
            A_S*x_bar <= b_S                % state constraints
            A_I*u_check <= b_I              % input constraints
            A_F*x_bar <= b_F                % terminal constraint
            
    cvx_end

end

