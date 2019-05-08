clc; clear; close all;

%% user inputs
% system parameters
t_f = 51;                       % final time to simulate to
T = 0.1;                        % sampling time
M = 1;                          % pendulum mass
L = 1;                          % pendulum length
f = 0.01;                   	% friction coefficient
g = 10;                         % gravity
A = [1,T;
    g*T/L,1-f*T/(M*L^2)];    	% A matrix
A_0 = A; A_0(2,1) = 0;          % nonlinear A matrix
B = [0;T];                      % input matrix
m = size(B,2);               	% number of inputs
n = size(A,1);                	% number of states
x_0 = [0.3;0];                	% initial condition
d_min = 0;                      % uniform noise lower bound
d_max = 0;                      % uniform noise upper bound
F = 0.3*A;                    	% perturbation to A
G = 0.3*B;                  	% perturbation to B
system = @(x,u,t) A*x + B*u ...         % nominal model
    + F*x + G*u ...                     % linear perturbation
    + [0;g*T*sin(x(1))/L] ...           % nonlinearity
    + (d_min+(d_max-d_min).*rand(n,1)); % uniform noise

% control synthesis parameters
Q = eye(n);                     % state deviation tuning matrix
R = eye(m);                     % control effort tuning matrix
N = 5;                          % horizon length

% control synthesis constraints
lb_X = -inf*ones(n,1);        	% state lower bound
ub_X = inf*ones(n,1);         	% state upper bound
lb_U = -inf*ones(m,1);         	% input lower bound
ub_U = inf*ones(m,1);         	% input upper bound
lb_E = [0;-g*T*sin(x_0(1))/L];	% error lower bound
ub_E = [0;g*T*sin(x_0(1))/L];  	% error upper bound
lb_D = d_min*ones(n,1);        	% noise lower bound
ub_D = d_max*ones(n,1);        	% noise upper bound

% lbmpc oracle choice
oracle_est = @(Y,X,U) ...
    linear_oracle_est(Y,X,U);   % define oracle estimator
oracle = @(L,x,u) ...
    linear_oracle(L,x,u);       % g(x,u) = H*x + G*u, H = L{1}, G = L{2}
H_t = zeros(size(A));           % initialize linear oracle parameter
G_t = zeros(size(B));           % initialize linear oracle parameter
L_t = {H_t,G_t};                % initialize oracle parameter set

%% initial calculations
% control synthesis parameters
[K,P,~] = dlqr(A,B,Q,R);      	% solve infinite horizon LQR problem
K = -K;                         % convert to u = K*x notation

% control synthesis constraints
X = Polyhedron('lb',lb_X,'ub',ub_X);    % state constraint polytope
U = Polyhedron('lb',lb_U,'ub',ub_U);    % input constraint polytope
E = Polyhedron('lb',lb_E,'ub',ub_E);    % modeling error polytope
D = Polyhedron('lb',lb_D,'ub',ub_D);   	% noise polytope
W = D+E;                                % overall disturbance polytope

mpt_system = LTISystem('A', A,'B',B);   % generate LTI model for MPT
mpt_system.x.with('setConstraint');     % declare state set constraint
mpt_system.x.setConstraint = X;         % state constraint
mpt_system.u.with('setConstraint');     % declare input set constraint
mpt_system.u.setConstraint = U;         % input constraint
Omega = mpt_system.invariantSet();      % compute control invariant set

%% uncontrolled
% autonomous system
x_t = 0*x_0;                 	% initialize state
X_un = zeros(n,t_f);            % initialize state trajectory matrix
for t = 1:t_f
    X_un(:,t) = x_t;            % store state
    u_t = zeros(m,1);          	% input to send to system
    x_t = system(x_t,u_t,t);  	% simulate system response
end

%% lqr
% solve lqr
x_t = x_0;                      % initialize state
X_lqr = zeros(n,t_f);           % initialize state trajectory matrix
U_lqr = zeros(m,t_f);           % initialize input trajectory matrix
for t = 1:t_f
    X_lqr(:,t) = x_t;       	% store state
    u_t = K*x_t;                % input to send to system
    U_lqr(:,t) = u_t;       	% store input
    x_t = system(x_t,u_t,t);  	% simulate system response
end

%% linear mpc
% solve for robust tubes
RT = robust_tubes(A,B,K,N,W);   % RT = [RT_0,RT_1,...,RT_N]

% solve for robust constraint polytopes
S = repmat(0*X,N,1);        	% initialize S polytope array
I = repmat(0*U,N,1);         	% initialize RT polytope array
for k = 1:N-1
   	S_k = X - RT(k+1);          % generate S_k
   	I_k = U - (K*RT(k+1));      % generate I_k
    S(k) = S_k;                 % store S_k
    I(k+1) = I_k;               % store I_k
end
S(N) = X - RT(N+1);             % store S_N
I(1) = U - (K*RT(1));           % store I_0
F_N = Omega - RT(N+1);          % generate F_N

% solve mpc
x_t = x_0;                      % initialize state
X_mpc = zeros(n,t_f);           % initialize state trajectory matrix
U_mpc = zeros(m,t_f);           % initialize input trajectory matrix
for t = 1:t_f
    X_mpc(:,t) = x_t;           % store state
    [x_bar,u_check,c] ...
        = mpc_opt(A,B,Q,R,P,K,N,x_t,S,I,F_N);	% solve
   	u_t = u_check(1:m);        	% input to send to system
    U_mpc(:,t) = u_t;         	% store input
    x_t = system(x_t,u_t,t);   	% simulate system response
end

%% lbmpc
% solve for robust tubes
RT = robust_tubes(A,B,K,N,W);   % RT = [RT_0,RT_1,...,RT_N]

% solve for robust constraint polytopes
S = repmat(0*X,N,1);        	% initialize S polytope array
I = repmat(0*U,N,1);         	% initialize RT polytope array
for k = 1:N-1
   	S_k = X - RT(k+1);          % generate S_k
   	I_k = U - (K*RT(k+1));      % generate I_k
    S(k) = S_k;                 % store S_k
    I(k+1) = I_k;               % store I_k
end
S(N) = X - RT(N+1);             % store S_N
I(1) = U - (K*RT(1));           % store I_0
F_N = Omega - RT(N+1);          % generate F_N

% solve lbmpc
x_t = x_0;                      % initialize state
X_lbmpc = zeros(n,t_f);         % initialize state trajectory matrix
U_lbmpc = zeros(m,t_f);         % initialize input trajectory matrix
for t = 1:t_f
    X_lbmpc(:,t) = x_t;     	% store state
    oracle_t = @(x,u)oracle(L_t,x,u);	% update oracle parameters
    [x_bar,x_tilde,u_check,c] ...
        = lbmpc_opt(A,B,Q,R,P,K,N,x_t,S,I,F_N,oracle_t);	% solve
   	u_t = u_check(1:m);         	% input to send to system
    U_lbmpc(:,t) = u_t;             % store input
    x_t = system(x_t,u_t,t);       	% simulate system response
    Y = [X_lbmpc(:,2:t),x_t] ...
        - A*X_lbmpc(:,1:t) - B*U_lbmpc(:,1:t);	% compute observation matrix
    L_t = oracle_est(Y, ...
        X_lbmpc(:,1:t),U_lbmpc(:,1:t));      	% update oracle parameter estimates
end

%% plot

% plot controlled response
t = 0:t_f-1;            % shift time indices so initial condition is t=0
figure();
hold on;
box on;
stairs(t,X_lbmpc(1,:),'bo','markersize',3);
stairs(t,X_lbmpc(2,:),'b*','markersize',3);
stairs(t,X_lqr(1,:),'r-o','markersize',3);
stairs(t,X_lqr(2,:),'r-*','markersize',3);
stairs(t,X_mpc(1,:),'go','markersize',3);
stairs(t,X_mpc(2,:),'g*','markersize',3);
xlim([min(t),max(t)]);
xlabel('$t$','interpreter','latex');
ylabel('$x_i(t)$','interpreter','latex');
legend({'LBMPC, $i=1$','LBMPC, $i=2$',...
    'LQR, $i=1$','LQR, $i=2$',...
    'MPC, $i=1$','MPC, $i=2$'},...
    'interpreter','latex','location','southeast');
set(gca,'ticklabelinterpreter','latex');
