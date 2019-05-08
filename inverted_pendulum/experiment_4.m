clc; clear; close all;

%% user inputs
% system parameters
t_f = 101;                    	% final time to simulate to
T = 0.1;                        % sampling time
M_p = 0.5;                    	% pendulum mass
M_c = 1;                      	% cart mass
L = 5;                        	% pendulum length
g = 10;                         % gravity
A = [1,T,0,0;
    0,1,T*M_p*g/M_c,0;
    0,0,1,T;
    0,0,T*(M_p+M_c)*g/(L*M_c),1];	% A matrix
B = [0;T/M_c;0;T/(L*M_c)];         	% input matrix
m = size(B,2);               	% number of inputs
n = size(A,1);                	% number of states
x_0 = [0;0;0.5;0];             	% initial condition
system = @(x,u,t) inverted_pend( x,M_p,M_c,L,g,u,T ); % define system

% control synthesis parameters
Q = eye(n);                     % state deviation tuning matrix
R = eye(m);                     % control effort tuning matrix
N = 5;                        	% horizon length

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

%% uncontrolled
% autonomous system
x_t = x_0;                      % initialize state
X_un = zeros(n,t_f);            % initialize state trajectory matrix
for t = 1:t_f
    X_un(:,t) = x_t;            % store state
    u_t = zeros(m,1);          	% input to send to system
    [~,x_t] = system(x_t,u_t,t);	% simulate system response
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
    [~,x_t] = system(x_t,u_t,t);  	% simulate system response
end

%% lbmpc

% solve lbmpc
x_t = x_0;                      % initialize state
X_lbmpc = zeros(n,t_f);         % initialize state trajectory matrix
U_lbmpc = zeros(m,t_f);         % initialize input trajectory matrix
for t = 1:t_f
    X_lbmpc(:,t) = x_t;     	% store state
    oracle_t = @(x,u)oracle(L_t,x,u);	% update oracle parameters
    [x_bar,x_tilde,u_check,c] ...
        = lbmpc_opt_unconstrained(A,B,Q,R,P,K,N,x_t,oracle_t);	% solve
   	u_t = u_check(1:m);         	% input to send to system
    U_lbmpc(:,t) = u_t;             % store input
    [~,x_t] = system(x_t,u_t,t);   	% simulate system response
    Y = [X_lbmpc(:,2:t),x_t] ...
        - A*X_lbmpc(:,1:t) - B*U_lbmpc(:,1:t);	% compute observation matrix
    L_t = oracle_est(Y, ...
        X_lbmpc(:,1:t),U_lbmpc(:,1:t));      	% update oracle parameter estimates
end

%% plot

% plot uncontrolled response
t = 0:t_f-1;            % shift time indices so initial condition is t=0
figure();
hold on;
box on;
stairs(t,X_un(3,:),'bo','markersize',3);
xlim([min(t),max(t)]);
xlabel('$t$','interpreter','latex');
ylabel('$x_i(t)$','interpreter','latex');
legend({'No Control, $i=3$'},...
    'interpreter','latex','location','northwest');
set(gca,'ticklabelinterpreter','latex');

% plot controlled response
t = 0:t_f-1;            % shift time indices so initial condition is t=0
figure();
hold on;
box on;
stairs(t,X_lbmpc(3,:),'bo','markersize',3);
stairs(t,X_lqr(3,:),'ro','markersize',3);
xlim([min(t),max(t)]);
xlabel('$t$','interpreter','latex');
ylabel('$x_i(t)$','interpreter','latex');
legend({'LBMPC, $i=3$','LQR, $i=3$'},...
    'interpreter','latex','location','northeast');
set(gca,'ticklabelinterpreter','latex');
tightfig;             	% remove excess whitespace
saveas(gcf,'cart_pend_stabilization.pdf'); % save figure
