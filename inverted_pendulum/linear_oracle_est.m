function [ L ] = linear_oracle_est( Y,X,U )
%LINEAR_ORACLE_EST Estimates the parameters of a linear oracle
%   Given the n-by-(t+1) state trajectory matrix X=[x(0),x(1),...,x(t)],
%   the m-by-(t+1) input matrix U=[u(0),u(1),...,u(t)], and the n-by-(t+1)
%   observation matrix Y=[Y(0),Y(1),...,Y(t)], where
%   Y(k)=x(k+1)-(A*x(k)+B*u(k)), as well as the LTI model coefficients A
%   (n-by-n) and B (n-by-m), computes the optimal parameters H and G of the
%   linear oracle of the form O(x,u)=H*x+G*u. These are returned as
%   L={H,G}.

n = size(X,1);
m = size(U,1);
cvx_begin
    cvx_quiet true
    variable H(n,n)
    variable G(n,m)
    minimize( norm( Y - H*X - G*U , 'fro' ) )
cvx_end
L = {H,G};      % updated oracle parameters

end

