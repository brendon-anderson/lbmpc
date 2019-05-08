function [ RT ] = robust_tubes( A,B,K,N,W )
%ROBUST_TUBES Computes the robust tubes for a lbmpc problem
%   Inputs:
%       State matrix, A
%       Input matrix, B
%       State feedback gain matrix, K
%       Horizon length, N
%       Error polytope, W

    RT = repmat(0*W,N,1);           % initialize RT polytope array
    for k = 1:N
        RT_k = 0*W;                 % initialize RT_k polytope
        for j = 0:k-1
            RT_k = RT_k + ((A+B*K)^j)*W;	% minkowski sum
        end
        RT(k) = RT_k;               % store RT_k
    end
    RT = [0*W,RT];                  % include RT_0

end

