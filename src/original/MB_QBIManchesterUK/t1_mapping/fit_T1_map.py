# NOT YET IMPLEMENTED - MATLAB CODE TO REWRITE BELOW
'''
function [T1_map] = fit_T1_map(FA_vols, FAs, varargin)
%COMPUTE_T1_MAP fit T1 map to set of variable flip angle volumes
%   [T1_map] = compute_T1_map(FA_vols, FAs, varargin)
%
% Inputs:
%      FA_vols - Flip angle volumes stacked along 4th dimension
%
%      FAs - vector contining flip angle associated with each volume
%      (length(FAs) must equal size(FA_vols,4)
%
%      varargin - *Insert description of input variable here*
%
%
% Outputs:
%      T1_map
%
%
% Example:
%
% Notes:
%
% See also:
%
% Created: 25-Oct-2017
% Author: Michael Berks 
% Email : michael.berks@manchester.ac.uk 
% Phone : +44 (0)161 275 7669 
% Copyright: (C) University of Manchester
[n_y, n_x, n_z, n_angles] = size(FA_vols);

if n_angles ~= length(FAs)
    error('Length of flip angles input (FAs) does not match number of flip angle volumes');
end

T1_map = zeros(n_y, n_x, n_z);
'''