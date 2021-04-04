# NOT YET IMPLEMENTED - MATLAB CODE TO REWRITE BELOW
'''
function [S0_map] = fit_S0_map(T1_map, FA_vols, FAs, TR, varargin)
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
if ~exist('TR', 'var')
    TR = 4;
end
[n_y, n_x, n_z, n_angles] = size(FA_vols);

if n_angles ~= length(FAs)
    error('Length of flip angles input (FAs) does not match number of flip angle volumes');
end

S0_map = zeros(n_y, n_x, n_z);

for i_y = 1:n_y
    for i_x = 1:n_x
        for i_z = 1:n_z
            %Get T1 and signals in FA volumes for this voxel
            FA_signals = squeeze(FA_vols(i_y,i_x,i_z,:));
            T1 = T1_map(i_y,i_x,i_z);
    
            %Compute predicted signal with S0 = 1
            FA_signals_fit = signal_from_T1(T1, 1, FAs, TR);
            
            %Compute optimal (in least squares sense) S0
            S0_map(i_y, i_x, i_z) = FA_signals_fit \ FA_signals;
        end
    end
end'''
