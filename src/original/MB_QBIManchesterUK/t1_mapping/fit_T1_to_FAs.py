# NOT YET IMPLEMENTED - MATLAB CODE TO REWRITE BELOW
'''
function [T1, S0, linear_fit, signal_sse, rsq] = fit_T1_to_FAs(FA_signals, FAs, TR, debug)
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
if ~exist('debug', 'var') 
    debug = false;
end

%Convert FA signals anf flip angles so we can write them as a linera
%function of Exp(-TR/T1)
y = FA_signals ./ sind(FAs);
x = FA_signals ./ tand(FAs);

%Fit best line
[p, S] = polyfit(x,y,1);

%Extract T1 and S0
E1 = p(1);
S0 = p(2) / (1 - E1);
T1 = -TR / log(p(1));
linear_fit = S.normr;

%Compute model fit in signal space
%19/10/18 DJM - Also output R-squared of fit
if nargout > 3 || debug
    fitted_signals = signal_from_T1(T1, S0, FAs, TR);
    signal_sse = sum((FA_signals-fitted_signals).^2);
    ss_diff_from_mean = sum((FA_signals - mean(FA_signals)).^2);
    rsq = 1 - (signal_sse./ss_diff_from_mean);
end

%Plot fit in linear and signal space
if debug
    FAs_cont = linspace(0, max(FAs), 20);
    fitted_signals_cont = signal_from_T1(T1, S0, FAs_cont, TR);
    figure;
    subplot(1,2,1);
    plot(x, y, 'rx'); hold all;
    plot(x, x*p(1) + p(2), 'bo');
    plot([0 max(x)], [p(2) max(x)*p(1) + p(2)], 'b--');
    title(['Linear fit, T_1 = ' num2str(T1, 3) ', S_0 = ' num2str(S0,3)]);
    xlabel('S_i / tan(\alpha)');
    ylabel('S_i / sin(\alpha)');
    subplot(1,2,2);
    plot(FAs, FA_signals, 'rx'); hold all;
    plot(FAs, fitted_signals, 'bo');
    plot(FAs_cont, fitted_signals_cont, 'b--');    
    title('Fit in signal space');
    xlabel('Flip angle - \alpha (degrees)');
    ylabel('Signal - S_i');
end'''
    
