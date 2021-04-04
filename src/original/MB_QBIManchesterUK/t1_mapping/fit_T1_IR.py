# NOT YET IMPLEMENTED - MATLAB CODE TO REWRITE BELOW
'''
function fit_T1_IR = fit_T1_IR(ims,TIs,TR,plot_y_n,chooseAlg)
%% Function for fitting T1 inversion recovery curve
% Inputs:
% ims - 4D matrix of images - 3D matrices for each inversion time
% TIs - vector of inversion times (s) corresponding to 4th dimension of ims
% TR - TR of sequence (s, assumed to be the same for each ti)
% plot_y_n - plot fit or not ('y' or 'n')
% chooseAlg - string specifying algorithm to use
%           - currently accepts 'fminsearchbnd' or 'nage04wd'
% Outputs:
% t1_ir_fit - 4D matrix - 3D matrices of T1, M0, R^2 of fit, exitflag, and
%                         index of starting values for accepted fit

%% Check inversion times match number of images
if size(ims,4)~=numel(TIs)
    error('Number of TI values does not match number of image matrices')
else
    % Continue, and make TIs a column vector if it isn't already
    TIs=TIs(:);
end

%% Check one TR is given
if numel(TR)~=1
    error('TR must be a single number')
end

%% Preallocate outputs
t1Maps=zeros(size(ims(:,:,:,1)));
m0Maps=zeros(size(ims(:,:,:,1)));
rsqMaps=zeros(size(ims(:,:,:,1)));
extflgMaps=zeros(size(ims(:,:,:,1)));
strtValsMaps=zeros(size(ims(:,:,:,1)));

%% Bounds for fitted parameters
m0LB=0;
m0UB=Inf;
t1LB=0;
t1UB=Inf;
options=optimset('Display', 'off','MaxFunEval',1000,'MaxIter',1000);

%% Number of starting values to try, and limits on T1 initial values
%  (M0 initial values picked per-voxel (see within loop))
numStrtVals=500;
seed=134564;
RandStream.setGlobalStream ...
    (RandStream('mt19937ar','seed',seed));
t1StrtLwr=1;%5e-3;
t1StrtUppr=500;%200e-3;

%% Signal model function
signalModel=@(a,TIs) abs(a(1).*(1-2.*exp(-TIs./a(2))+exp(-TR/a(2))));

%% Loop over slices and voxels, performing fit
for s=1:size(ims,3)
    for i=1:size(ims,1)
        for j=1:size(ims,2)
            thisVoxelTIs=squeeze(ims(i,j,s,:));
            % Get range for M0 initial values from signal at longest TI
            m0StrtLwr=thisVoxelTIs(end);
            m0StrtUppr=thisVoxelTIs(end)*20;
            
            % Loop over starting values
            ofv(1)=Inf;
            for strtValInd=1:numStrtVals
                a0=[(m0StrtLwr + (m0StrtUppr-m0StrtLwr).*rand(1,1));
                    (t1StrtLwr + (t1StrtUppr-t1StrtLwr).*rand(1,1))];
                switch chooseAlg
                    case 'fminsearchbnd'
                        [a,objFnVal,exitflag]=fminsearchbnd(@t1_ir_ls,a0,...
                            [m0LB t1LB],[m0UB t1UB],options);
                    case 'nage04wd'
                        [a,objFnVal,exitflag]=nag_e04wd_wrapper(@nagt1_ir_ls,a0,...
                            [],[],[],[m0LB t1LB],[m0UB t1UB],[],'cold');
                    otherwise
                        error('Invalid entry for chooseAlg')
                end
                ofv(strtValInd+1)=objFnVal;
                if ofv(strtValInd+1)<min(ofv(1:strtValInd))
                    % Calculate R^2 of fit
                    res=thisVoxelTIs-signalModel(a,TIs);
                    sumSqRes=sum(res.^2);
                    diffFromMean=thisVoxelTIs-mean(thisVoxelTIs);
                    sumSqDiffFromMean=sum(diffFromMean.^2);
                    rsqMaps(i,j,s)=1-(sumSqRes./sumSqDiffFromMean);
                    
                    % Collect values to output
                    t1Maps(i,j,s)=a(2);
                    m0Maps(i,j,s)=a(1);
                    extflgMaps(i,j,s)=exitflag;
                    strtValsMaps(i,j,s)=strtValInd;
                else
                    %if these starting values do not improve obj fn, do nothing
                end
            end
            % Plot if needed
            switch plot_y_n
                case 'y'
                    figure
                    xgrid=linspace(min(TIs),max(TIs));
                    fitVals=[m0Maps(i,j,s) t1Maps(i,j,s)];
                    paramFit=signalModel(fitVals,xgrid);
                    plot(TIs.*1e3,thisVoxelTIs,'kx',xgrid.*1e3,...
                        paramFit,'r-.','MarkerSize',18,'LineWidth',2);
                    xlabel('TI (ms)','FontSize',18);
                    ylabel('Signal intensity','FontSize',18);
                    set(gca,'YGrid','on','box','off','FontSize',18);
                case 'n'
                otherwise
                    error('!!! plot_y_n must be y or n')
            end
        end
    end
end

%% Assign output
fit_T1_IR=cat(4,t1Maps,m0Maps,rsqMaps,extflgMaps);

    function logliklihood = t1_ir_ls(a)
        % Least squares approach for now...
        signalModelEval=signalModel(a,TIs);
        logliklihood=sum((thisVoxelTIs-signalModelEval).^2);
    end

    function [mode, x, grad, user] = nagt1_ir_ls(mode, n, a, grad, nstate, user)
        % Least squares approach for now...
        signalModelEval=signalModel(a,TIs);
        x=sum((thisVoxelTIs-signalModelEval).^2);
    end

end '''