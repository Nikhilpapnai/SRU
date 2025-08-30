%% GMM + per-cluster sparse GPR + PSO (Output 1 only)
% - Automatic cluster selection (silhouette + elbow diagnostic)
% - Skip tiny clusters (< 50 samples, fallback to global GPR)
% - ActiveSetSize = min(3000, round(0.7 * Ntrain))
% - Strong PSO config for overnight runs
%
% NOTE: Run a quick test first by setting quickTest = true below.

clearvars; close all; clc;
rng(0);

%% -------------------- User options --------------------
input_file  = "/Users/sam/Desktop/untitled folder/Cleaned_IN_Table.csv";
output_file = "/Users/sam/Desktop/untitled folder/Cleaned_OUT_Table.csv";

quickTest = false;          % true = small run, false = overnight run
maxClusters = 8;            % max clusters to evaluate
minClusterSize = 50;        % min cluster size allowed
ActiveSetCeiling = 3000;    % memory cap for sparse GPR

% PSO settings
if quickTest
    swarmSize = 10; maxIter = 10;
else
    swarmSize = 60; maxIter = 300;
end
w = 0.72; c1 = 1.49; c2 = 1.49;

%% -------------------- Load and clean data --------------------
X_all = readmatrix(input_file);   % NxM
Y_all = readmatrix(output_file);  % NxP
if isempty(X_all) || isempty(Y_all)
    error('Input files empty or wrong paths.');
end
Y1_all = Y_all(:,1);              % Only output 1

% Remove NaN rows
valid_rows = all(~isnan([X_all, Y1_all]),2);
X_all = X_all(valid_rows,:);
Y1_all = Y1_all(valid_rows,:);

fprintf('Loaded %d samples after NaN removal.\n', size(X_all,1));

% Outlier removal
Z = zscore([X_all, Y1_all]);
outlier_mask = any(abs(Z) > 3, 2);
fprintf('Outliers removed: %d\n', sum(outlier_mask));
X_all = X_all(~outlier_mask,:);
Y1_all = Y1_all(~outlier_mask,:);
fprintf('Remaining samples: %d\n', size(X_all,1));

%% -------------------- Train/Validation split --------------------
N = size(X_all,1);
idx = randperm(N);
nTrain = round(0.7 * N);
Xtrain = X_all(idx(1:nTrain),:);
Y1train = Y1_all(idx(1:nTrain),:);
Xvalid = X_all(idx(nTrain+1:end),:);
Y1valid = Y1_all(idx(nTrain+1:end),:);

fprintf('Train: %d samples, Valid: %d samples\n', size(Xtrain,1), size(Xvalid,1));

%% -------------------- Standardize inputs --------------------
muX = mean(Xtrain,1);
sigmaX = std(Xtrain,[],1); sigmaX(sigmaX==0) = 1;
Xtrain_z = (Xtrain - muX) ./ sigmaX;
Xvalid_z = (Xvalid - muX) ./ sigmaX;

%% -------------------- Cluster selection (silhouette + elbow) --------------------
maxK_try = min(maxClusters, size(Xtrain,1)-1);
if maxK_try < 2
    fprintf('Not enough data to try multiple clusters. Using single cluster (K=1).\n');
    Krange = [];
    silVals = [];
    bestK = 1;
else
    Krange = 2:maxK_try;
    silVals = nan(numel(Krange),1);
    wcssVals = nan(numel(Krange),1);
    fprintf('Evaluating cluster counts for K = %d..%d (silhouette + elbow diagnostic)...\n', Krange(1), Krange(end));
    for ii = 1:numel(Krange)
        K = Krange(ii);
        try
            idxK = kmeans(Xtrain_z,K,'Replicates',5,'MaxIter',500,'Display','off');
            s = silhouette(Xtrain_z, idxK);
            silVals(ii) = mean(s);
            % compute WCSS (within-cluster sum of squares) in a robust way
            wcss = 0;
            for kk = 1:K
                pts = Xtrain_z(idxK==kk,:);
                if ~isempty(pts)
                    cen = mean(pts,1);
                    dif = pts - cen;
                    wcss = wcss + sum(dif(:).^2);
                end
            end
            wcssVals(ii) = wcss;
            fprintf('K=%d  Silhouette=%.4f  WCSS=%.4e\n', K, silVals(ii), wcssVals(ii));
        catch ME
            warning('K=%d failed: %s', K, ME.message);
            silVals(ii) = -Inf;
            wcssVals(ii) = NaN;
        end
    end

    % Choose best K by silhouette
    [~,bestIdx] = max(silVals);
    if isempty(bestIdx) || isnan(bestIdx)
        bestK = 1;
    else
        bestK = Krange(bestIdx);
    end
    fprintf('Selected K=%d clusters by silhouette.\n', bestK);

    % Optional elbow diagnostic print (not used for selection)
    if numel(Krange) >= 2
        relDecrease = diff(wcssVals)./wcssVals(1:end-1);
        fprintf('Elbow diagnostic (relative decreases of WCSS):\n');
        disp(relDecrease');
    end
end

%% -------------------- Fit GMM for gating (if K>1) --------------------
if bestK >= 2
    try
        bestGMM = fitgmdist(Xtrain_z, bestK, 'CovarianceType','full', ...
            'RegularizationValue',1e-6, 'Options', statset('MaxIter',1000,'TolFun',1e-6,'Display','off'), ...
            'Replicates', 5);
        fprintf('Fitted GMM with K=%d components.\n', bestK);
    catch ME
        warning('GMM fit failed for K=%d: %s. Falling back to K=1 (no gating).', bestK, ME.message);
        bestGMM = [];
        bestK = 1;
    end
else
    bestGMM = [];
end
K = bestK;

%% -------------------- Global fallback GPR --------------------
fallbackActive = min(1000, ActiveSetCeiling);
fprintf('Training global fallback sparse GPR (ActiveSet=%d)...\n', fallbackActive);
try
    globalGPR = fitrgp(Xtrain, Y1train, ...
        'Basis','constant', 'KernelFunction','ardsquaredexponential', ...
        'Standardize', true, 'FitMethod','sr', 'ActiveSetSize', fallbackActive, 'Verbose', 1);
catch ME
    warning('Global fallback GPR failed at ActiveSet=%d: %s. Trying smaller ActiveSet.', fallbackActive, ME.message);
    globalGPR = fitrgp(Xtrain, Y1train, ...
        'Basis','constant', 'KernelFunction','ardsquaredexponential', ...
        'Standardize', true, 'FitMethod','sr', 'ActiveSetSize', max(200,fallbackActive-200), 'Verbose', 1);
end

%% -------------------- Cluster assignment --------------------
if ~isempty(bestGMM)
    posterior_train = posterior(bestGMM, Xtrain_z);   % Ntrain x K
    [~, hard_assign_train] = max(posterior_train, [], 2);
else
    hard_assign_train = ones(size(Xtrain,1),1); % everyone in cluster 1
end

%% -------------------- Train per-cluster sparse GPR experts --------------------
gprExperts = cell(K,1);
clusterSizes = zeros(K,1);
ActiveSetSize = min(ActiveSetCeiling, round(0.7 * size(Xtrain,1)));
ActiveSetSize = max(100, ActiveSetSize); % ensure not too small

for k = 1:K
    idx_k = find(hard_assign_train == k);
    clusterSizes(k) = numel(idx_k);
    fprintf('Cluster %d: %d training samples\n', k, clusterSizes(k));
   
    if clusterSizes(k) < minClusterSize
        fprintf('⚠️ Cluster %d too small (<%d). Using global fallback.\n', k, minClusterSize);
        gprExperts{k} = globalGPR;
        continue;
    end
   
    Xk = Xtrain(idx_k,:);
    Yk = Y1train(idx_k,:);
    asz = min(ActiveSetSize, clusterSizes(k)-1);
    asz = max(100, asz); % enforce minimum
   
    fprintf('Training cluster %d local GPR (ActiveSet=%d)...\n', k, asz);
    try
        gprExperts{k} = fitrgp(Xk, Yk, ...
            'Basis','constant', 'KernelFunction','ardsquaredexponential', ...
            'Standardize', true, 'FitMethod','sr', 'ActiveSetSize', asz, 'Verbose', 1);
    catch ME
        warning('Cluster %d GPR failed: %s. Using global fallback.\n', k, ME.message);
        gprExperts{k} = globalGPR;
    end
end

%% -------------------- Validation (mixture-weighted prediction) --------------------
if ~isempty(bestGMM)
    post_valid = posterior(bestGMM, Xvalid_z);  % Nvalid x K
else
    post_valid = ones(size(Xvalid,1),1);
end

Ypred_valid = zeros(size(Xvalid,1),1);
for i = 1:size(Xvalid,1)
    p_k = post_valid(i,:);           % 1 x K (or scalar 1)
    mu_k = zeros(K,1);
    for k = 1:K
        try
            mu_k(k) = predict(gprExperts{k}, Xvalid(i,:));
        catch
            mu_k(k) = predict(globalGPR, Xvalid(i,:));
        end
    end
    Ypred_valid(i) = sum(p_k .* mu_k'); % weighted mean (works if p_k scalar or row vector)
end

rmse1 = sqrt(mean((Y1valid - Ypred_valid).^2,'omitnan'));
SSres = sum((Y1valid - Ypred_valid).^2,'omitnan');
SStot = sum((Y1valid - mean(Y1valid)).^2,'omitnan');
R2_1 = 1 - SSres / SStot;

fprintf('\nValidation RMSE (output1, mixture GPR): %.6f\n', rmse1);
fprintf('Validation R^2 (output1, mixture GPR): %.6f\n', R2_1);

figure('Name','True vs Mixture-GPR Prediction');
plot(1:length(Y1valid), Y1valid,'r','DisplayName','True'); hold on;
plot(1:length(Y1valid), Ypred_valid,'k--','DisplayName','Mixture GPR Pred');
legend; grid on;
title(sprintf('Output1: True vs Pred (R^2 = %.4f)', R2_1));

%% -------------------- PSO using the mixture surrogate --------------------
lb = min(Xtrain,[],1);
ub = max(Xtrain,[],1);
velMax = 0.2*(ub-lb);
velMin = -velMax;

% Initialize swarm
Xswarm = repmat(lb, swarmSize, 1) + rand(swarmSize, size(Xtrain,2)) .* repmat(ub-lb, swarmSize,1);
Vswarm = zeros(swarmSize, size(Xtrain,2));
pbest = Xswarm;
pbestVal = inf(swarmSize,1);
for i = 1:swarmSize
    pbestVal(i) = pso_mixture_fitness(Xswarm(i,:), bestGMM, gprExperts, globalGPR, muX, sigmaX);
end
[gbestVal, gbestIdx] = min(pbestVal);
gbest = pbest(gbestIdx,:);

convergence = zeros(maxIter,1);
fprintf('Starting PSO (swarm=%d, iter=%d)...\n', swarmSize, maxIter);
for iter = 1:maxIter
    for i = 1:swarmSize
        r1 = rand(1,size(Xtrain,2)); r2 = rand(1,size(Xtrain,2));
        Vswarm(i,:) = w*Vswarm(i,:) + c1*r1.*(pbest(i,:) - Xswarm(i,:)) + c2*r2.*(gbest - Xswarm(i,:));
        Vswarm(i,:) = max(min(Vswarm(i,:), velMax), velMin);
        Xswarm(i,:) = Xswarm(i,:) + Vswarm(i,:);
        Xswarm(i,:) = max(min(Xswarm(i,:), ub), lb);
       
        val = pso_mixture_fitness(Xswarm(i,:), bestGMM, gprExperts, globalGPR, muX, sigmaX);
        if val < pbestVal(i)
            pbestVal(i) = val; pbest(i,:) = Xswarm(i,:);
        end
        if val < gbestVal
            gbestVal = val; gbest = Xswarm(i,:);
        end
    end
    convergence(iter) = -gbestVal;
    if mod(iter,20) == 0 || iter == 1
        fprintf('PSO iter %d/%d best predicted (mixture) = %.6f\n', iter, maxIter, -gbestVal);
    end
end

% Report PSO results
opt_input = gbest;
if isempty(bestGMM)
    opt_post = 1;
else
    opt_post = posterior(bestGMM, (opt_input - muX)./sigmaX); % 1 x K
end

mu_k_opt = zeros(K,1);
for k = 1:K
    try
        mu_k_opt(k) = predict(gprExperts{k}, opt_input);
    catch
        mu_k_opt(k) = predict(globalGPR, opt_input);
    end
end
opt_predicted = sum(opt_post .* mu_k_opt');

fprintf('\nPSO found input (optimize output1):\n'); disp(opt_input);
fprintf('Mixture-GPR predicted output1 at found input: %.6f\n', opt_predicted);

figure('Name','PSO Convergence'); plot(1:maxIter, convergence, 'LineWidth', 1.4); grid on;
xlabel('Iteration'); ylabel('Predicted output1'); title('PSO convergence (mixture surrogate)');

%% -------------------- Save results --------------------
results.opt_input = opt_input;
results.opt_predicted = opt_predicted;
results.R2_1 = R2_1;
results.gmm = bestGMM;
results.gprExperts = gprExperts;
results.globalGPR = globalGPR;
results.clusterSizes = clusterSizes;
save('GMM_GPR_PSO_output1_full.mat','results');
fprintf('Saved results to GMM_GPR_PSO_output1_full.mat\n');

%% -------------------- Local function --------------------
function f = pso_mixture_fitness(x, gmmModel, gprExperts, globalGPR, muX, sigmaX)
    % x: 1xn raw-scale input
    % returns value to minimize (negative predicted output1)
    try
        xr_z = (x - muX) ./ sigmaX;
        if isempty(gmmModel)
            p_k = 1; % single-cluster case
            K = 1;
        else
            p_k = posterior(gmmModel, xr_z); % 1 x K
            K = numel(gprExperts);
        end
        mu_k = zeros(K,1);
        for k = 1:K
            try
                mu_k(k) = predict(gprExperts{k}, x); % raw-scale input for GPR
            catch
                mu_k(k) = predict(globalGPR, x);
            end
        end
        pred = sum(p_k .* mu_k');
        if isnan(pred)
            f = 1e6;
        else
            f = -pred; % minimize negative -> maximize predicted output
        end
    catch
        f = 1e6;
    end
end
