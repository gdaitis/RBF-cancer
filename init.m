clear all;

%% Data preparation
data = load('breastcancer.mat');
x = data.cancerInputs;
y = data.cancerTargets(:,1);

% Use 80% of the set for training and 20% for testing.
indexes = randsample(1:size(x,1), round(size(x,1)*0.8));
x_train = x(indexes, :)';
y_train = y(indexes, :)';
x_test = x;
x_test(indexes, :) = [];
x_test = x_test';
y_test = y;
y_test(indexes, :) = [];
y_test = y_test';

N_train = size(x_train,2); % no. of training data. Should be 140.
N_test=size(x_test,2); % no. of test data. Should be 559.

Ms = [2, 5, 10, 15, 20, 50, 100, 200, 300, 400, 559]; % Select amounts of centers

results_data = {};

for m = 1 : length(Ms)
    M = Ms(m);
    
    %% Trying to find best parameters
    attempts = 20; % no. of random attempts

    lowest_MSE = inf;
    best_centers = [];
    best_d_max = [];

    for attempt = 1 : attempts
        index = randsample(1:size(x_train,2), M); % random selection of centers
        centers = x_train(:,index); 

        % Compute d_max. d_max is the maximum distance between the selected
        % centers.
        d_max=0.0;
        for i = 1 : M
            for j = 1: M
                d_max = max(d_max, dist(centers(:,i)',centers(:,j)));
            end
        end

        % Define the radial basis function used.
        % input is an input data and i is the i-th center.
        rbf_i = @(input,i) exp( -M / d_max^2.0 * dist(input', centers(:,i))^2.0 );
        % Construct the interpolation matrix.
        interpolation_mat = zeros(N_train, M);
        for r = 1: N_train
            for c = 1: M
                interpolation_mat(r,c) = rbf_i(x_train(:,r), c);
            end
        end
        interpolation_mat = horzcat(ones(N_train, 1), interpolation_mat);
        % Calculate the weights
        w = pinv(interpolation_mat) * y_train';

        %% RBFN Testing:
        bias = w(1,1); %retrieve bias for cleaner code later.

        y_test_outcome = zeros(1,N_test); 
        for i = 1 : N_test
            for j = 1 : M
                y_test_outcome(1,i) = y_test_outcome(1,i) + w(j+1,1) * rbf_i( x_test(:,i), j);
            end
            y_test_outcome(1,i) = y_test_outcome(1,i) + bias;
        end

        %% Performance of RBFN:

        abs_errors_test = abs(y_test_outcome - y_test);
        % Count no. of correct predictions.
        epsilon = 0.5;
        correct_test=0; 
        for i = 1 : N_test
            if( abs_errors_test(1, i) < epsilon)
                correct_test = correct_test + 1;
            end
        end

        % Compute MSE
        MSE_testing = (abs_errors_test.^2 * ones(N_test, 1))/ N_test;
        if (MSE_testing < lowest_MSE)
            lowest_MSE = MSE_testing;
            best_centers = centers;
            best_d_max = d_max;
        end
    end

    %% Re-train with best parameters found
    % Define the radial basis function used.
    % input is an input data and i is the i-th best center.
    rbf_i = @(input,i) exp( -M / best_d_max^2.0 * dist(input', best_centers(:,i))^2.0 );
    % Construct the interpolation matrix.
    interpolation_mat = zeros(N_train, M);
    for r = 1: N_train
        for c = 1: M
            interpolation_mat(r,c) = rbf_i(x_train(:,r), c);
        end
    end
    interpolation_mat = horzcat(ones(N_train, 1), interpolation_mat);
    % Calculate the weights
    w = pinv(interpolation_mat) * y_train';

    %% RBFN Testing:
    bias = w(1,1); %retrieve bias for cleaner code later.
    y_training_outcome = zeros(1,N_train); 
    for i = 1 : N_train
        for j = 1 : M
            y_training_outcome(1,i) = y_training_outcome(1,i) + w(j+1,1) * rbf_i( x_train(:,i), j);
        end
        y_training_outcome(1,i) = y_training_outcome(1,i) + bias;
    end

    y_test_outcome = zeros(1,N_test); 
    for i = 1 : N_test
        for j = 1 : M
            y_test_outcome(1,i) = y_test_outcome(1,i) + w(j+1,1) * rbf_i( x_test(:,i), j);
        end
        y_test_outcome(1,i) = y_test_outcome(1,i) + bias;
    end

    %% Performance of RBFN:
    abs_errors_training = abs(y_training_outcome - y_train);
    % Count no. of correct predictions.
    epsilon = 0.5;
    correct_training=0; 
    for i = 1 : N_train
        if( abs_errors_training(1, i) < epsilon)
            correct_training = correct_training + 1;
        end
    end

    abs_errors_test = abs(y_test_outcome - y_test);
    % Count no. of correct predictions.
    epsilon = 0.5;
    correct_test=0; 
    for i = 1 : N_test
        if( abs_errors_test(1, i) < epsilon)
            correct_test = correct_test + 1;
        end
    end

    % Compute MSE & accuracy
    MSE_training = (abs_errors_training.^2 * ones(N_train, 1))/ N_train;
    accuracy_training = correct_training / N_train;
    fprintf('No. of centers: %i. Training sample accuracy = %f%%. MSE = %f. \n', M, accuracy_training*100, MSE_training);
    MSE_testing = (abs_errors_test.^2 * ones(N_test, 1))/ N_test;
    accuracy_test = correct_test / N_test;
    fprintf('No. of centers: %i. Test sample accuracy = %f%%. MSE = %f. \n', M, accuracy_test*100, MSE_testing);
    
    results_data{m, 1} = M;
    results_data{m, 2} = MSE_training;
    results_data{m, 3} = MSE_testing;
    results_data{m, 4} = accuracy_training;
    results_data{m, 5} = accuracy_test;
    results_data{m, 6} = y_training_outcome;
    results_data{m, 7} = y_test_outcome;
end

%% Create visuals
% find the smallest MSE_testing and it's index
MSEs_test = cell2mat(results_data(:,cat(1,3)));
[M,I] = min(MSEs_test);
% get the row of data
best_classifier_data = results_data(cat(1,I),:);

% plot train sample confusion matrix    
figure(1)
plotconfusion(y_train, best_classifier_data{1,6})
title(sprintf('Confusion matrix of the best classifier (M=%i) performing on training sample', best_classifier_data{1}))

% plot test sample confusion matrix
figure(2)
plotconfusion(y_test, best_classifier_data{1,7})
title(sprintf('Confusion matrix of the best classifier (M=%i) performing on testing sample', best_classifier_data{1}))

% plot train_MSEs and test_MSEs against Ms
MSEs_train = cell2mat(results_data(:,cat(1,2)));
Ms = cell2mat(results_data(:,cat(1,1)));
figure(3)
semilogx(Ms,MSEs_train,'-go',Ms,MSEs_test,'-ro')
title('Mean-square error of test and training samples against number of centers')
xlabel('Number of centers')
ylabel('Mean-square error')
legend('Training','Testing')
grid on

% plot accuracy_trainings and accuracy_tests against Ms
accuracy_trainings = cell2mat(results_data(:,cat(1,4)));
accuracy_tests = cell2mat(results_data(:,cat(1,5)));
figure(4)
semilogx(Ms,accuracy_trainings,'-go',Ms,accuracy_tests,'-ro')
title('Accuracies of test and training samples against number of centers')
xlabel('Number of centers')
ylabel('Accuracy')
legend('Training','Testing')
grid on

