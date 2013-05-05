%% Machine Learning Online Class
%  Exercise 1: Linear regression with multiple variables
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear regression exercise. 
%
%  You will need to complete the following functions in this 
%  exericse:
%
%     warmUpExercise.m
%     plotData.m
%     gradientDescent.m
%     computeCost.m
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m
%
%  For this part of the exercise, you will need to change some
%  parts of the code below for various experiments (e.g., changing
%  learning rates).
%

%% Initialization

%% ================ Part 1: Feature Normalization ================

%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

fprintf('Program paused. Press enter to continue.\n');
pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];


%% ================ Part 2: Gradient Descent ================

% ====================== YOUR CODE HERE ======================
% Instructions: We have provided you with the following starter
%               code that runs gradient descent with a particular
%               learning rate (alpha). 
%
%               Your task is to first make sure that your functions - 
%               computeCost and gradientDescent already work with 
%               this starter code and support multiple variables.
%
%               After that, try running gradient descent with 
%               different values of alpha and see which one gives
%               you the best result.
%
%               Finally, you should complete the code at the end
%               to predict the price of a 1650 sq-ft, 3 br house.
%
% Hint: By using the 'hold on' command, you can plot multiple
%       graphs on the same figure.
%
% Hint: At prediction, make sure you do the same feature normalization.
%

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alphas = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003];
num_iters = 100;

% Set up a figure to plot cost curves for each alpha
figure;
title('Cost history for each possible learning rate (alpha)');
xlabel('Number of iterations');
ylabel('Cost J');
hold on;
colors = jet(length(alphas));

% Run Gradient Descent for each alpha, plot the results, and pick the winning alpha
winning_alpha = alphas(1);
theta = zeros(3, 1);
J_history = Inf(num_iters, 1);

for i = 1:length(alphas),
	alpha = alphas(i);
	color = colors(i,:);
    res_theta = zeros(3, 1);
    res_J_history = ones(num_iters, 1);
	[res_theta, res_J_history] = gradientDescentMulti(X, y, res_theta, alpha, num_iters);

	plot(1:numel(res_J_history), res_J_history, '-b', 'LineWidth', 2, 'Color', color);
	fprintf(' J = %f at alpha = %f \n', res_J_history(length(J_history)), alpha);

	if (J_history(1) == Inf) || ((res_J_history(1) - res_J_history(2)) > (J_history(1) - J_history(2))),
		winning_alpha = alpha;
		theta = res_theta;
		J_history = res_J_history;
	end	
end

legend(num2str(alphas'));
hold off;

% Display winning alpha
fprintf('Winning alpha: \n');
fprintf(' %f \n', winning_alpha);
fprintf('\n');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.
price = 0; % You should change this

v = [1650 3]; % new example house (v)
v = (v .- mu) ./ sigma; %normalize v using remembered mu and sigma from training set
price = [1, v] * theta;

% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 3: Normal Equations ================

fprintf('Solving with normal equations...\n');

% ====================== YOUR CODE HERE ======================
% Instructions: The following code computes the closed form 
%               solution for linear regression using the normal
%               equations. You should complete the code in 
%               normalEqn.m
%
%               After doing so, you should complete this code 
%               to predict the price of a 1650 sq-ft, 3 br house.
%

%% Load Data
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');


% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
price = 0; % You should change this

v = [1650 3]; % new example house (v)
price = [1, v] * theta;

% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price);

