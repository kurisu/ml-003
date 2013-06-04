function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

predictions = X*theta;     % predictions of hypothesis on all m examples
sqErrors = (predictions-y).^2; % squared errors

J = (1/(2*m)) * sum(sqErrors) + ((lambda/(2*m))*sum(theta(2:length(theta)).^2));

grad_1 = (1/m) * X(:,1)'*(X*theta - y);
grad_rest = ((1/m) * X'*(X*theta - y))(2:length(theta)) + (lambda/m)*theta(2:length(theta));

grad = [grad_1;grad_rest];

% =========================================================================

grad = grad(:);

end
