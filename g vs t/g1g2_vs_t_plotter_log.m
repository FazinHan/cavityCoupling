% Load data from CSV file
filename = 'combined_plots_params.csv';
data = readtable(filename);

% Extract columns
t = data.t;
g1 = data.g1;
g2 = data.g2;

% Define logarithmic model: y = a * log(b * x) + c
logarithmicModel = fittype('a * log(b * x) + c', 'independent', 'x', 'coefficients', {'a', 'b', 'c'});

% Set initial guesses for g1 and g2 fits
a_g1_guess = 1; % Change this value for g1 initial guess for a
b_g1_guess = 1; % Change this value for g1 initial guess for b
c_g1_guess = 0; % Change this value for g1 initial guess for c
a_g2_guess = 1; % Change this value for g2 initial guess for a
b_g2_guess = 1; % Change this value for g2 initial guess for b
c_g2_guess = 0; % Change this value for g2 initial guess for c

% Set options for trust-region fitting
options = fitoptions('Method', 'NonlinearLeastSquares', 'Robust', 'on', 'StartPoint', [a_g1_guess, b_g1_guess, c_g1_guess]);

% Fit g1 data
[g1_fit, g1_gof] = fit(t, g1, logarithmicModel, options);

% Update options for g2 with new initial guesses
options.StartPoint = [a_g2_guess, b_g2_guess, c_g2_guess];

% Fit g2 data
[g2_fit, g2_gof] = fit(t, g2, logarithmicModel, options);

% Display fit parameters and R-squared for g1
disp('g1 Logarithmic Fit:');
coeff_g1 = coeffvalues(g1_fit);
rsq_g1 = g1_gof.rsquare;
fprintf('a = %.2f, b = %.2f, c = %.2f, R^2 = %.2f\n', coeff_g1(1), coeff_g1(2), coeff_g1(3), rsq_g1);

% Display fit parameters and R-squared for g2
disp('g2 Logarithmic Fit:');
coeff_g2 = coeffvalues(g2_fit);
rsq_g2 = g2_gof.rsquare;
fprintf('a = %.2f, b = %.2f, c = %.2f, R^2 = %.2f\n', coeff_g2(1), coeff_g2(2), coeff_g2(3), rsq_g2);

% Evaluate the fits over the data range
t_fit = linspace(min(t), max(t), 100);
g1_fit_vals = g1_fit.a * log(g1_fit.b * t_fit) + g1_fit.c;
g2_fit_vals = g2_fit.a * log(g2_fit.b * t_fit) + g2_fit.c;

% Plot data and fits
figure;
hold on;

% g1 data and fit
scatter(t, g1, 'b', 'DisplayName', 'g1 data');
plot(t_fit, g1_fit_vals, 'b--', 'DisplayName', sprintf('g1 fit: y = %.2f * log(%.2f * x) + %.2f', coeff_g1(1), coeff_g1(2), coeff_g1(3)));

% g2 data and fit
scatter(t, g2, 'r', 'DisplayName', 'g2 data');
plot(t_fit, g2_fit_vals, 'r--', 'DisplayName', sprintf('g2 fit: y = %.2f * log(%.2f * x) + %.2f', coeff_g2(1), coeff_g2(2), coeff_g2(3)));

% Customize plot
xlabel('t');
ylabel('g values');
title(sprintf('Logarithmic Fits for g1 and g2 (R^2: g1 = %.2f, g2 = %.2f)', rsq_g1, rsq_g2));
legend('show');
grid on;
hold off;
