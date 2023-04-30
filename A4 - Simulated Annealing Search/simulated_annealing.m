clear; clc;
% Defining function to minimize
easom = @(x) -cos(x(1))*cos(x(2))*exp(-((x(1)-pi)^2+(x(2)-pi)^2));

% Annealing options
init_temperature = 100;
alpha = 1;
annealing_schedule = @(t) t-alpha;

% Starting Point
x0 = [-100 100];

% options = optimoptions(@simulannealbnd,'TemperatureFcn',annealing_schedule ...
%                                       ,'InitialTemperature',init_temperature ...
%                                       ,'FunctionTolerance',0)

[x,fval,exitflag,output] = simulannealbnd(easom,x0);