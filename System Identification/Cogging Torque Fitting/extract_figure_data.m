%{
Adam Welker     Utah Robotics Center    Summer 25

extract_figure_data.m -- extracts data from a matlab figure on
the mtc cogging model
%}

new_fig = openfig('fit_8_1_update.fig');

a = get(gca,"Children");
xdata = get(a,"XData");
ydata = get(a,"YData");


% Extract data
new_angle_data = cell2mat(xdata(4));
new_torque_data = cell2mat(ydata(4));

% Extract Confidence intervales
CI_X1 = cell2mat(xdata(1))';
CI_Y1 = cell2mat(ydata(1))';

CI_X2 = cell2mat(xdata(2))';
CI_Y2 = cell2mat(ydata(2))';

% Extract the Fit data
fit_x = cell2mat(xdata(3))';
fit_y = cell2mat(ydata(3))';

save('fit_81.mat')