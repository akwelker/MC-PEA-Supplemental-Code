new_mag_fig = openfig('new_figure_fit.fig');

surface_PPI = 267;
set(new_mag_fig, 'Units', 'inches');
set(new_mag_fig, 'Position', [0, 0, 2.5, 2.5]);
xlabel('Motor Angle (rad)', 'FontSize', 9);
ylabel('Motor Current (amp)', 'FontSize', 9);
title('');
xlim([-0.5,0.5])
ylim([-10,10])

set(gca,'Fontsize',9)
legend('Motor Current', 'Function Fit', "99% CI",'FontSize',8);
grid()
saveas(new_mag_fig, 'new_mag_fig.png');
hgsave(new_mag_fig, 'new_mag_fig.fig', '-v7');