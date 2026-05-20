set terminal png size 800,600
set output 'benchmark_plot.png'
set title 'Execution Time vs Data Size (MySuperFastMiner_v1)'
set xlabel 'Data Size (MB)'
set ylabel 'Execution Time (ms)'
plot '-' with linespoints title 'Exec Time'
1.00 0.76
10.00 8.32
50.00 40.87
100.00 79.07
e
