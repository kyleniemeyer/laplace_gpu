reset
set terminal postscript eps enhanced "Helvetica" 20 color

unset logscale

set pm3d map
#set contour base
set view map
set key outside

set size square

set xrange [0 : 1]
set yrange [0 : 1]
set isosamples 50
set palette rgbformulae 22,13,10
set cbrange [0:1]

set output "plot_cpu.eps"
splot "temp_cpu.dat" using 1:2:3 notitle

set output "plot_gpu.eps"
splot "temp_gpu.dat" using 1:2:3 notitle
replot