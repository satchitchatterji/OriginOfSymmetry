#!/bin/bash

if [ "$1" == '--gnuplot' ]
then
  process(){
    clear
    gnuplot -e "
      set terminal dumb;
      set key above;
      set xlabel 'Samples';
      set ylabel 'CPU';
      set y2label 'MEM';
      set yrange [0:*];
      set y2range [0:*];
      set y2tics;
      plot '$log' using 0:2 with lines title 'cpu',
              '' using 0:3 with lines axes x1y2 title 'mem';
      "
    tail -n1 $log
    sleep 1
  }
else
  process(){
    tail -n1 $log
    sleep 1
  }
fi

log=memory.log
while :
do
  (
  date -Iseconds | tr '\n' ' '
  ps -u $USER -o "%cpu=,%mem=" \
  | awk '{cpu += $1; mem += $2; }
     END {print cpu, mem; }'
  ) >> $log
  process
done
