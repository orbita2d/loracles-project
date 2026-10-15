#!/usr/bin/env bash
# On the owl/butterfly pod: push one finished student straight to the main pod.
t=$1; cd ~/sp
tar cf - loras/$t logs/train_$t.log | ssh -i /root/.ssh/k_main -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 40990 root@62.169.159.178 "cd ~/sp && tar xf - && echo PUSHED $t"
