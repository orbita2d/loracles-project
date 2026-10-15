# source me: pod 'cmd'  /  podcp src dst
SSHO="-i /home/claude/.sky/generated/ssh-keys/jlens-qual.key -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes -o LogLevel=ERROR -o ServerAliveInterval=30"
pod() { ssh -T $SSHO -p 40990 root@62.169.159.178 "$@"; }
podcp() { scp $SSHO -P 40990 "$@"; }
