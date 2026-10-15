# source me, then: podn <name> 'cmd' / podncp <name> src... dst   (addresses in pods.txt: name ip port)
SSHO="-i $HOME/.sky/generated/ssh-keys/jlens-qual.key -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes -o LogLevel=ERROR -o ServerAliveInterval=30 -o ConnectTimeout=15"
_addr() { awk -v n="$1" '$1==n {print $2, $3}' "$(dirname "${BASH_SOURCE[0]}")/pods.txt"; }
podn() { local a; a=($(_addr "$1")); shift; ssh -T $SSHO -p "${a[1]}" root@"${a[0]}" "$@"; }
podncp() { local a; a=($(_addr "$1")); shift; local args=("$@"); args=("${args[@]/#@:/root@${a[0]}:}"); scp $SSHO -P "${a[1]}" "${args[@]}"; }
