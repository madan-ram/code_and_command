#!/bin/bash
### BEGIN INIT INFO
# Provides:          pasties
# Required-Start:    $all
# Required-Stop:
# Default-Start:     2 3 4 5
# Default-Stop:
# Short-Description: Service to download pasties
### END INIT INFO
#

# Add date to existing output
adddate() {
    while IFS= read -r line; do
        echo "$(date) $line"
    done
}

# List all the process created by this scripts
list_descendants ()
{
  local children=$(ps -o pid= --ppid "$1")

  for pid in $children
  do
    list_descendants "$pid"
  done

  echo "$children"
}

case "$1" in 
start)
    if [ -e /var/run/pasties.pid ]; then
      echo pasties.sh is already running, pid=`cat /var/run/pasties.pid`
    else
      cd /var/www/pastie_scrapper
      mkdir /var/log/pasties
      sudo python pystemon.py | adddate >> /var/log/pasties/access.log 2>&1 &
      echo $(list_descendants $$)>/var/run/pasties.pid
      echo pasties.sh is running, pid=`cat /var/run/pasties.pid`
      exit 1
   fi
   ;;
stop)
   kill `cat /var/run/pasties.pid`
   rm /var/run/pasties.pid
   ;;
restart)
   $0 stop
   $0 start
   ;;
status)
   if [ -e /var/run/pasties.pid ]; then
      echo pasties.sh is running, pid=`cat /var/run/pasties.pid`
   else
      echo pasties.sh is NOT running
      exit 1
   fi
   ;;
*)
   echo "Usage: $0 {start|stop|status|restart}"
esac

exit 0
