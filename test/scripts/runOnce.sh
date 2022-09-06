{ ./runVastCCOnce.sh ; } > /dev/null
res=$(echo $?)
if [ $res = "0" ]; then
	./runVastOptOnce.sh
fi

